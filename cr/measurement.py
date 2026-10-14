import re
import uuid
from dataclasses import dataclass
from threading import RLock

import numpy as np

from .game import MeasuredChannel
from .prompts import Exposure, receiver_prompt
from .storage import append_json, signature


class BudgetExhausted(RuntimeError):
    pass


class ReceiverChanged(RuntimeError):
    pass


def parse_posterior(text, states, sum_tolerance=0.05):
    pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    for line in text.splitlines():
        found = re.findall(pattern, line)
        if len(found) >= states:
            vector = np.asarray([float(value) for value in found[:states]])
            if not np.isfinite(vector).all() or np.any(vector < 0) or np.any(vector > 1):
                raise ValueError("Posterior report contains invalid probabilities")
            total = float(vector.sum())
            if total <= 0 or abs(total - 1) > sum_tolerance + 1e-12:
                raise ValueError("Posterior report exceeds the normalization tolerance")
            return vector / total
    raise ValueError("Posterior report has insufficient state coordinates")


def parse_action(text, actions):
    token = re.match(r"\s*[\"'\(\[\{]*\s*([^\s.,;:!?\"'\)\]\}]+)", text)
    if token is None:
        raise ValueError("Receiver action report has no label")
    first = token.group(1).casefold()
    matched = [index for index, label in enumerate(actions) if label.split()[0].casefold() == first]
    if len(matched) != 1:
        raise ValueError("Receiver action report has no unambiguous declared label")
    return matched[0]


class QueryLedger:
    def __init__(self, limit, path=None):
        if isinstance(limit, bool) or int(limit) != limit or limit < 0:
            raise ValueError("Receiver-query budget is a nonnegative integer")
        self.limit = int(limit)
        self.path = path
        self.ledger_id = uuid.uuid4().hex
        self.records = []
        self.charged = 0
        self.shared = 0
        self.stage_limits = {}
        self.stage_counts = {}
        self.completed_pairs = []
        self.completed_measurements = 0
        self.imported_accounts = {}
        self.lock = RLock()

    def set_stage_limit(self, phase, round_id, limit):
        if isinstance(limit, bool) or int(limit) != limit or limit < 0:
            raise ValueError("Stage-query allowance is a nonnegative integer")
        self.stage_limits[(phase, round_id)] = int(limit)

    def remaining(self, phase=None, round_id=None):
        value = self.limit - self.charged
        if phase is not None and (phase, round_id) in self.stage_limits:
            value = min(value, self.stage_limits[(phase, round_id)] - self.stage_counts.get((phase, round_id), 0))
        return max(value, 0)

    def begin(self, record):
        with self.lock:
            phase, round_id = record["phase"], record["round"]
            shared = phase == "shared_pool"
            if not shared and self.remaining(phase, round_id) == 0:
                raise BudgetExhausted("Receiver-query allowance is exhausted")
            if shared:
                self.shared += 1
            else:
                self.charged += 1
                key = (phase, round_id)
                self.stage_counts[key] = self.stage_counts.get(key, 0) + 1
            result = dict(record, call=len(self.records) + 1, charged=not shared, status="started")
            self.records.append(result)
            if self.path:
                append_json(self.path, dict(result, event="request_started"))
            return result

    def finish(self, record, **fields):
        with self.lock:
            record.update(fields)
            if self.path:
                append_json(self.path, dict(record, event="request_finished"))

    def complete_pair(self, message, phase, round_id, repeat, receiver_signature):
        with self.lock:
            record = {"event": "pair_completed", "message_id": message.identifier,
                      "message_fingerprint": message.fingerprint, "phase": phase, "round": round_id,
                      "repeat": repeat, "receiver_signature": receiver_signature, "charged": phase != "shared_pool"}
            self.completed_pairs.append(record)
            if self.path:
                append_json(self.path, record)

    def complete_measurement(self, message, phase, round_id, repeats, receiver_signature):
        with self.lock:
            self.completed_measurements += 1
            if self.path:
                append_json(self.path, {"event": "measurement_completed", "message_id": message.identifier,
                                        "phase": phase, "round": round_id, "repeats": repeats,
                                        "receiver_signature": receiver_signature})

    def import_initial_account(self, account, archive_signature):
        if account is None:
            self.imported_accounts.setdefault(archive_signature, {"archive_signature": archive_signature, "raw_calls": None})
            return
        identifier = account.get("ledger_id") or archive_signature
        calls = account.get("issued_raw_calls", account.get("total_raw_calls", account.get("attempts")))
        if calls is None or isinstance(calls, bool) or int(calls) != calls or calls < 0:
            raise ValueError("Imported initial measurement account has a nonnegative raw-call count")
        previous = self.imported_accounts.get(identifier, {})
        if previous.get("raw_calls") is None or calls >= previous["raw_calls"]:
            self.imported_accounts[identifier] = {"archive_signature": archive_signature, "raw_calls": int(calls),
                                                  "ledger_id": account.get("ledger_id"), "original_query_account": account}
        for nested in account.get("imported_initial_accounts", []):
            source = nested["archive_signature"]
            original = nested.get("original_query_account")
            if original is None and nested["raw_calls"] is not None:
                original = {"ledger_id": nested.get("ledger_id"), "issued_raw_calls": nested["raw_calls"]}
            self.import_initial_account(original, source)

    def summary(self):
        imports = list(self.imported_accounts.values())
        known_imports = all(item["raw_calls"] is not None for item in imports)
        imported_calls = sum(item["raw_calls"] for item in imports) if known_imports else None
        return {"limit": self.limit, "charged_calls": self.charged, "shared_calls": self.shared,
                "ledger_id": self.ledger_id, "budget_scope": "per_method_excluding_one_off_shared_pool",
                "issued_raw_calls": len(self.records), "imported_initial_calls": imported_calls,
                "end_to_end_raw_calls": len(self.records) + imported_calls if known_imports else None,
                "imported_initial_accounts": imports,
                "attempts": len(self.records), "successful_responses": sum(row["status"] == "parsed" for row in self.records),
                "completed_pairs": len(self.completed_pairs), "completed_measurements": self.completed_measurements,
                "charged_pairs": sum(row["charged"] for row in self.completed_pairs),
                "shared_pairs": sum(not row["charged"] for row in self.completed_pairs)}


@dataclass(frozen=True)
class Measurement:
    message_id: str
    message_fingerprint: str
    posterior: np.ndarray
    action: int
    receiver_signature: str
    posterior_samples: np.ndarray
    action_samples: tuple
    phase: str
    round_id: int
    data_kind: str

    @property
    def posterior_standard_error(self):
        if len(self.posterior_samples) < 2:
            return None
        return self.posterior_samples.std(axis=0, ddof=1) / np.sqrt(len(self.posterior_samples))


class MeasurementEngine:
    def __init__(self, game, backend, ledger, seed=0, retries=2,
                 headline_temperature=0.0, repeat_temperature=0.3):
        if retries < 0 or int(retries) != retries or seed < 0:
            raise ValueError("Invalid receiver retry count or measurement seed")
        self.game, self.backend, self.ledger = game, backend, ledger
        self.seed, self.retries = int(seed), int(retries)
        self.headline_temperature, self.repeat_temperature = float(headline_temperature), float(repeat_temperature)
        if not np.isfinite([self.headline_temperature, self.repeat_temperature]).all() or min(self.headline_temperature, self.repeat_temperature) < 0:
            raise ValueError("Receiver decoding temperatures are finite and nonnegative")

    @property
    def checkpoint_signature(self):
        return signature({"backend": self.backend.identity, "game": self.game.record(),
                          "headline_temperature": self.headline_temperature, "repeat_temperature": self.repeat_temperature,
                          "protocol": "separate_posterior_action_v1"})

    def temperature(self, repeats):
        return self.headline_temperature if repeats == 1 else self.repeat_temperature

    def response_signature(self, exposure, repeats=1):
        return signature({"checkpoint": self.checkpoint_signature, "exposure": exposure,
                          "temperature": self.temperature(repeats), "repeats": repeats,
                          "posterior_aggregation": "arithmetic_mean", "action_aggregation": "modal_index_ties"})

    def request(self, message, component, phase, round_id, repeat, exposure, temperature, repeats):
        prompts = receiver_prompt(self.game, message, component, exposure)
        for retry in range(self.retries + 1):
            request_seed = int(signature({"seed": self.seed, "message": message.fingerprint, "component": component,
                                          "phase": phase, "round": round_id, "repeat": repeat, "retry": retry})[:8], 16) % (2 ** 31)
            record = self.ledger.begin({"message_id": message.identifier, "message_fingerprint": message.fingerprint,
                                         "component": component, "phase": phase, "round": round_id,
                                         "repeat": repeat, "retry": retry, "seed": request_seed,
                                         "temperature": temperature, "receiver_signature": self.response_signature(exposure, repeats),
                                         "input_signature": signature(prompts)})
            try:
                response = self.backend.complete(prompts, temperature=temperature, seed=request_seed)
            except Exception as error:
                self.ledger.finish(record, status="backend_error", error_type=type(error).__name__)
                continue
            text = response.get("text") if isinstance(response, dict) else response
            if not isinstance(text, str):
                self.ledger.finish(record, status="parse_error", error_type="nontext_response")
                continue
            if record["receiver_signature"] != self.response_signature(exposure, repeats):
                self.ledger.finish(record, status="receiver_changed")
                raise ReceiverChanged("Receiver changed during a paired measurement")
            try:
                result = (parse_posterior(text, len(self.game.prior)) if component == "posterior"
                          else parse_action(text, self.game.actions))
            except (ValueError, TypeError, OverflowError):
                self.ledger.finish(record, status="parse_error", text=text)
                continue
            self.ledger.finish(record, status="parsed", text=text,
                               usage=response.get("usage", {}) if isinstance(response, dict) else {})
            return result
        return None

    def measure(self, message, phase, repeats=1, round_id=0, exposure=None):
        if isinstance(repeats, bool) or int(repeats) != repeats or repeats < 1:
            raise ValueError("Receiver repeat count is a positive integer")
        exposure = exposure or Exposure()
        expected_signature = self.response_signature(exposure, repeats)
        temperature = self.temperature(repeats)
        beliefs, actions = [], []
        for repeat in range(repeats):
            posterior = self.request(message, "posterior", phase, round_id, repeat, exposure, temperature, repeats)
            if posterior is None:
                return None
            action = self.request(message, "action", phase, round_id, repeat, exposure, temperature, repeats)
            if expected_signature != self.response_signature(exposure, repeats):
                raise ReceiverChanged("Receiver changed between paired measurement components")
            if action is None:
                return None
            beliefs.append(posterior)
            actions.append(action)
            self.ledger.complete_pair(message, phase, round_id, repeat, expected_signature)
        observations = np.asarray(beliefs)
        action = int(np.argmax(np.bincount(actions, minlength=len(self.game.actions))))
        self.ledger.complete_measurement(message, phase, round_id, repeats, expected_signature)
        return Measurement(message.identifier, message.fingerprint, observations.mean(axis=0), action,
                           expected_signature, observations, tuple(actions), phase, round_id,
                           getattr(self.backend, "data_kind", "measured_receiver"))

    def pool(self, messages, phase, repeats=1, round_id=0, exposure=None):
        records, retained = [], []
        expected_signature = self.response_signature(exposure or Exposure(), repeats)
        for message in messages:
            reading = self.measure(message, phase, repeats, round_id, exposure)
            if self.response_signature(exposure or Exposure(), repeats) != expected_signature:
                raise ReceiverChanged("Receiver changed during a pool measurement")
            if reading is not None:
                retained.append(message)
                records.append(reading)
        return retained, records

    def channel(self, measurements):
        signatures = {record.receiver_signature for record in measurements}
        if len(signatures) != 1:
            raise ValueError("A channel has one receiver checkpoint and exposure protocol")
        return MeasuredChannel(self.game, tuple(record.message_id for record in measurements),
                               [record.posterior for record in measurements],
                               np.asarray([record.action for record in measurements]), next(iter(signatures)))
