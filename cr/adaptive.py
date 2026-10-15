from dataclasses import replace

import numpy as np

from .learning import acquisition_reward
from .measurement import BudgetExhausted, ReceiverChanged
from .messages import common_context
from .movement import drift_audit, residual
from .program import InfeasiblePool, hull_distance, policy_marginal, price_decomposition, solve_sender, validate_policy
from .prompts import Exposure
from .storage import append_json, signature, write_channel, write_json


def aligned_proposal(record, game, messages):
    identifiers = [message.identifier for message in messages]
    recorded = record["message_ids"]
    if len(set(recorded)) != len(recorded) or not set(recorded).issubset(identifiers):
        raise ValueError("Proposed policy support has unmeasured or duplicate message identifiers")
    supplied = validate_policy(record["policy"], len(game.prior), len(recorded))
    policy = np.zeros((len(game.prior), len(messages)))
    for index, identifier in enumerate(recorded):
        policy[:, identifiers.index(identifier)] = supplied[:, index]
    return validate_policy(policy, len(game.prior), len(messages))


class AdaptiveSender:
    def __init__(self, game, released_messages, generator, engine, settings, output,
                 trainer=None, method="dual_guided", proposals=None, initial_measurements=None):
        from pathlib import Path
        self.game, self.catalog, self.generator, self.engine = game, list(released_messages), generator, engine
        self.settings, self.output = dict(settings), Path(output)
        self.trainer, self.method, self.proposals = trainer, method, list(proposals or [])
        if not self.catalog or len({message.identifier for message in self.catalog}) != len(self.catalog):
            raise ValueError("Released message catalog is nonempty with distinct identifiers")
        self.context, self.proposition = common_context(self.catalog)
        self.group_id = self.catalog[0].group_id
        if self.output.exists() and any(self.output.iterdir()):
            raise ValueError("A new adaptive run requires an empty output directory")
        self.output.mkdir(parents=True, exist_ok=True)
        self.messages, self.measurements = [], []
        self.exposure = Exposure(self.settings.get("exposure", "blind"), tuple(self.settings.get("history", [])))
        if self.exposure.kind == "announced":
            if "initial_policy" not in self.settings:
                raise ValueError("Initial announced exposure requires a declared sender policy")
            value = self.settings["initial_policy"]
            self.exposure = Exposure("announced", (), tuple(value["message_ids"]), value["policy"])
        self.initial_measurements = initial_measurements
        self.records = []
        self.receiver_checkpoint = self.engine.checkpoint_signature
        self.repair_attempts = set()
        self.repair_batches = 0
        self.pool_repeats = 1

    def channel(self):
        return self.engine.channel(self.measurements)

    def coefficients(self, channel):
        return channel.classical_payoffs if self.method == "posterior_value" else channel.payoffs

    def solve(self):
        channel = self.channel()
        return solve_sender(channel, self.coefficients(channel))

    def validate_frozen_receiver(self, solution, reading=None):
        expected = self.engine.response_signature(self.exposure, self.pool_repeats)
        if self.engine.checkpoint_signature != self.receiver_checkpoint or solution.receiver_signature != expected:
            raise ReceiverChanged("Frozen prices have a different receiver or exposure protocol")
        if reading is not None and reading.receiver_signature != solution.receiver_signature:
            raise ReceiverChanged("Candidate reading has a different protocol from the frozen dual")

    def add(self, messages, measurements):
        if len(messages) != len(measurements):
            raise ValueError("Admission has incomplete paired observations")
        known_ids = {message.identifier for message in self.messages}
        known_text = {message.fingerprint for message in self.messages}
        for message, measurement in zip(messages, measurements):
            if message.group_id != self.group_id:
                raise ValueError("A candidate belongs to a different context-conditioned channel")
            if message.identifier != measurement.message_id or message.fingerprint != measurement.message_fingerprint:
                raise ValueError("Measured candidate identity differs from its proposed message")
            if message.identifier in known_ids or message.fingerprint in known_text:
                continue
            if self.measurements and measurement.receiver_signature != self.measurements[0].receiver_signature:
                raise ValueError("Pool expansion changes the frozen response protocol")
            self.messages.append(message)
            self.measurements.append(measurement)
            known_ids.add(message.identifier)
            known_text.add(message.fingerprint)

    def persist_pool(self):
        if self.messages:
            write_channel(self.output / "pool.json", self.game, self.messages, self.measurements,
                          {"group_id": self.group_id, "pool_repeats": self.pool_repeats,
                           "data_kind": getattr(self.engine.backend, "data_kind", "measured_receiver")})

    def measure_batch(self, messages, phase, repeats, round_id=0, exposure=None):
        return self.engine.pool(messages, phase, repeats, round_id, exposure or self.exposure)

    def repair(self, round_id):
        while self.messages:
            try:
                return self.solve()
            except InfeasiblePool:
                distance = hull_distance(self.game.prior, self.channel().posteriors)
                append_json(self.output / "feasibility.jsonl", {"round": round_id, "hull_distance": distance,
                                                                "within_hull_tolerance": distance <= self.settings.get("hull_distance_tolerance", 1e-6),
                                                                "exact_program_feasible": False})
                if self.method == "static_mr":
                    return None
                seen = {message.identifier for message in self.messages}
                current = self.engine.response_signature(self.exposure, self.pool_repeats)
                remaining = [message for message in self.catalog if message.identifier not in seen
                             and (current, message.identifier) not in self.repair_attempts]
                count = min(int(self.settings.get("repair_batch", 32)),
                            self.engine.ledger.remaining("repair", round_id) // (2 * self.pool_repeats))
                if count < 1:
                    return None
                if remaining:
                    batch = remaining[:count]
                else:
                    if self.method == "random_expansion":
                        return None
                    # Feasibility repair can expand a context's pool after its released messages are exhausted.
                    seed = int(self.settings.get("seed", 0)) + 2000000 + self.repair_batches
                    self.repair_batches += 1
                    candidates = self.generator.propose([(self.context, self.proposition)], count, seed)
                    fingerprints = {message.fingerprint for message in self.messages}
                    batch = []
                    for message in candidates:
                        if message.group_id != self.group_id:
                            raise ValueError("Feasibility candidates require the current conditioning context")
                        if (message.identifier in seen or message.fingerprint in fingerprints
                                or (current, message.identifier) in self.repair_attempts):
                            continue
                        batch.append(message)
                        seen.add(message.identifier)
                        fingerprints.add(message.fingerprint)
                    if not batch:
                        return None
                self.repair_attempts.update((current, message.identifier) for message in batch)
                messages, readings = self.measure_batch(batch, "repair", self.pool_repeats, round_id)
                if not messages:
                    return None
                self.add(messages, readings)
                self.persist_pool()
        return None

    def initialize(self):
        generator = np.random.default_rng(self.settings.get("seed", 0))
        count = min(int(self.settings.get("initial_pool", 32)), len(self.catalog))
        if count < 1:
            raise ValueError("Initial message pool is nonempty")
        selected = [self.catalog[index] for index in generator.permutation(len(self.catalog))[:count]]
        self.repair_attempts.update((self.engine.response_signature(self.exposure), message.identifier) for message in selected)
        if self.initial_measurements is not None:
            records = {reading.message_id: reading for reading in self.initial_measurements}
            if any(message.identifier not in records for message in selected):
                raise ValueError("Recorded initial channel omitted a selected released message")
            readings = [records[message.identifier] for message in selected]
            if any(reading.receiver_signature != self.engine.response_signature(self.exposure) for reading in readings):
                raise ValueError("Recorded initial channel has a different receiver protocol")
            self.add(selected, readings)
        else:
            messages, readings = self.measure_batch(selected, "shared_pool", 1)
            self.add(messages, readings)
        self.persist_pool()
        return self.repair(0)

    def proposed_policy(self, round_id):
        if self.proposals:
            if round_id >= len(self.proposals):
                raise ValueError("The declared proposal schedule ended before the round limit")
            return aligned_proposal(self.proposals[round_id], self.game, self.messages)
        return self.generator.proposed_policy(self.messages)

    def refresh_required(self, proposed_marginal, round_id, proposed_policy=None):
        channel = self.channel()
        norm = float(np.linalg.norm(residual(self.game.prior, proposed_marginal, channel.posteriors), 1))
        checkpoint_changed = self.engine.checkpoint_signature != self.receiver_checkpoint
        announcement_changed = False
        if self.exposure.kind == "announced":
            if proposed_policy is None:
                raise ValueError("Announced-policy refresh checks require the proposed conditional policy")
            announcement_changed = signature(self.exposure) != signature(Exposure("announced", (), tuple(message.identifier for message in self.messages), proposed_policy))
        rule = "never" if self.method == "cached_response" else self.settings.get("refresh_rule", "residual")
        if rule == "never":
            trigger = False
        elif rule == "continual":
            trigger = True
        elif rule == "periodic":
            interval = int(self.settings.get("refresh_interval", 2))
            if interval < 1:
                raise ValueError("Periodic refresh interval is positive")
            trigger = (round_id + 1) % interval == 0
        elif rule == "residual":
            trigger = norm > self.settings.get("residual_threshold", 0.041)
        else:
            raise ValueError("Unknown receiver-refresh rule")
        return bool(trigger or checkpoint_changed or announcement_changed), norm, checkpoint_changed

    def refresh(self, policy, round_id):
        old = self.channel()
        marginal = policy_marginal(self.game.prior, policy)
        kind = self.settings.get("exposure", "blind")
        exposure = (Exposure("announced", (), tuple(message.identifier for message in self.messages), policy)
                    if kind == "announced" else self.exposure)
        repeats = int(self.settings.get("refresh_repeats", 4))
        messages, readings = self.measure_batch(self.messages, "refresh", repeats,
                                                 round_id, exposure)
        if not messages:
            return None
        self.messages, self.measurements, self.exposure = messages, readings, exposure
        self.pool_repeats = repeats
        self.receiver_checkpoint = self.engine.checkpoint_signature
        self.persist_pool()
        fresh = self.channel()
        if fresh.message_ids == old.message_ids:
            audit = drift_audit(self.game.prior, marginal, old.posteriors, fresh.posteriors)
            audit["action_change_mass"] = float(marginal @ (old.actions != fresh.actions))
            audit["cached_map_error"] = float(abs(marginal @ old.payoffs - marginal @ fresh.payoffs))
            return audit
        return {"drift_audit_available": False, "dropped_messages": list(set(old.message_ids) - set(fresh.message_ids))}

    def acquire(self, solution, contexts, round_id, limit, candidates=None):
        self.engine.ledger.set_stage_limit("acquisition", round_id, limit)
        count = int(self.settings.get("candidate_batch", 16))
        seed = self.settings.get("seed", 0) + 1000000 + 10000 * round_id
        if candidates is not None:
            candidates = list(candidates)
        elif self.method == "random_expansion":
            known = {message.identifier for message in self.messages}
            available = [message for message in self.catalog if message.identifier not in known]
            order = np.random.default_rng(seed).permutation(len(available))[:count]
            candidates = [available[index] for index in order]
        else:
            candidates = self.generator.propose(contexts, count, seed)
        seen = {message.fingerprint for message in self.messages}
        scored = []
        for message in candidates:
            self.validate_frozen_receiver(solution)
            if message.fingerprint in seen:
                continue
            seen.add(message.fingerprint)
            try:
                reading = self.engine.measure(message, "acquisition", self.pool_repeats, round_id, self.exposure)
            except BudgetExhausted:
                break
            if reading is None:
                continue
            self.validate_frozen_receiver(solution, reading)
            payoff = (self.game.classical_value(reading.posterior) if self.method == "posterior_value"
                      else acquisition_reward(self.game, reading.posterior, reading.action))
            reduced = float(solution.reduced_costs(reading.posterior[None, :], [payoff])[0])
            scored.append((message, reading, reduced))
        return scored

    def confirm(self, scored, solution, round_id):
        admitted = 0
        for message, acquisition, predicted in scored:
            self.validate_frozen_receiver(solution, acquisition)
            if self.method == "shared_measurement":
                confirmation = replace(acquisition, phase="shared_admission")
                independent = False
            else:
                confirmation = self.engine.measure(message, "confirmation", int(self.settings.get("confirmation_repeats", 4)),
                                                      round_id, self.exposure)
                independent = True
            if confirmation is None:
                continue
            self.validate_frozen_receiver(solution)
            payoff = (self.game.classical_value(confirmation.posterior) if self.method == "posterior_value"
                      else acquisition_reward(self.game, confirmation.posterior, confirmation.action))
            confirmed = float(solution.reduced_costs(confirmation.posterior[None, :], [payoff])[0])
            passed = confirmed > self.settings.get("admission_threshold", 0.01)
            canonical, canonical_cost, accepted = None, None, False
            if passed:
                if self.method == "shared_measurement" and confirmation.receiver_signature == self.engine.response_signature(self.exposure, self.pool_repeats):
                    canonical = confirmation
                else:
                    canonical = self.engine.measure(message, "admission_pool", self.pool_repeats, round_id, self.exposure)
                if canonical is not None:
                    self.validate_frozen_receiver(solution, canonical)
                    value = (self.game.classical_value(canonical.posterior) if self.method == "posterior_value"
                             else acquisition_reward(self.game, canonical.posterior, canonical.action))
                    canonical_cost = float(solution.reduced_costs(canonical.posterior[None, :], [value])[0])
                    accepted = canonical_cost > self.settings.get("admission_threshold", 0.01)
                    if accepted:
                        self.add([message], [canonical])
                        self.persist_pool()
                        admitted += 1
            append_json(self.output / "admission.jsonl", {"round": round_id, "message": message,
                                                         "acquisition_reduced_cost": predicted,
                                                         "confirmed_reduced_cost": confirmed,
                                                         "canonical_reduced_cost": canonical_cost,
                                                         "confirmation_passed": passed,
                                                         "independent_confirmation": independent,
                                                         "frozen_dual_signature": signature({"lambda": solution.state_prices,
                                                                                              "nu": solution.normalization_price}),
                                                         "accepted": accepted, "confirmation": confirmation,
                                                         "canonical_measurement": canonical})
        return admitted

    def incomplete(self, reason):
        self.persist_pool()
        result = {"status": "additional_measurement_required", "reason": reason,
                  "method": self.method, "query_account": self.engine.ledger.summary(), "rounds": self.records}
        write_json(self.output / "result.json", result)
        return result

    def prepare(self):
        write_json(self.output / "metadata.json", {"game": self.game.record(), "settings": self.settings,
                                                   "method": self.method, "receiver": self.engine.backend.identity,
                                                   "generator": self.generator.backend.identity,
                                                   "released_catalog_signature": signature(self.catalog), "group_id": self.group_id,
                                                   "initial_query_account": self.engine.ledger.summary()})

    def finish(self, rounds, save_adapter=True):
        final_messages, final_measurements = self.measure_batch(self.messages, "final", int(self.settings.get("final_repeats", 8)), rounds)
        if not final_messages:
            return self.incomplete("final_measurement_unparseable")
        self.messages, self.measurements = final_messages, final_measurements
        self.pool_repeats = int(self.settings.get("final_repeats", 8))
        try:
            final = self.solve()
        except InfeasiblePool:
            return self.incomplete("final_pool_infeasible")
        channel = self.channel()
        write_channel(self.output / "channel.json", self.game, self.messages, self.measurements,
                      {"data_kind": getattr(self.engine.backend, "data_kind", "measured_receiver"),
                       "measurement_stage": "independent_final", "group_id": self.group_id,
                       "query_account": self.engine.ledger.summary()})
        write_json(self.output / "messages.json", self.messages)
        policy = self.generator.proposed_policy(self.messages) if self.method == "direct_generator" else final.policy
        from .evaluation import evaluate_policy
        policy_audit = evaluate_policy(channel, policy)
        result = {"status": "ready", "method": self.method, "solution": final, "policy": policy,
                  "policy_definition": "finite_pool_canonical_sequence_scores" if self.method == "direct_generator" else "compatible_reconstruction",
                  "price": price_decomposition(channel), "query_account": self.engine.ledger.summary(),
                  "optimized_mixture_value": float(final.marginal @ channel.payoffs),
                  "deployed_policy_audit": policy_audit, "rounds": len(self.records), "group_id": self.group_id}
        write_json(self.output / "result.json", result)
        if save_adapter and self.trainer is not None:
            self.generator.backend.save_adapter(self.output / "sender_adapter.npz")
        return result

    def run(self):
        self.prepare()
        try:
            solution = self.initialize()
            if solution is None:
                return self.incomplete("initial_pool_infeasible")
            contexts = list(dict.fromkeys((message.context, message.proposition) for message in self.catalog))
            rounds = 0 if self.method == "static_mr" else int(self.settings.get("rounds", 4))
            if rounds < 0:
                raise ValueError("Expansion-round limit is nonnegative")
            total_acquisition = int(self.settings.get("acquisition_calls", 4096))
            for round_id in range(rounds):
                solution = self.solve()
                if self.engine.checkpoint_signature != self.receiver_checkpoint:
                    if self.refresh(solution.policy, round_id) is None:
                        return self.incomplete("changed_receiver_measurement_failed")
                    solution = self.repair(round_id)
                    if solution is None:
                        return self.incomplete("changed_receiver_pool_infeasible")
                stage_limit = total_acquisition // rounds + int(round_id < total_acquisition % rounds)
                training = []
                if self.trainer is not None:
                    reserve = 2 * self.pool_repeats * int(self.settings.get("candidate_batch", 16))
                    training = self.trainer.update(contexts, solution, self.exposure, round_id,
                                                   self.method == "payoff_guided", max(stage_limit - reserve, 0),
                                                   self.method == "posterior_value", self.pool_repeats)
                scored = self.acquire(solution, contexts, round_id, stage_limit)
                admitted = self.confirm(scored, solution, round_id)
                policy = self.proposed_policy(round_id)
                marginal = policy_marginal(self.game.prior, policy)
                trigger, norm, changed = self.refresh_required(marginal, round_id, policy)
                audit = self.refresh(policy, round_id) if trigger else None
                if trigger and audit is None:
                    return self.incomplete("receiver_refresh_failed")
                solution = self.repair(round_id)
                if solution is None:
                    return self.incomplete("refreshed_pool_infeasible")
                record = {"round": round_id, "training": training, "admitted": admitted, "pool_size": len(self.messages),
                          "proposed_policy": policy, "proposed_marginal": marginal, "cached_residual_norm": norm,
                          "refreshed": trigger, "receiver_checkpoint_changed": changed, "drift_audit": audit,
                          "solution": solution, "query_account": self.engine.ledger.summary()}
                self.records.append(record)
                write_json(self.output / ("round_" + str(round_id).zfill(3) + ".json"), record)
                if self.engine.ledger.remaining() == 0:
                    break
            return self.finish(rounds)
        except BudgetExhausted:
            return self.incomplete("receiver_budget_exhausted")
        except ReceiverChanged:
            return self.incomplete("receiver_changed_during_measurement")
