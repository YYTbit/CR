import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cr.adaptive import AdaptiveSender
from cr.backends import ApiBackend
from cr.evaluation import GroupedMixtureSender, MixtureSender, evaluate_native
from cr.measurement import BudgetExhausted, MeasurementEngine, QueryLedger
from cr.messages import Message, context_group
from cr.models import Completion, LanguageGenerator
from cr.movement import binary_action_capacity
from cr.resource import ResourceSender
from cr.storage import read_channel, read_jsonl

from test_adaptive import GeneratorFixture, ReceiverFixture, catalog
from test_program import game


def test_effective_decoding_protocol_is_part_of_channel_identity():
    engine = MeasurementEngine(game(), ReceiverFixture(), QueryLedger(100))
    message = Message("a", "candidate", "context")
    headline = engine.measure(message, "acquisition", 1)
    repeated = engine.measure(message, "confirmation", 4)
    assert headline.receiver_signature != repeated.receiver_signature
    with pytest.raises(ValueError):
        engine.channel([headline, repeated])
    account = engine.ledger.summary()
    assert account["attempts"] == 10 and account["completed_pairs"] == 5
    assert account["completed_measurements"] == 2


def test_accepted_confirmation_gets_a_fresh_canonical_pool_reading(tmp_path):
    engine = MeasurementEngine(game(), ReceiverFixture(), QueryLedger(100))
    runner = AdaptiveSender(game(), catalog(), GeneratorFixture(), engine,
                             {"initial_pool": 2, "confirmation_repeats": 4}, tmp_path / "run")
    solution = runner.initialize()
    candidate = Message("new", "candidate", "context", "topic")
    acquisition = engine.measure(candidate, "acquisition")
    assert runner.confirm([(candidate, acquisition, .5)], solution, 0) == 1
    assert len({reading.receiver_signature for reading in runner.measurements}) == 1
    assert runner.measurements[-1].phase == "admission_pool"
    assert len(runner.measurements[-1].posterior_samples) == 1
    record = read_jsonl(tmp_path / "run/admission.jsonl")[0]
    assert record["accepted"] and record["confirmation_passed"]
    assert record["confirmation"]["receiver_signature"] != record["canonical_measurement"]["receiver_signature"]


def test_budget_exhaustion_keeps_prior_committed_admissions_and_matching_records(tmp_path):
    engine = MeasurementEngine(game(), ReceiverFixture(), QueryLedger(8))
    runner = AdaptiveSender(game(), catalog(), GeneratorFixture(), engine,
                             {"initial_pool": 2, "confirmation_repeats": 1}, tmp_path / "run")
    solution = runner.initialize()
    candidates = [Message(str(index), "candidate " + str(index), "context", "topic") for index in range(2)]
    scored = [(message, engine.measure(message, "acquisition"), .5) for message in candidates]
    with pytest.raises(BudgetExhausted):
        runner.confirm(scored, solution, 0)
    assert candidates[0] in runner.messages and candidates[1] not in runner.messages
    channel, archive = read_channel(tmp_path / "run/pool.json")
    assert candidates[0].identifier in channel.message_ids
    records = read_jsonl(tmp_path / "run/admission.jsonl")
    assert len(records) == 1 and records[0]["accepted"]


def test_query_start_is_durable_before_dispatch_and_unfinished_attempt_is_visible(tmp_path):
    path = tmp_path / "queries.jsonl"
    ledger = QueryLedger(1, path)
    record = ledger.begin({"phase": "acquisition", "round": 0})
    events = read_jsonl(path)
    assert len(events) == 1 and events[0]["event"] == "request_started"
    assert events[0]["status"] == "started" and events[0]["charged"]
    ledger.finish(record, status="parsed")
    assert read_jsonl(path)[1]["event"] == "request_finished"
    assert ledger.summary()["attempts"] == 1
    with pytest.raises(BudgetExhausted):
        ledger.begin({"phase": "acquisition", "round": 0})


def test_imported_cumulative_and_chained_accounts_preserve_each_source_ledger():
    first = {"ledger_id": "first", "issued_raw_calls": 10}
    middle = QueryLedger(10)
    middle.import_initial_account(first, "first_archive")
    middle.import_initial_account(dict(first, issued_raw_calls=6), "first_archive")
    row = middle.begin({"phase": "acquisition", "round": 0})
    middle.finish(row, status="parsed")
    latest = QueryLedger(10)
    latest.import_initial_account(middle.summary(), "middle_archive")
    latest.import_initial_account(first, "first_archive")
    latest.import_initial_account(middle.summary(), "middle_archive")
    summary = latest.summary()
    assert summary["issued_raw_calls"] == 0
    assert summary["imported_initial_calls"] == 11
    assert summary["end_to_end_raw_calls"] == 11
    assert len(summary["imported_initial_accounts"]) == 2


def test_unknown_imported_cost_stays_unknown_through_archive_reuse():
    middle = QueryLedger(10)
    middle.import_initial_account(None, "unaccounted_archive")
    latest = QueryLedger(10)
    latest.import_initial_account(middle.summary(), "middle_archive")
    assert latest.summary()["imported_initial_calls"] is None
    assert latest.summary()["end_to_end_raw_calls"] is None
    assert any(row["raw_calls"] is None for row in latest.summary()["imported_initial_accounts"])


def test_legacy_accounts_without_ledger_identifiers_remain_separate():
    ledger = QueryLedger(10)
    ledger.import_initial_account({"ledger_id": None, "attempts": 3}, "first_archive")
    ledger.import_initial_account({"ledger_id": None, "attempts": 4}, "second_archive")
    assert ledger.summary()["imported_initial_calls"] == 7


def test_endpoint_identity_distinguishes_services_without_publishing_credentials():
    first = ApiBackend("same_alias", "version", "https://first.example/v1/chat/completions?key=secret")
    second = ApiBackend("same_alias", "version", "https://second.example/v1/chat/completions")
    try:
        assert first.identity != second.identity
        assert "secret" not in json.dumps(first.identity)
    finally:
        first.close()
        second.close()


def test_small_nonzero_action_residual_is_adjusted_and_capacity_does_not_allow_extra_mass():
    p = .5 - 1e-12
    result = binary_action_capacity([.5, .5], [1], [[1 - p, p]], [[.4, .6]])
    assert result["feasible"] and result["posteriors"][0, 1] != p
    assert result["mean_constraint_residual"] == 0
    assert result["movement"] == pytest.approx(result["minimum_movement"], abs=1e-15)
    blocked = binary_action_capacity([.4 - 1e-12, .6 + 1e-12], [1], [[.5, .5]], [[.4, .6]])
    assert not blocked["feasible"]


def test_single_program_rejects_mixed_conditioning_contexts(tmp_path):
    messages = catalog() + [Message("other", "different text", "other context", "other topic")]
    with pytest.raises(ValueError):
        AdaptiveSender(game(), messages, GeneratorFixture(), MeasurementEngine(game(), ReceiverFixture(), QueryLedger(10)), {}, tmp_path / "run")
    with pytest.raises(ValueError):
        MixtureSender(game(), messages, np.ones((2, 3)) / 3)


def test_resource_uses_separate_programs_and_one_query_budget(tmp_path):
    messages = catalog() + [Message("other_" + message.identifier, message.text, "other context", "other topic") for message in catalog()]
    engine = MeasurementEngine(game(), ReceiverFixture(), QueryLedger(8))
    runner = ResourceSender(game(), messages, GeneratorFixture(), engine,
                            {"initial_pool": 2, "final_repeats": 1}, tmp_path / "run", method="static_mr")
    result = runner.run()
    assert result["status"] == "ready" and len(result["groups"]) == 2
    assert result["query_account"]["charged_calls"] == 8
    assert result["query_account"]["shared_calls"] == 8
    assert result["query_account"]["completed_pairs"] == 8
    for identifier, group in runner.groups.items():
        channel, archive = read_channel(group.output / "channel.json")
        assert all(message.group_id == identifier for message in group.messages)
        assert result["groups"][identifier]["deployed_policy_audit"]["compatible"]


def test_resource_expansion_keeps_new_candidates_in_their_conditioned_program(tmp_path):
    messages = catalog() + [Message("other_" + message.identifier, message.text, "other context", "other topic") for message in catalog()]
    engine = MeasurementEngine(game(), ReceiverFixture(), QueryLedger(100))
    settings = {"initial_pool": 2, "candidate_batch": 1, "rounds": 1, "acquisition_calls": 10,
                "confirmation_repeats": 2, "refresh_repeats": 1, "final_repeats": 1}
    runner = ResourceSender(game(), messages, GeneratorFixture(), engine, settings, tmp_path / "run")
    result = runner.run()
    assert result["status"] == "ready" and result["rounds"] == 1
    assert sum(record["admitted"] for record in runner.records[0]["groups"].values()) == 1
    assert all(value["deployed_policy_audit"]["compatible"] for value in result["groups"].values())
    assert all(len({message.group_id for message in group.messages}) == 1 for group in runner.groups.values())


def test_checkpoint_change_before_acquisition_returns_incomplete_without_stale_dual(tmp_path):
    receiver = ReceiverFixture()
    engine = MeasurementEngine(game(), receiver, QueryLedger(100))
    class Generator(GeneratorFixture):
        def propose(self, contexts, count, seed, free=False):
            receiver.identity = {"kind": "software_fixture", "version": 1}
            return super().propose(contexts, count, seed, free)
    runner = AdaptiveSender(game(), catalog(), Generator(), engine,
                             {"initial_pool": 2, "candidate_batch": 1, "rounds": 1, "acquisition_calls": 10}, tmp_path / "run")
    result = runner.run()
    assert result["status"] == "additional_measurement_required"
    assert result["reason"] == "receiver_changed_during_measurement"
    assert "policy" not in result and engine.ledger.charged == 0


class NativeFixture:
    def sender_context(self, record):
        return record["context"]

    def proposition(self, record):
        return record["proposition"]

    def sender_state(self, record):
        return record["state"]

    def score(self, record, text):
        assert text == record["expected"]
        return {"data_kind": "measured_run", "score": 1}


def test_native_grouped_policy_matches_context_and_rejects_unmeasured_items():
    pools = [[Message("a", "first", "first context", "first topic")],
             [Message("b", "second", "second context", "second topic")]]
    groups = {pool[0].group_id: {"messages": [{"identifier": pool[0].identifier, "text": pool[0].text,
                                             "context": pool[0].context, "proposition": pool[0].proposition}],
                               "policy": [[1], [1]]} for pool in pools}
    sender = GroupedMixtureSender(game(), groups)
    records = [{"context": pool[0].context, "proposition": pool[0].proposition, "state": "true", "expected": pool[0].text}
               for pool in pools]
    records[0]["state"] = 0
    records[1]["state"] = 1
    assert len(evaluate_native(NativeFixture(), records, sender, game())) == 2
    with pytest.raises(ValueError):
        sender.select(NativeFixture(), {"context": "unmeasured", "proposition": "topic", "state": 0}, 0)


def test_native_generator_receives_actual_private_state():
    class Generator:
        def propose(self, contexts, count, seed, states):
            assert states == [1]
            return [Message("a", "informed", contexts[0][0], contexts[0][1])]
    record = {"context": "context", "proposition": "topic", "state": 1, "expected": "informed"}
    assert evaluate_native(NativeFixture(), [record], Generator(), game())[0]["score"] == 1


def test_finite_pool_scores_terminated_sequences_under_one_context():
    class Tokenizer:
        eos_token_id = 2

        def encode(self, text, add_special_tokens):
            return [5] if text == "one" else [6]
    class Backend:
        tokenizer = Tokenizer()
        max_completion_tokens = 3

        def __init__(self):
            self.sequences = []
            self.prompts = []

        def encode(self, prompt):
            self.prompts.append(prompt)
            return (1, 3)

        def token_log_probabilities(self, completion):
            self.sequences.append(completion.completion_ids)
            return torch.tensor([-.2, -.3])
    backend = Backend()
    generator = LanguageGenerator(backend, game())
    pool = [Message("a", "one", "context", "topic"), Message("b", "two", "context", "topic")]
    np.testing.assert_allclose(generator.proposed_policy(pool), np.full((2, 2), .5))
    assert all(sequence[-1] == 2 for sequence in backend.sequences)
    assert "PRIVATE OBSERVED STATE" in backend.prompts[0][1]["content"]
    with pytest.raises(ValueError):
        generator.proposed_policy(pool + [Message("c", "three", "other context", "topic")])


def test_informed_proposals_preserve_sampled_tokens_and_state():
    class Backend:
        identity = {"kind": "software_fixture"}

        def sample(self, prompt, temperature, seed):
            assert "PRIVATE OBSERVED STATE" in prompt[1]["content"]
            return Completion((1, 3), (5, 2), "informed", seed)
    generator = LanguageGenerator(Backend(), game())
    message = generator.propose([("context", "topic")], 1, 7, states=[1])[0]
    assert message.metadata["sampled_state"] == game().states[1]
    assert message.metadata["completion_ids"] == (5, 2)
