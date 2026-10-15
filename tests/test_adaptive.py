from types import SimpleNamespace

import numpy as np
import pytest

from cr.adaptive import AdaptiveSender
from cr.measurement import MeasurementEngine, QueryLedger
from cr.messages import Message
from cr.prompts import Exposure
from cr.storage import read_channel

from test_program import game


class ReceiverFixture:
    identity = {"kind": "software_fixture", "version": 0}
    data_kind = "software_fixture"

    def complete(self, messages, temperature, seed):
        text = messages[1]["content"]
        against = "MESSAGE\nanchor_against" in text
        favorable = "MESSAGE\nanchor_for" in text
        if "posterior probability" in messages[0]["content"]:
            return "1 0" if against else "0 1" if favorable else ".5 .5"
        return "against" if against else "for"


class GeneratorFixture:
    backend = SimpleNamespace(identity={"kind": "software_fixture"})

    def propose(self, contexts, count, seed, free=False):
        context, proposition = contexts[0]
        return [Message("candidate_" + str(seed), "candidate", context, proposition)]

    def proposed_policy(self, messages):
        result = np.zeros((2, len(messages)))
        target = next(index for index, message in enumerate(messages) if message.text == "anchor_for")
        result[:, target] = 1
        return result


def catalog():
    return [Message("against", "anchor_against", "context", "topic"),
            Message("for", "anchor_for", "context", "topic")]


def test_loop_confirms_independently_monitors_proposal_and_reconstructs_final_policy(tmp_path):
    instance = game()
    engine = MeasurementEngine(instance, ReceiverFixture(), QueryLedger(100), seed=2)
    settings = {"initial_pool": 2, "candidate_batch": 1, "rounds": 1, "acquisition_calls": 10,
                "confirmation_repeats": 2, "refresh_repeats": 1, "final_repeats": 2, "seed": 2}
    runner = AdaptiveSender(instance, catalog(), GeneratorFixture(), engine, settings, tmp_path / "run")
    result = runner.run()
    assert result["status"] == "ready"
    assert result["deployed_policy_audit"]["compatible"]
    assert result["rounds"] == 1 and runner.records[0]["admitted"] == 1
    assert runner.records[0]["cached_residual_norm"] == 1
    assert runner.records[0]["refreshed"]
    channel, archive = read_channel(tmp_path / "run/channel.json")
    np.testing.assert_allclose(result["solution"].marginal @ channel.posteriors, instance.prior)
    assert all(reading.phase == "final" for reading in runner.measurements)
    assert archive["metadata"]["data_kind"] == "software_fixture"
    phases = {row["phase"] for row in engine.ledger.records}
    assert {"acquisition", "confirmation", "refresh", "final"}.issubset(phases)


def test_compatible_incumbent_does_not_disable_proposal_trigger(tmp_path):
    instance = game()
    engine = MeasurementEngine(instance, ReceiverFixture(), QueryLedger(100))
    runner = AdaptiveSender(instance, catalog(), GeneratorFixture(), engine, {"initial_pool": 2}, tmp_path / "run")
    solution = runner.initialize()
    trigger, norm, changed = runner.refresh_required(solution.marginal, 0)
    assert not trigger and norm == 0 and not changed
    policy = runner.generator.proposed_policy(runner.messages)
    trigger, norm, changed = runner.refresh_required(instance.prior @ policy, 0)
    assert trigger and norm == 1


def test_query_exhaustion_returns_measurement_required_and_no_stale_deployment(tmp_path):
    instance = game()
    engine = MeasurementEngine(instance, ReceiverFixture(), QueryLedger(0))
    runner = AdaptiveSender(instance, catalog(), GeneratorFixture(), engine,
                             {"initial_pool": 2, "rounds": 0, "final_repeats": 2}, tmp_path / "run", method="static_mr")
    result = runner.run()
    assert result["status"] == "additional_measurement_required"
    assert result["reason"] == "receiver_budget_exhausted"
    assert "policy" not in result


def test_direct_generator_reports_its_own_deployed_value_and_compatibility(tmp_path):
    instance = game()
    engine = MeasurementEngine(instance, ReceiverFixture(), QueryLedger(100))
    runner = AdaptiveSender(instance, catalog(), GeneratorFixture(), engine,
                             {"initial_pool": 2, "rounds": 0, "final_repeats": 1}, tmp_path / "run", method="direct_generator")
    result = runner.run()
    assert result["status"] == "ready"
    assert not result["deployed_policy_audit"]["compatible"]
    assert result["deployed_policy_audit"]["sender_utility"] == 1
    assert result["optimized_mixture_value"] == .5


def test_feasibility_repair_generates_messages_for_an_exhausted_context_pool(tmp_path):
    class Generator(GeneratorFixture):
        def propose(self, contexts, count, seed, free=False):
            context, proposition = contexts[0]
            return [Message("generated_for", "anchor_for", context, proposition)]

    engine = MeasurementEngine(game(), ReceiverFixture(), QueryLedger(6))
    runner = AdaptiveSender(game(), catalog()[:1], Generator(), engine,
                            {"initial_pool": 1, "rounds": 0, "final_repeats": 1},
                            tmp_path / "run")
    result = runner.run()
    assert result["status"] == "ready"
    assert result["deployed_policy_audit"]["compatible"]
    assert [message.text for message in runner.messages] == ["anchor_against", "anchor_for"]
    assert engine.ledger.charged == 6 and engine.ledger.shared == 2
    assert [row["phase"] for row in engine.ledger.records].count("repair") == 2


def test_feasibility_repair_stops_when_generation_repeats_a_known_message(tmp_path):
    class Generator(GeneratorFixture):
        def propose(self, contexts, count, seed, free=False):
            context, proposition = contexts[0]
            return [Message("duplicate", "anchor_against", context, proposition)]

    engine = MeasurementEngine(game(), ReceiverFixture(), QueryLedger(10))
    runner = AdaptiveSender(game(), catalog()[:1], Generator(), engine,
                            {"initial_pool": 1}, tmp_path / "run")
    assert runner.initialize() is None
    assert runner.repair_batches == 1 and engine.ledger.charged == 0


def test_feasibility_repair_rejects_a_candidate_from_another_context(tmp_path):
    class Generator(GeneratorFixture):
        def propose(self, contexts, count, seed, free=False):
            return [Message("other", "anchor_for", "other context", "other topic")]

    engine = MeasurementEngine(game(), ReceiverFixture(), QueryLedger(10))
    runner = AdaptiveSender(game(), catalog()[:1], Generator(), engine,
                            {"initial_pool": 1}, tmp_path / "run")
    with pytest.raises(ValueError, match="conditioning context"):
        runner.initialize()


@pytest.mark.parametrize("method", ["static_mr", "random_expansion"])
def test_pool_baselines_do_not_use_generated_feasibility_candidates(method, tmp_path):
    class Generator(GeneratorFixture):
        def propose(self, *args, **kwargs):
            raise AssertionError("The pool baseline does not generate feasibility candidates")

    engine = MeasurementEngine(game(), ReceiverFixture(), QueryLedger(10))
    released = catalog() if method == "static_mr" else catalog()[:1]
    runner = AdaptiveSender(game(), released, Generator(), engine,
                            {"initial_pool": 1}, tmp_path / "run", method=method)
    assert runner.initialize() is None
    assert len(runner.messages) == 1 and engine.ledger.charged == 0


def test_random_pool_baseline_can_repair_with_remaining_released_messages(tmp_path):
    class Generator(GeneratorFixture):
        def propose(self, *args, **kwargs):
            raise AssertionError("The pool baseline uses its released catalog")

    engine = MeasurementEngine(game(), ReceiverFixture(), QueryLedger(2))
    runner = AdaptiveSender(game(), catalog(), Generator(), engine,
                            {"initial_pool": 1}, tmp_path / "run", method="random_expansion")
    solution = runner.initialize()
    np.testing.assert_allclose(solution.marginal @ runner.channel().posteriors, game().prior)
    assert len(runner.messages) == 2 and engine.ledger.charged == 2
    assert engine.ledger.shared == 2
