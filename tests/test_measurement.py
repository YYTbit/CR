import numpy as np
import pytest

from cr.measurement import (BudgetExhausted, MeasurementEngine, QueryLedger, ReceiverChanged,
                            parse_action, parse_posterior)
from cr.messages import Message
from cr.prompts import Exposure, receiver_prompt

from test_program import game


class BackendFixture:
    data_kind = "software_fixture"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.version = 0
        self.calls = []

    @property
    def identity(self):
        return {"kind": "software_fixture", "version": self.version}

    def complete(self, messages, temperature, seed):
        self.calls.append((messages, temperature, seed))
        return next(self.responses)


def test_parser_uses_probability_domain_and_first_action_token():
    np.testing.assert_allclose(parse_posterior("heading\n0.2 0.79", 2), [.2 / .99, .79 / .99])
    assert parse_action('"For" explanation', ("against", "undecided", "for")) == 2
    for value in ["-.1 1.1", ".8 .8", "nan .4", "0 0"]:
        with pytest.raises(ValueError):
            parse_posterior(value, 2)


def test_every_failed_attempt_and_separate_component_consumes_a_query():
    backend = BackendFixture(["invalid", ".3 .7", "unknown", "for"])
    ledger = QueryLedger(20)
    engine = MeasurementEngine(game(), backend, ledger, seed=2)
    result = engine.measure(Message("a", "message", "context"), "acquisition")
    assert result.action == 2 and ledger.charged == 4
    assert len(ledger.records) == 4
    assert [row["status"] for row in ledger.records] == ["parse_error", "parsed", "parse_error", "parsed"]
    assert len({row["seed"] for row in ledger.records}) == 4


def test_unparseable_messages_drop_after_three_attempts():
    backend = BackendFixture(["invalid"] * 3)
    ledger = QueryLedger(10)
    reading = MeasurementEngine(game(), backend, ledger).measure(Message("a", "message", "context"), "initial")
    assert reading is None and ledger.charged == 3


def test_incomplete_pair_at_budget_boundary_is_never_returned():
    engine = MeasurementEngine(game(), BackendFixture([".5 .5"]), QueryLedger(1))
    with pytest.raises(BudgetExhausted):
        engine.measure(Message("a", "message", "context"), "acquisition")
    assert engine.ledger.charged == 1


def test_independent_confirmation_and_repeat_aggregation():
    backend = BackendFixture([".2 .8", "for", ".3 .7", "for", ".4 .6", "undecided"])
    engine = MeasurementEngine(game(), backend, QueryLedger(20), seed=3)
    message = Message("a", "message", "context")
    acquisition = engine.measure(message, "acquisition")
    confirmation = engine.measure(message, "confirmation", 2)
    np.testing.assert_allclose(confirmation.posterior, [.35, .65])
    assert confirmation.action == 1
    assert engine.ledger.charged == 6
    assert engine.ledger.records[0]["seed"] != engine.ledger.records[2]["seed"]
    assert acquisition.phase != confirmation.phase
    assert confirmation.posterior_standard_error is not None


def test_receiver_checkpoint_change_invalidates_partial_pair():
    class ChangingBackend(BackendFixture):
        def complete(self, messages, temperature, seed):
            self.version += 1
            return ".5 .5"
    engine = MeasurementEngine(game(), ChangingBackend([]), QueryLedger(10))
    with pytest.raises(ReceiverChanged):
        engine.measure(Message("a", "message", "context"), "acquisition")
    assert engine.ledger.records[0]["status"] == "receiver_changed"


def test_shared_pool_cost_and_stage_caps_remain_distinct():
    ledger = QueryLedger(1)
    engine = MeasurementEngine(game(), BackendFixture([".5 .5", "for", ".5 .5"]), ledger)
    message = Message("a", "message", "context")
    engine.measure(message, "shared_pool")
    assert ledger.shared == 2 and ledger.charged == 0
    ledger.set_stage_limit("acquisition", 0, 1)
    with pytest.raises(BudgetExhausted):
        engine.measure(message, "acquisition")
    assert ledger.shared == 2 and ledger.charged == 1


def test_announced_exposure_has_public_likelihoods_and_preserves_private_state():
    instance, message = game(), Message("a", "message", "context")
    exposure = Exposure("announced", (), ("a", "b"), np.array([[.8, .2], [.3, .7]]))
    prompt = receiver_prompt(instance, message, "posterior", exposure)
    assert "[0.8, 0.3]" in prompt[1]["content"]
    assert "PRIVATE OBSERVED STATE" not in prompt[1]["content"]
