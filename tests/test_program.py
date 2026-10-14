import itertools

import numpy as np
import pytest

from cr.evaluation import evaluate_policy, optimization_gap, posterior_projection
from cr.game import Game, MeasuredChannel
from cr.messages import Message
from cr.program import InfeasiblePool, classical_sender_value, hull_distance, price_decomposition, solve_sender
from cr.storage import read_channel, write_channel


def game(prior=0.5):
    return Game(np.array([1 - prior, prior]), np.array([[0, 0], [.5, .5], [1, 1]]),
                np.array([[1, 0], [.6, .6], [0, 1]]), ("false", "true"), ("against", "undecided", "for"))


def test_primal_dual_policy_and_exhaustive_binary_mixtures():
    instance = game(.45)
    p = np.array([.05, .2, .45, .55, .8, 1.0])
    actions = np.array([0, 1, 1, 2, 2, 2])
    channel = MeasuredChannel(instance, tuple(map(str, range(len(p)))), np.column_stack((1 - p, p)), actions)
    solution = solve_sender(channel)
    values = []
    for first, second in itertools.combinations(range(len(p)), 2):
        if p[first] <= .45 <= p[second]:
            weight = (.45 - p[first]) / (p[second] - p[first])
            values.append((1 - weight) * channel.payoffs[first] + weight * channel.payoffs[second])
    values.append(channel.payoffs[2])
    assert solution.value == pytest.approx(max(values))
    assert solution.dual_value == pytest.approx(solution.value)
    assert len(solution.support) <= 2
    np.testing.assert_allclose(solution.policy.sum(axis=1), 1)
    np.testing.assert_allclose(instance.prior @ solution.policy, solution.marginal)
    np.testing.assert_allclose(solution.marginal @ channel.posteriors, instance.prior)
    assert max(solution.reduced_costs(channel.posteriors, channel.payoffs)) <= 1e-8
    assert evaluate_policy(channel, solution.policy)["compatible"]
    assert optimization_gap(channel, solution, solution.policy)["optimization_gap"] == pytest.approx(0)
    with pytest.raises(ValueError):
        solution.state_prices[0] = 10


def test_equal_posteriors_keep_distinct_measured_actions():
    channel = MeasuredChannel(game(), ("a", "b"), [[.5, .5], [.5, .5]], np.array([0, 2]))
    solution = solve_sender(channel)
    assert solution.value == 1 and solution.support == ("b",)


def test_infeasible_hull_and_exact_binary_distance():
    channel = MeasuredChannel(game(), ("a", "b"), [[.9, .1], [.8, .2]], np.array([0, 0]))
    with pytest.raises(InfeasiblePool):
        solve_sender(channel)
    assert hull_distance(channel.game.prior, channel.posteriors) == pytest.approx(np.sqrt(2) * .3)


@pytest.mark.parametrize("prior", [.1, .4, .6, .9])
def test_classical_ternary_value_and_sender_favorable_ties(prior):
    instance = game(prior)
    assert classical_sender_value(instance)["value"] == pytest.approx(min(prior / .6, 1))
    assert instance.classical_value([.6, .4]) == .5
    assert instance.classical_value([.4, .6]) == 1


def test_price_is_signed_and_decomposes_coverage_and_action_effects():
    channel = MeasuredChannel(game(.1), ("a",), [[.9, .1]], np.array([2]))
    result = price_decomposition(channel)
    assert result["price_of_language"] < 0
    assert not result["pointwise_best_response_bound"]
    assert result["coverage_gap"] + result["action_gap"] == pytest.approx(result["price_of_language"])


def test_general_states_and_zero_prior_state_removal():
    instance = Game(np.array([.2, .3, .5]), np.eye(3), np.eye(3), ("a", "b", "c"), ("x", "y", "z"))
    channel = MeasuredChannel(instance, ("a", "b", "c"), np.eye(3), np.array([0, 1, 2]))
    solution = solve_sender(channel)
    assert solution.value == pytest.approx(1)
    np.testing.assert_allclose(solution.policy, np.eye(3))
    retained = Game.from_record({"prior": [1, 0], "states": ["known", "absent"], "actions": ["x"],
                                 "sender_utility": [[1, 2]], "receiver_utility": [[3, 4]]})
    assert retained.states == ("known",) and retained.sender_utility.shape == (1, 1)


def test_channel_archive_and_nearest_posterior_projection(tmp_path):
    instance = game()
    channel = MeasuredChannel(instance, ("a", "b"), [[1, 0], [0, 1]], np.array([0, 2]), "fixture")
    messages = [Message("a", "first", "context"), Message("b", "second", "context")]
    records = [{"message_id": message.identifier, "message_fingerprint": message.fingerprint,
                "posterior": belief, "action": int(action), "receiver_signature": "fixture"}
               for message, belief, action in zip(messages, channel.posteriors, channel.actions)]
    write_channel(tmp_path / "channel.json", instance, messages, records,
                  {"data_kind": "software_fixture"})
    loaded, record = read_channel(tmp_path / "channel.json")
    np.testing.assert_allclose(loaded.posteriors, channel.posteriors)
    assert posterior_projection(channel, [[.1, .9]])["message_ids"] == ["b"]
