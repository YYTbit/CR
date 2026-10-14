import numpy as np
import pytest

from cr.movement import (action_preserving_reinterpretation, binary_action_capacity, binary_action_interval,
                         controlled_transfer, drift_audit, local_stability_radius, minimum_reinterpretation)

from test_program import game


@pytest.mark.parametrize("marginal", [[.2, .8], [0, 1], [.5, .5]])
def test_constructive_reinterpretation_attains_exact_weighted_l1_bound(marginal):
    beliefs = np.eye(2)
    result = minimum_reinterpretation([.5, .5], marginal, beliefs)
    np.testing.assert_allclose(np.asarray(marginal) @ result["posteriors"], [.5, .5])
    assert result["movement"] == pytest.approx(result["minimum"])
    assert np.all(result["posteriors"] >= 0)
    np.testing.assert_allclose(result["posteriors"].sum(axis=1), 1)


def test_constructive_multi_state_transport_keeps_zero_mass_rows():
    beliefs = np.array([[.8, .1, .1], [.1, .7, .2], [.2, .2, .6]])
    marginal = np.array([.8, .2, 0])
    result = minimum_reinterpretation([.2, .3, .5], marginal, beliefs)
    np.testing.assert_allclose(marginal @ result["posteriors"], [.2, .3, .5])
    np.testing.assert_array_equal(result["posteriors"][2], beliefs[2])
    assert result["movement"] == pytest.approx(np.linalg.norm(marginal @ beliefs - [.2, .3, .5], 1))


def test_binary_capacity_closed_boundaries_and_assumption_validation():
    instance = game()
    intervals = [binary_action_interval(instance.receiver_utility, index) for index in [0, 1, 2]]
    np.testing.assert_allclose(intervals, [[0, .4], [.4, .6], [.6, 1]])
    result = binary_action_capacity([.5, .5], [.5, .5], [[.9, .1], [.3, .7]], [intervals[0], intervals[2]])
    assert result["feasible"] and result["minimum_movement"] == pytest.approx(.2)
    np.testing.assert_allclose(np.array([.5, .5]) @ result["posteriors"], [.5, .5])
    blocked = binary_action_capacity([.2, .8], [1], [[.5, .5]], [intervals[1]])
    assert not blocked["feasible"] and blocked["posteriors"] is None
    with pytest.raises(ValueError):
        binary_action_capacity([.5, .5], [1], [[.5, .5]], [intervals[2]])


def test_general_action_program_matches_binary_capacity_and_blocks_impossible_actions():
    instance = game()
    beliefs = [[.9, .1], [.3, .7]]
    result = action_preserving_reinterpretation(instance, [.5, .5], beliefs, np.array([0, 2]))
    assert result["feasible"] and result["movement"] == pytest.approx(.2)
    np.testing.assert_allclose(np.array([.5, .5]) @ result["posteriors"], instance.prior)
    blocked = action_preserving_reinterpretation(game(.8), [1], [[.5, .5]], np.array([1]))
    assert not blocked["feasible"]


def test_drift_bound_uses_residual_difference_and_recorded_error_bounds():
    old = np.array([[.9, .1], [.1, .9]])
    fresh = np.array([[.8, .2], [.3, .7]])
    result = drift_audit([.5, .5], [.2, .8], old, fresh, [.02, .03], [.01, .02])
    assert result["posterior_movement"] >= result["residual_difference_bound"]
    assert result["error_adjusted_lower_bound"] == pytest.approx(max(result["residual_difference_bound"] - .046, 0))
    assert result["statistical_error_bounds_supplied"]


def test_local_radius_and_controlled_transfer_respect_their_domain():
    assert local_stability_radius([.5, .5], np.eye(2), .2)["maximum_residual"] == pytest.approx(.2)
    np.testing.assert_allclose(controlled_transfer([.4, .6], 0, 1, .2), [.6, .4])
    with pytest.raises(ValueError):
        controlled_transfer([.4, .6], 0, 1, .7)
    with pytest.raises(ValueError):
        local_stability_radius([.1, .9], np.eye(2), .3)
