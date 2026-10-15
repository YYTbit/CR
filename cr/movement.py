import numpy as np
from scipy.optimize import linprog

from .game import probability_vector


def measured_map(prior, marginal, posteriors):
    prior = probability_vector(prior)
    beliefs = np.asarray(posteriors, dtype=float)
    marginal = probability_vector(marginal, len(beliefs))
    if beliefs.shape != (len(marginal), len(prior)) or not np.isfinite(beliefs).all():
        raise ValueError("Posterior map has incompatible dimensions")
    if np.any(beliefs < 0) or not np.allclose(beliefs.sum(axis=1), 1, atol=1e-9, rtol=0):
        raise ValueError("Posterior map rows are probability vectors")
    return prior, marginal, beliefs


def residual(prior, marginal, posteriors):
    prior, marginal, beliefs = measured_map(prior, marginal, posteriors)
    return marginal @ beliefs - prior


def minimum_reinterpretation(prior, marginal, posteriors):
    prior, marginal, beliefs = measured_map(prior, marginal, posteriors)
    joint = marginal[:, None] * beliefs
    mean = joint.sum(axis=0)
    difference = mean - prior
    surplus = np.maximum(difference, 0)
    deficit = np.maximum(-difference, 0)
    moved = float(surplus.sum())
    if moved <= 1e-15:
        return {"posteriors": beliefs.copy(), "movement": 0.0, "minimum": float(np.linalg.norm(difference, 1)),
                "residual": difference}
    removal_fraction = np.divide(surplus, mean, out=np.zeros_like(mean), where=mean > 0)
    removed = joint * removal_fraction
    # Rowwise redistribution preserves message mass while restoring the prior marginal.
    adjusted = joint - removed + removed.sum(axis=1)[:, None] * deficit[None, :] / moved
    updated = beliefs.copy()
    positive = marginal > 0
    updated[positive] = adjusted[positive] / marginal[positive, None]
    movement = float(np.sum(marginal[:, None] * np.abs(updated - beliefs)))
    if not np.allclose(marginal @ updated, prior, atol=1e-10, rtol=0) or np.any(updated < -1e-12):
        raise RuntimeError("Constructive reinterpretation failed its mean constraint")
    return {"posteriors": updated, "movement": movement, "minimum": float(np.linalg.norm(difference, 1)),
            "residual": difference}


def binary_action_interval(receiver_utility, action, tolerance=1e-12):
    utility = np.asarray(receiver_utility, dtype=float)
    if utility.ndim != 2 or utility.shape[1] != 2 or not 0 <= action < len(utility) or not np.isfinite(utility).all():
        raise ValueError("Binary action region has an action-by-two-state utility table")
    lower, upper = 0.0, 1.0
    for alternative in range(len(utility)):
        difference = utility[action] - utility[alternative]
        intercept, slope = difference[0], difference[1] - difference[0]
        if abs(slope) <= tolerance:
            if intercept < -tolerance:
                return None
        elif slope > 0:
            lower = max(lower, -intercept / slope)
        else:
            upper = min(upper, -intercept / slope)
    return (max(0.0, lower), min(1.0, upper)) if lower <= upper + tolerance else None


def binary_action_capacity(prior, marginal, posteriors, intervals, tolerance=1e-10):
    prior, marginal, beliefs = measured_map(prior, marginal, posteriors)
    intervals = np.asarray(intervals, dtype=float)
    if len(prior) != 2 or intervals.shape != (len(marginal), 2) or not np.isfinite(intervals).all():
        raise ValueError("Binary capacity requires one closed action interval per message")
    lower, upper = intervals.T
    p = beliefs[:, 1]
    if np.any(lower < 0) or np.any(upper > 1) or np.any(lower > upper):
        raise ValueError("Closed action intervals lie in the binary simplex")
    if np.any(p < lower - tolerance) or np.any(p > upper + tolerance):
        raise ValueError("The measured action-region assumption is violated")
    delta = float(prior[1] - marginal @ p)
    positive_capacity = float(marginal @ (upper - p))
    negative_capacity = float(marginal @ (p - lower))
    feasible = float(marginal @ lower) <= prior[1] <= float(marginal @ upper)
    updated = p.copy()
    if feasible and delta != 0:
        capacities = np.maximum(upper - p if delta > 0 else p - lower, 0)
        weighted = marginal * capacities
        if weighted.sum() <= 0:
            raise RuntimeError("Action-preserving capacity has no admissible direction")
        updated += np.sign(delta) * abs(delta) * capacities / weighted.sum()
        updated = np.clip(updated, lower, upper)
    constraint_residual = float(marginal @ updated - prior[1]) if feasible else None
    if feasible and abs(constraint_residual) > tolerance:
        raise RuntimeError("Action-preserving reinterpretation failed its mean constraint")
    return {"feasible": bool(feasible), "delta": delta, "increase_capacity": positive_capacity,
            "decrease_capacity": negative_capacity, "minimum_movement": 2 * abs(delta) if feasible else None,
            "movement": float(2 * marginal @ np.abs(updated - p)) if feasible else None,
            "mean_constraint_residual": constraint_residual,
            "posteriors": np.column_stack((1 - updated, updated)) if feasible else None}


def action_preserving_reinterpretation(game, marginal, posteriors, actions):
    prior, marginal, beliefs = measured_map(game.prior, marginal, posteriors)
    actions = np.asarray(actions)
    messages, states = beliefs.shape
    if actions.shape != (messages,) or actions.dtype.kind not in "iu" or np.any(actions < 0) or np.any(actions >= len(game.actions)):
        raise ValueError("Action-preserving map has invalid measured actions")
    entries = messages * states
    equality, targets = [], []
    for message in range(messages):
        row = np.zeros(2 * entries)
        row[message * states:(message + 1) * states] = 1
        equality.append(row)
        targets.append(marginal[message])
    for state in range(states):
        row = np.zeros(2 * entries)
        row[state:entries:states] = 1
        equality.append(row)
        targets.append(prior[state])
    rows, bounds = [], []
    joint = (marginal[:, None] * beliefs).ravel()
    for coordinate in range(entries):
        positive, negative = np.zeros(2 * entries), np.zeros(2 * entries)
        positive[coordinate], positive[entries + coordinate] = 1, -1
        negative[coordinate], negative[entries + coordinate] = -1, -1
        rows.extend((positive, negative))
        bounds.extend((joint[coordinate], -joint[coordinate]))
    for message, action in enumerate(actions):
        for alternative in range(len(game.actions)):
            if alternative != action:
                row = np.zeros(2 * entries)
                row[message * states:(message + 1) * states] = game.receiver_utility[alternative] - game.receiver_utility[action]
                rows.append(row)
                bounds.append(0.0)
    objective = np.r_[np.zeros(entries), np.ones(entries)]
    result = linprog(objective, A_ub=np.asarray(rows), b_ub=bounds, A_eq=np.asarray(equality), b_eq=targets,
                     bounds=(0, None), method="highs-ds")
    if result.status == 2:
        return {"feasible": False, "movement": None, "posteriors": None}
    if not result.success:
        raise RuntimeError("Action-preserving program failed numerical validation")
    updated = beliefs.copy()
    positive = marginal > 0
    updated[positive] = result.x[:entries].reshape(messages, states)[positive] / marginal[positive, None]
    return {"feasible": True, "movement": float(result.fun), "posteriors": updated,
            "unconstrained_minimum": float(np.linalg.norm(marginal @ beliefs - prior, 1))}


def drift_audit(prior, marginal, cached, refreshed, cached_errors=None, refreshed_errors=None):
    prior, marginal, cached = measured_map(prior, marginal, cached)
    _, _, refreshed = measured_map(prior, marginal, refreshed)
    old_residual, new_residual = marginal @ cached - prior, marginal @ refreshed - prior
    difference = float(np.linalg.norm(old_residual - new_residual, 1))
    movement = float(np.sum(marginal[:, None] * np.abs(cached - refreshed)))
    errors = np.zeros(len(marginal))
    for supplied in (cached_errors, refreshed_errors):
        if supplied is not None:
            array = np.asarray(supplied, dtype=float)
            if array.shape != errors.shape or not np.isfinite(array).all() or np.any(array < 0):
                raise ValueError("Posterior error radii are finite nonnegative message-level l1 bounds")
            errors += array
    return {"posterior_movement": movement, "cached_residual": old_residual, "refreshed_residual": new_residual,
            "residual_difference_bound": difference,
            "residual_norm_bound": max(float(np.linalg.norm(old_residual, 1) - np.linalg.norm(new_residual, 1)), 0.0),
            "error_adjusted_lower_bound": max(difference - float(marginal @ errors), 0.0),
            "statistical_error_bounds_supplied": cached_errors is not None and refreshed_errors is not None}


def controlled_transfer(marginal, recipient, donor, amount):
    marginal = probability_vector(marginal)
    if not 0 <= recipient < len(marginal) or not 0 <= donor < len(marginal) or recipient == donor:
        raise ValueError("A transfer uses two distinct message indices")
    if not np.isfinite(amount) or not 0 <= amount <= marginal[donor]:
        raise ValueError("Policy transfer preserves nonnegative message mass")
    result = marginal.copy()
    result[recipient] += amount
    result[donor] -= amount
    return result


def local_stability_radius(marginal, posteriors, radius, prior=None):
    marginal = probability_vector(marginal)
    center = marginal @ np.asarray(posteriors, dtype=float) if prior is None else prior
    center, _, beliefs = measured_map(center, marginal, posteriors)
    if not np.allclose(marginal @ beliefs, center, atol=1e-10, rtol=0):
        raise ValueError("The local radius has a compatible reference mixture")
    if not np.isfinite(radius) or not 0 <= radius <= 2 * np.min(marginal):
        raise ValueError("The exact local radius is bounded by twice the smallest reference mass")
    diameter = max(float(np.linalg.norm(first - second, 1)) for first in beliefs for second in beliefs)
    return {"posterior_diameter": diameter, "maximum_residual": radius * diameter / 2, "reference_prior": center}
