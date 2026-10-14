from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog, minimize

from .game import probability_vector


class InfeasiblePool(ValueError):
    pass


@dataclass(frozen=True)
class SenderSolution:
    message_ids: tuple
    marginal: np.ndarray
    policy: np.ndarray
    value: float
    state_prices: np.ndarray
    normalization_price: float
    dual_value: float
    residual: float
    dual_violation: float
    complementarity: float
    receiver_signature: str

    def __post_init__(self):
        for name in ("marginal", "policy", "state_prices"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def support(self):
        return tuple(self.message_ids[index] for index in np.flatnonzero(self.marginal > 1e-10))

    def reduced_costs(self, posteriors, payoffs):
        beliefs = np.asarray(posteriors, dtype=float)
        values = np.asarray(payoffs, dtype=float)
        if beliefs.ndim != 2 or beliefs.shape[1] != len(self.state_prices) or values.shape != (len(beliefs),):
            raise ValueError("Candidate dimensions differ from the frozen dual solution")
        if not np.isfinite(beliefs).all() or not np.isfinite(values).all():
            raise ValueError("Candidate values and beliefs are finite")
        return values - beliefs @ self.state_prices - self.normalization_price


def validate_policy(policy, states, messages, tolerance=1e-8):
    policy = np.array(policy, dtype=float, copy=True)
    if policy.shape != (states, messages) or not np.isfinite(policy).all() or np.any(policy < 0):
        raise ValueError("Policy has finite nonnegative state-by-message entries")
    if not np.allclose(policy.sum(axis=1), 1, atol=tolerance, rtol=0):
        raise ValueError("Every state-conditional policy row is normalized")
    return policy


def policy_marginal(prior, policy):
    prior = probability_vector(prior)
    policy = validate_policy(policy, len(prior), np.asarray(policy).shape[-1])
    return prior @ policy


def reconstruct_policy(prior, posteriors, marginal, tolerance=1e-8):
    prior = probability_vector(prior)
    marginal = probability_vector(marginal, len(posteriors), tolerance)
    posteriors = np.asarray(posteriors, dtype=float)
    if posteriors.shape != (len(marginal), len(prior)) or np.any(prior <= 0):
        raise ValueError("Reconstruction uses retained positive-prior states")
    if not np.allclose(marginal @ posteriors, prior, atol=tolerance, rtol=0):
        raise ValueError("Message mixture has a posterior-mean residual")
    policy = (marginal[:, None] * posteriors).T / prior[:, None]
    return validate_policy(policy, len(prior), len(marginal), tolerance)


def solve_sender(channel, payoffs=None, tolerance=1e-8):
    values = channel.payoffs if payoffs is None else np.asarray(payoffs, dtype=float)
    if values.shape != (len(channel.message_ids),) or not np.isfinite(values).all():
        raise ValueError("Sender coefficients have one finite value per message")
    equality = np.vstack((channel.posteriors.T, np.ones(len(values))))
    target = np.append(channel.game.prior, 1)
    result = linprog(-values, A_eq=equality, b_eq=target, bounds=(0, None), method="highs-ds",
                     options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
    if result.status == 2:
        raise InfeasiblePool("The measured posterior hull excludes the prior")
    if not result.success:
        raise RuntimeError("Sender linear program failed numerical validation")
    marginal = np.maximum(result.x, 0)
    marginal /= marginal.sum()
    prices = -np.asarray(result.eqlin.marginals)
    reduced = values - equality.T @ prices
    value = float(marginal @ values)
    dual_value = float(target @ prices)
    residual = float(np.max(np.abs(equality @ marginal - target)))
    dual_violation = float(max(0.0, np.max(reduced)))
    complementarity = float(np.max(np.abs(marginal * reduced)))
    scale = max(1.0, abs(value), abs(dual_value), float(np.max(np.abs(values))))
    if residual > tolerance or dual_violation > tolerance * scale or abs(dual_value - value) > tolerance * scale:
        raise RuntimeError("Sender primal and dual certificates disagree")
    policy = reconstruct_policy(channel.game.prior, channel.posteriors, marginal, tolerance)
    return SenderSolution(channel.message_ids, marginal, policy, value, prices[:-1], float(prices[-1]),
                          dual_value, residual, dual_violation, complementarity, channel.receiver_signature)


def hull_distance(prior, posteriors):
    prior = probability_vector(prior)
    beliefs = np.asarray(posteriors, dtype=float)
    if beliefs.ndim != 2 or beliefs.shape[1] != len(prior) or not len(beliefs):
        raise ValueError("Hull distance requires a nonempty posterior pool")
    if not np.isfinite(beliefs).all() or np.any(beliefs < 0) or not np.allclose(beliefs.sum(axis=1), 1, atol=1e-9, rtol=0):
        raise ValueError("Hull points are posterior probability vectors")
    if len(prior) == 2:
        coordinate = float(np.clip(prior[1], beliefs[:, 1].min(), beliefs[:, 1].max()))
        return float(np.sqrt(2) * abs(coordinate - prior[1]))
    equality = np.vstack((beliefs.T, np.ones(len(beliefs))))
    result = linprog(np.zeros(len(beliefs)), A_eq=equality, b_eq=np.append(prior, 1), bounds=(0, None), method="highs")
    if result.success:
        return float(np.linalg.norm(result.x @ beliefs - prior))
    def objective(weights):
        change = weights @ beliefs - prior
        return 0.5 * np.dot(change, change)
    def gradient(weights):
        return beliefs @ (weights @ beliefs - prior)
    result = minimize(objective, np.full(len(beliefs), 1 / len(beliefs)), jac=gradient, method="SLSQP",
                      bounds=[(0, 1)] * len(beliefs),
                      constraints={"type": "eq", "fun": lambda q: q.sum() - 1, "jac": lambda q: np.ones_like(q)},
                      options={"ftol": 1e-14, "maxiter": 2000})
    if not result.success:
        raise RuntimeError("Posterior hull projection failed numerical validation")
    return float(np.linalg.norm(result.x @ beliefs - prior))


def classical_sender_value(game):
    actions, states = game.sender_utility.shape
    rows = []
    for chosen in range(actions):
        for alternative in range(actions):
            if chosen != alternative:
                row = np.zeros((actions, states))
                row[chosen] = game.receiver_utility[alternative] - game.receiver_utility[chosen]
                rows.append(row.ravel())
    equality = np.tile(np.eye(states), (1, actions))
    result = linprog(-game.sender_utility.ravel(), A_ub=np.asarray(rows) if rows else None,
                     b_ub=np.zeros(len(rows)) if rows else None, A_eq=equality, b_eq=game.prior,
                     bounds=(0, None), method="highs-ds")
    if not result.success:
        raise RuntimeError("Classical persuasion program failed")
    joint = result.x.reshape(actions, states)
    return {"value": float(np.sum(joint * game.sender_utility)), "joint_recommendations": joint,
            "policy": joint.T / game.prior[:, None]}


def price_decomposition(channel):
    measured = solve_sender(channel)
    covered = solve_sender(channel, channel.classical_payoffs)
    classical = classical_sender_value(channel.game)
    return {"classical_value": classical["value"], "coverage_value": covered.value,
            "measured_value": measured.value, "price_of_language": classical["value"] - measured.value,
            "coverage_gap": classical["value"] - covered.value, "action_gap": covered.value - measured.value,
            "pointwise_best_response_bound": bool(np.all(channel.payoffs <= channel.classical_payoffs + 1e-9))}
