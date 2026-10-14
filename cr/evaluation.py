import importlib

import numpy as np

from .game import probability_vector
from .messages import Message, common_context, context_group
from .program import policy_marginal, validate_policy


def evaluate_policy(channel, policy):
    policy = validate_policy(policy, len(channel.game.prior), len(channel.message_ids))
    joint = channel.game.prior[:, None] * policy
    marginal = joint.sum(axis=0)
    induced = channel.posteriors.copy()
    positive = marginal > 0
    induced[positive] = joint[:, positive].T / marginal[positive, None]
    discrepancy = float(np.sum(marginal[:, None] * np.abs(induced - channel.posteriors)))
    sender = float(np.sum(joint.T * channel.game.sender_utility[channel.actions]))
    receiver = float(np.sum(joint.T * channel.game.receiver_utility[channel.actions]))
    return {"sender_utility": sender, "receiver_utility": receiver, "marginal": marginal,
            "reported_objective": float(marginal @ channel.payoffs),
            "posterior_mean_residual": marginal @ channel.posteriors - channel.game.prior,
            "policy_posterior_discrepancy": discrepancy, "compatible": discrepancy <= 1e-8}


def posterior_projection(channel, posteriors):
    posteriors = np.asarray(posteriors, dtype=float)
    if posteriors.ndim != 2 or posteriors.shape[1] != len(channel.game.prior):
        raise ValueError("Generated posterior projection has incompatible state dimensions")
    indices, distances = [], []
    for posterior in posteriors:
        probability_vector(posterior, len(channel.game.prior))
        norms = np.linalg.norm(channel.posteriors - posterior, axis=1)
        index = int(np.argmin(norms))
        indices.append(index)
        distances.append(float(norms[index]))
    return {"indices": indices, "message_ids": [channel.message_ids[index] for index in indices],
            "distances": distances, "projected_payoffs": channel.payoffs[indices]}


def optimization_gap(channel, solution, policy):
    record = evaluate_policy(channel, policy)
    record["optimization_gap"] = solution.value - record["sender_utility"] if record["compatible"] else None
    return record


def load_native_evaluator(factory, options):
    module, name = factory.split(":", 1)
    evaluator = getattr(importlib.import_module(module), name)(**options)
    if not callable(getattr(evaluator, "score", None)):
        raise ValueError("Native evaluator exposes score on complete released records")
    return evaluator


class MixtureSender:
    def __init__(self, game, messages, policy):
        self.game, self.messages = game, list(messages)
        common_context(self.messages)
        self.group_id = self.messages[0].group_id
        self.policy = validate_policy(policy, len(game.prior), len(messages))

    def select(self, evaluator, record, seed):
        if context_group(evaluator.sender_context(record), evaluator.proposition(record)) != self.group_id:
            raise ValueError("Native record has no measured policy for its context and proposition")
        state = sender_state(self.game, evaluator.sender_state(record))
        index = int(np.random.default_rng(seed).choice(len(self.messages), p=self.policy[state]))
        return self.messages[index]


def sender_state(game, state):
    if isinstance(state, str):
        return game.states.index(state)
    if isinstance(state, bool) or not isinstance(state, (int, np.integer)) or not 0 <= state < len(game.prior):
        raise ValueError("Private sender state lies outside the retained state set")
    return int(state)


class GroupedMixtureSender:
    def __init__(self, game, groups):
        self.game = game
        self.groups = {identifier: MixtureSender(game, [Message.from_record(row) for row in value["messages"]], value["policy"])
                       for identifier, value in groups.items()}
        if not self.groups or any(key != sender.group_id for key, sender in self.groups.items()):
            raise ValueError("Grouped policies have consistent context identifiers")

    def select(self, evaluator, record, seed):
        identifier = context_group(evaluator.sender_context(record), evaluator.proposition(record))
        if identifier not in self.groups:
            raise ValueError("Native evaluation context requires its own measured channel")
        return self.groups[identifier].select(evaluator, record, seed)


def evaluate_native(evaluator, records, sender, game, seed=0):
    result = []
    for index, record in enumerate(records):
        if isinstance(sender, (MixtureSender, GroupedMixtureSender)):
            message = sender.select(evaluator, record, seed + index)
        else:
            context = evaluator.sender_context(record)
            state = sender_state(game, evaluator.sender_state(record))
            proposed = sender.propose([(context, evaluator.proposition(record))], 1, seed + index, states=[state])
            if len(proposed) != 1:
                raise ValueError("Native sender returned no complete message")
            message = proposed[0]
        value = evaluator.score(record, message.text)
        if not isinstance(value, dict) or value.get("data_kind") != "measured_run":
            raise ValueError("Released evaluator returns measured native outputs")
        result.append(dict(value, item=index, message=message.text, sender_seed=seed + index))
    return result
