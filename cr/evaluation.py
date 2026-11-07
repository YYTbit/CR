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


def evaluate_native(evaluator, records, sender, game, seed=0, completed=(), on_result=None):
    result = list(completed)
    for index, record in enumerate(records):
        if index < len(completed):
            continue
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
        row = dict(value, item=index, message=message.text, sender_seed=seed + index)
        if on_result is not None:
            on_result(row)
        result.append(row)
    return result


def run_native_evaluation(args):
    import inspect
    import json
    from pathlib import Path

    from .cli import create_generator, read_config, read_game, read_messages
    from .data import read_records
    from .storage import append_json, file_digest, recover_jsonl, signature, write_json

    output = Path(args.output)
    progress = output if output.suffix == ".jsonl" else output.with_suffix(".progress.jsonl")
    metadata_path = output.with_suffix(".metadata.json")
    requests_path = output.with_suffix(".requests.jsonl")
    if not args.resume and any(path.exists() for path in {output, progress, metadata_path, requests_path}):
        raise ValueError("Native evaluation output exists; use --resume with the same inputs")
    if args.resume and not metadata_path.is_file():
        raise ValueError("Native evaluation resume requires its saved metadata")
    output.parent.mkdir(parents=True, exist_ok=True)
    configuration = read_config(args.evaluator)
    game = read_game(args.game)
    records = read_records(args.records)
    sender, evaluator = None, None
    try:
        if args.procedure == "mixture":
            if not args.policy:
                raise ValueError("Native mixture evaluation requires the reconstructed policy")
            policy = json.loads(Path(args.policy).read_text())
            if isinstance(policy, dict) and policy.get("format") == "cr_group_policies_v1":
                if policy["game"] != game.record():
                    raise ValueError("Grouped native policies have a different game")
                sender = GroupedMixtureSender(game, policy["groups"])
            else:
                if not args.messages:
                    raise ValueError("Native single-channel evaluation requires its final message pool")
                sender = MixtureSender(game, read_messages(args.messages), policy["policy"] if isinstance(policy, dict) else policy)
        else:
            sender = create_generator(args, read_config(args.config), game)
        evaluator = load_native_evaluator(configuration["factory"], configuration.get("options", {}))
        participants = {"receiver": evaluator}
        if getattr(evaluator, "judge", None) is not None:
            participants["judge"] = evaluator.judge
        inputs = {"protocol": 3, "game": game.record(), "records": file_digest(args.records),
                  "evaluator_config": signature(configuration), "procedure": args.procedure, "seed": args.seed,
                  "sender": sender.backend.identity if args.procedure == "generator" else None,
                  "policy": file_digest(args.policy) if args.policy else None,
                  "messages": file_digest(args.messages) if args.messages else None,
                  "adapter": file_digest(args.adapter) if args.adapter else None,
                  "evaluator_source": file_digest(inspect.getfile(type(evaluator))),
                  "source_revision": getattr(evaluator, "revision", None),
                  "source_hashes": getattr(evaluator, "source_hashes", {}),
                  "schema": getattr(evaluator, "schema", None),
                  "participants": {name: {"backend": getattr(getattr(participant, "backend", None), "identity", None),
                                           "source_revision": getattr(participant, "revision", None),
                                           "source_hashes": getattr(participant, "source_hashes", {})}
                                   for name, participant in participants.items()}}
        run_id = signature(inputs)
        metadata = {"run_id": run_id, "inputs": inputs, "items": len(records)}
        if args.resume and json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("Native evaluation inputs, model or source configuration changed")
        write_json(metadata_path, metadata)
        completed = recover_jsonl(progress) if args.resume else []
        if len(completed) > len(records) or any(row.get("item") != index or row.get("run_id") != run_id
                                               for index, row in enumerate(completed)):
            raise ValueError("Native evaluation progress is not a committed prefix of this run")
        calls = recover_jsonl(requests_path) if args.resume else []
        if calls and any(row.get("run_id") != run_id for row in calls):
            raise ValueError("Native request journal belongs to another evaluation")
        for name, participant in participants.items():
            if hasattr(participant, "request_recorder"):
                participant.request_count = max((row["call"] for row in calls if row.get("participant") == name), default=0)
                participant.request_recorder = lambda row, name=name: append_json(
                    requests_path, dict(row, run_id=run_id, participant=name))
        def commit(row):
            row["run_id"] = run_id
            append_json(progress, row)

        rows = evaluate_native(evaluator, records, sender, game, args.seed, completed, commit)
        if output != progress:
            write_json(output, rows)
    finally:
        if evaluator is not None and callable(getattr(evaluator, "close", None)):
            evaluator.close()
        if args.procedure == "generator" and sender is not None:
            sender.backend.close()
