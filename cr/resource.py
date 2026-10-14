from pathlib import Path

import numpy as np

from .adaptive import AdaptiveSender
from .measurement import BudgetExhausted, ReceiverChanged
from .messages import partition_messages
from .program import policy_marginal
from .storage import signature, write_json


class ResourceSender:
    def __init__(self, game, messages, generator, engine, settings, output,
                 trainer=None, method="dual_guided", proposals=None, initial_measurements=None):
        self.game, self.catalog, self.generator, self.engine = game, list(messages), generator, engine
        self.settings, self.output, self.trainer, self.method = dict(settings), Path(output), trainer, method
        if not self.catalog or len({message.identifier for message in self.catalog}) != len(self.catalog):
            raise ValueError("A resource has a nonempty catalog with distinct message identifiers")
        if self.output.exists() and any(self.output.iterdir()):
            raise ValueError("A new resource run requires an empty output directory")
        groups = partition_messages(self.catalog)
        self.groups = {}
        self.records = []
        for identifier, pool in sorted(groups.items()):
            schedule = [record["groups"][identifier] for record in proposals] if proposals else None
            readings = ([reading for reading in initial_measurements if reading.message_id in {message.identifier for message in pool}]
                        if initial_measurements is not None else None)
            group_settings = dict(self.settings)
            if self.settings.get("exposure") == "announced":
                group_settings["initial_policy"] = self.settings["initial_policies"][identifier]
            self.groups[identifier] = AdaptiveSender(game, pool, generator, engine, group_settings,
                                                     self.output / "groups" / identifier, trainer, method, schedule, readings)

    def incomplete(self, reason, group_id=None):
        for runner in self.groups.values():
            runner.persist_pool()
        result = {"status": "additional_measurement_required", "reason": reason, "group_id": group_id,
                  "method": self.method, "rounds": len(self.records), "query_account": self.engine.ledger.summary()}
        write_json(self.output / "result.json", result)
        return result

    def run(self):
        write_json(self.output / "metadata.json", {"game": self.game.record(), "settings": self.settings,
                                                   "method": self.method, "receiver": self.engine.backend.identity,
                                                   "generator": self.generator.backend.identity,
                                                   "initial_query_account": self.engine.ledger.summary(),
                                                   "catalog_signature": signature(self.catalog),
                                                   "groups": {key: {"context": runner.context, "proposition": runner.proposition,
                                                                    "released_messages": len(runner.catalog)}
                                                              for key, runner in self.groups.items()}})
        try:
            for identifier, runner in self.groups.items():
                runner.prepare()
                if runner.initialize() is None:
                    return self.incomplete("initial_pool_infeasible", identifier)
            rounds = 0 if self.method == "static_mr" else int(self.settings.get("rounds", 4))
            if rounds < 0:
                raise ValueError("Expansion-round limit is nonnegative")
            contexts = [(runner.context, runner.proposition) for runner in self.groups.values()]
            total_acquisition = int(self.settings.get("acquisition_calls", 4096))
            for round_id in range(rounds):
                solutions = {}
                for identifier, runner in self.groups.items():
                    solution = runner.solve()
                    if self.engine.checkpoint_signature != runner.receiver_checkpoint:
                        if runner.refresh(solution.policy, round_id) is None:
                            return self.incomplete("changed_receiver_measurement_failed", identifier)
                        solution = runner.repair(round_id)
                        if solution is None:
                            return self.incomplete("changed_receiver_pool_infeasible", identifier)
                    solutions[identifier] = solution
                stage_limit = total_acquisition // rounds + int(round_id < total_acquisition % rounds)
                training = []
                if self.trainer is not None:
                    reserve = 2 * max(runner.pool_repeats for runner in self.groups.values()) * int(self.settings.get("candidate_batch", 16))
                    training = self.trainer.update(contexts, solutions,
                                                   {key: runner.exposure for key, runner in self.groups.items()},
                                                   round_id, self.method == "payoff_guided", max(stage_limit - reserve, 0),
                                                   self.method == "posterior_value",
                                                   {key: runner.pool_repeats for key, runner in self.groups.items()})
                count = int(self.settings.get("candidate_batch", 16))
                seed = int(self.settings.get("seed", 0)) + 1000000 + 10000 * round_id
                if self.method == "random_expansion":
                    known = {message.identifier for runner in self.groups.values() for message in runner.messages}
                    available = [message for message in self.catalog if message.identifier not in known]
                    order = np.random.default_rng(seed).permutation(len(available))[:count]
                    candidates = [available[index] for index in order]
                else:
                    order = np.random.default_rng(seed).permutation(len(contexts))
                    candidates = self.generator.propose([contexts[index] for index in order], count, seed)
                proposed = partition_messages(candidates)
                if not set(proposed).issubset(self.groups):
                    raise ValueError("Generated candidates have an unmeasured conditioning context")
                group_records = {}
                for identifier, runner in self.groups.items():
                    scored = runner.acquire(solutions[identifier], [(runner.context, runner.proposition)], round_id,
                                            stage_limit, proposed.get(identifier, []))
                    admitted = runner.confirm(scored, solutions[identifier], round_id)
                    policy = runner.proposed_policy(round_id)
                    marginal = policy_marginal(self.game.prior, policy)
                    trigger, norm, changed = runner.refresh_required(marginal, round_id, policy)
                    audit = runner.refresh(policy, round_id) if trigger else None
                    if trigger and audit is None:
                        return self.incomplete("receiver_refresh_failed", identifier)
                    solution = runner.repair(round_id)
                    if solution is None:
                        return self.incomplete("refreshed_pool_infeasible", identifier)
                    record = {"round": round_id, "group_id": identifier, "admitted": admitted,
                              "pool_size": len(runner.messages), "proposed_policy": policy, "proposed_marginal": marginal,
                              "cached_residual_norm": norm, "refreshed": trigger,
                              "receiver_checkpoint_changed": changed, "drift_audit": audit, "solution": solution}
                    runner.records.append(record)
                    write_json(runner.output / ("round_" + str(round_id).zfill(3) + ".json"), record)
                    group_records[identifier] = record
                record = {"round": round_id, "training": training, "groups": group_records,
                          "query_account": self.engine.ledger.summary()}
                self.records.append(record)
                write_json(self.output / ("round_" + str(round_id).zfill(3) + ".json"), record)
                if self.engine.ledger.remaining() == 0:
                    break
            results, policies = {}, {}
            for identifier, runner in self.groups.items():
                result = runner.finish(rounds, save_adapter=False)
                if result["status"] != "ready":
                    return self.incomplete(result["reason"], identifier)
                results[identifier] = result
                policies[identifier] = {"messages": runner.messages, "policy": result["policy"],
                                        "policy_definition": result["policy_definition"],
                                        "receiver_signature": runner.channel().receiver_signature}
            write_json(self.output / "group_policies.json", {"format": "cr_group_policies_v1",
                                                              "game": self.game.record(), "groups": policies})
            if self.trainer is not None:
                self.generator.backend.save_adapter(self.output / "sender_adapter.npz")
            weights = {identifier: len(runner.catalog) / len(self.catalog) for identifier, runner in self.groups.items()}
            result = {"status": "ready", "method": self.method, "groups": results, "rounds": len(self.records),
                      "resource_utility_weighting": "released_message_counts", "group_weights": weights,
                      "sender_utility": sum(weights[key] * value["deployed_policy_audit"]["sender_utility"] for key, value in results.items()),
                      "query_account": self.engine.ledger.summary(), "policy_manifest": "group_policies.json"}
            write_json(self.output / "result.json", result)
            return result
        except BudgetExhausted:
            return self.incomplete("receiver_budget_exhausted")
        except ReceiverChanged:
            return self.incomplete("receiver_changed_during_measurement")
