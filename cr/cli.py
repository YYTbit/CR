import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

from .game import Game
from .measurement import Measurement, MeasurementEngine, QueryLedger
from .messages import Message, partition_messages
from .movement import (action_preserving_reinterpretation, binary_action_capacity, binary_action_interval,
                       drift_audit, minimum_reinterpretation)
from .program import classical_sender_value, price_decomposition, solve_sender
from .storage import file_digest, json_value, read_channel, read_channel_collection, read_jsonl, write_channel, write_json


def read_config(path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Configuration is a mapping")
    return value


def read_game(path):
    return Game.from_record(json.loads(Path(path).read_text()))


def read_messages(path):
    rows = read_jsonl(path) if Path(path).suffix == ".jsonl" else json.loads(Path(path).read_text())
    return [Message.from_record(row) for row in rows]


def create_receiver(args, config):
    checkpoint = args.receiver or config["receiver"]["checkpoint"]
    if checkpoint in config["models"]:
        checkpoint = config["models"][checkpoint]["checkpoint"]
    revision = args.receiver_revision or config["receiver"].get("revision")
    if args.receiver_backend == "api":
        from .backends import ApiBackend
        return ApiBackend(checkpoint, revision, args.endpoint, timeout=config["protocol"]["timeout_seconds"],
                          max_tokens=config["execution"]["max_completion_tokens"])
    from .models import CausalBackend
    execution = config["execution"]
    return CausalBackend.from_pretrained(checkpoint, revision, args.device_map, execution["dtype"],
                                         max_input_tokens=execution["max_input_tokens"],
                                         max_completion_tokens=execution["max_completion_tokens"],
                                         enable_thinking=execution["enable_thinking"])


def create_engine(args, config, game, destination):
    protocol = config["protocol"]
    budget = args.query_budget if args.query_budget is not None else protocol["query_budget"]
    ledger = QueryLedger(budget, destination / "queries.jsonl")
    backend = create_receiver(args, config)
    engine = MeasurementEngine(game, backend, ledger, args.seed, protocol["retries"],
                                protocol["headline_temperature"], protocol["repeat_temperature"])
    return engine


def create_generator(args, config, game):
    from .models import CausalBackend, LanguageGenerator
    profile = config["models"][args.sender]
    execution = config["execution"]
    backend = CausalBackend.from_pretrained(profile["checkpoint"], profile.get("revision"), args.device_map,
                                            execution["dtype"], max_input_tokens=execution["max_input_tokens"],
                                            max_completion_tokens=execution["max_completion_tokens"],
                                            enable_thinking=execution["enable_thinking"])
    if args.adapter:
        backend.load_adapter(args.adapter)
    return LanguageGenerator(backend, game, execution["sender_temperature"])


def solve_command(args):
    groups = read_channel_collection(args.channel)
    if len(groups) == 1:
        channel, record = next(iter(groups.values()))
        solution = solve_sender(channel)
        write_json(args.output, {"solution": solution, "policy": solution.policy, "price": price_decomposition(channel)})
        print(json.dumps({"value": solution.value, "dual_value": solution.dual_value, "support": solution.support}))
    else:
        results = {}
        for identifier, (channel, record) in groups.items():
            solution = solve_sender(channel)
            results[identifier] = {"messages": record["messages"], "policy": solution.policy,
                                   "solution": solution, "price": price_decomposition(channel)}
        write_json(args.output, {"format": "cr_group_policies_v1", "game": channel.game.record(), "groups": results})
        print(json.dumps({"groups": len(results)}))


def movement_command(args):
    channel, record = read_channel(args.channel)
    marginal = json.loads(Path(args.marginal).read_text())
    result = minimum_reinterpretation(channel.game.prior, marginal, channel.posteriors)
    if len(channel.game.prior) == 2:
        intervals = [binary_action_interval(channel.game.receiver_utility, int(action)) for action in channel.actions]
        if all(interval is not None for interval in intervals):
            try:
                result["binary_action_capacity"] = binary_action_capacity(channel.game.prior, marginal, channel.posteriors, intervals)
            except ValueError:
                result["binary_action_capacity"] = {"assumption_satisfied": False}
    result["action_preserving_program"] = action_preserving_reinterpretation(channel.game, marginal, channel.posteriors, channel.actions)
    if args.refreshed:
        updated, refreshed = read_channel(args.refreshed)
        if updated.message_ids != channel.message_ids or updated.game.record() != channel.game.record():
            raise ValueError("Drift audit uses matching game and message identities")
        result["drift_audit"] = drift_audit(channel.game.prior, marginal, channel.posteriors, updated.posteriors)
    write_json(args.output, result)
    print(json.dumps({"minimum_movement": result["minimum"], "constructive_movement": result["movement"]}))


def measure_command(args):
    config, game = read_config(args.config), read_game(args.game)
    destination = Path(args.output)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("A new measurement requires an empty output directory")
    messages = read_messages(args.messages)
    engine = create_engine(args, config, game, destination)
    try:
        groups, paths, count = partition_messages(messages), {}, 0
        for identifier, pool in groups.items():
            retained, readings = engine.pool(pool, "shared_pool" if args.shared else "measurement", args.repeats)
            if not retained:
                raise ValueError("Measurement returned no complete belief-action pair in a context group")
            relative = "channel.json" if len(groups) == 1 else "groups/" + identifier + "/channel.json"
            write_channel(destination / relative, game, retained, readings,
                          {"data_kind": engine.backend.data_kind, "query_account": engine.ledger.summary(), "group_id": identifier})
            paths[identifier] = relative
            count += len(retained)
        if len(groups) > 1:
            write_json(destination / "channels.json", {"format": "cr_group_channels_v1", "game": game.record(), "groups": paths})
        print(json.dumps({"messages": count, "groups": len(groups), "queries": engine.ledger.summary()}))
    finally:
        engine.backend.close()


def run_command(args):
    from .adaptive import AdaptiveSender
    from .resource import ResourceSender
    from .learning import GRPOTrainer
    config, game = read_config(args.config), read_game(args.game)
    settings = dict(config["protocol"], seed=args.seed)
    if args.variant:
        variants = read_config(args.ablations)
        lookup = {key: value for section in variants.values() for key, value in section.items()}
        override = dict(lookup[args.variant])
        args.method = override.pop("method", args.method)
        settings.update(override)
    if args.exposure:
        settings.update(json.loads(Path(args.exposure).read_text()))
    destination = Path(args.output)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("A new adaptive run requires an empty output directory")
    engine = create_engine(args, config, game, destination)
    generator = None
    try:
        generator = create_generator(args, config, game)
        trainer = None
        if args.method in {"dual_guided", "payoff_guided", "posterior_value", "shared_measurement", "cached_response", "direct_generator"}:
            training = dict(config["training"], total_optimizer_steps=settings["rounds"] * config["training"]["updates_per_round"])
            trainer = GRPOTrainer(generator, engine, training, args.seed)
        proposals = json.loads(Path(args.proposals).read_text()) if args.proposals else None
        initial = None
        if args.initial_channel:
            initial = []
            archive_signature = file_digest(args.initial_channel)
            for channel, record in read_channel_collection(args.initial_channel).values():
                if channel.game.record() != game.record():
                    raise ValueError("Recorded initial channel has a different game")
                engine.ledger.import_initial_account(record.get("metadata", {}).get("query_account"), archive_signature)
                initial.extend(Measurement(row["message_id"], row["message_fingerprint"], np.asarray(row["posterior"]), row["action"],
                                           row["receiver_signature"], np.asarray(row["posterior_samples"]), tuple(row["action_samples"]),
                                           row["phase"], row["round_id"], row["data_kind"]) for row in record["measurements"])
        messages = read_messages(args.messages)
        runner_class = ResourceSender if len(partition_messages(messages)) > 1 else AdaptiveSender
        runner = runner_class(game, messages, generator, engine, settings, destination,
                              trainer, args.method, proposals, initial)
        result = runner.run()
        print(json.dumps({"status": result["status"], "method": args.method, "queries": result["query_account"]}))
    finally:
        engine.backend.close()
        if generator is not None:
            generator.backend.close()


def baseline_command(args):
    from .baselines import select_message
    config, game = read_config(args.config), read_game(args.game)
    destination = Path(args.output)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("A baseline run requires an empty output directory")
    destination.mkdir(parents=True, exist_ok=True)
    engine = create_engine(args, config, game, destination)
    generator = None
    try:
        generator = create_generator(args, config, game)
        groups = partition_messages(read_messages(args.messages))
        results = {identifier: select_message(generator, engine, [(pool[0].context, pool[0].proposition)], args.candidates, args.seed + index,
                                              confirmation_repeats=config["protocol"]["confirmation_repeats"],
                                              final_repeats=config["protocol"]["final_repeats"])
                   for index, (identifier, pool) in enumerate(sorted(groups.items()))}
        result = next(iter(results.values())) if len(results) == 1 else {"status": "ready" if all(row["status"] == "ready" for row in results.values())
                                                                      else "additional_measurement_required", "groups": results,
                                                                      "query_account": engine.ledger.summary()}
        write_json(destination / "result.json", result)
        print(json.dumps({"status": result["status"]}))
    finally:
        engine.backend.close()
        if generator is not None:
            generator.backend.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="The Cost of Reinterpretation: Sender Adaptation against Measured Language Receivers")
    commands = parser.add_subparsers(dest="operation", required=True)
    solve = commands.add_parser("solve")
    solve.add_argument("--channel", required=True)
    solve.add_argument("--output", required=True)
    solve.set_defaults(function=solve_command)
    movement = commands.add_parser("movement")
    movement.add_argument("--channel", required=True)
    movement.add_argument("--marginal", required=True)
    movement.add_argument("--refreshed")
    movement.add_argument("--output", required=True)
    movement.set_defaults(function=movement_command)
    def receiver_arguments(command):
        command.add_argument("--config", default="configs/paper.yaml")
        command.add_argument("--game", required=True)
        command.add_argument("--messages", required=True)
        command.add_argument("--receiver")
        command.add_argument("--receiver-revision")
        command.add_argument("--receiver-backend", choices=["local", "api"], default="local")
        command.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions")
        command.add_argument("--query-budget", type=int)
        command.add_argument("--device-map", default="auto")
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--output", required=True)
    measurement = commands.add_parser("measure")
    receiver_arguments(measurement)
    measurement.add_argument("--repeats", type=int, default=1)
    measurement.add_argument("--shared", action="store_true")
    measurement.set_defaults(function=measure_command)
    run = commands.add_parser("run")
    receiver_arguments(run)
    run.add_argument("--sender", choices=["Qwen3", "Llama3", "Gemma3"], default="Qwen3")
    run.add_argument("--adapter")
    run.add_argument("--initial-channel")
    run.add_argument("--proposals")
    run.add_argument("--exposure")
    run.add_argument("--method", choices=["dual_guided", "payoff_guided", "static_mr", "random_expansion", "posterior_value",
                                         "shared_measurement", "cached_response", "direct_generator"], default="dual_guided")
    run.add_argument("--variant")
    run.add_argument("--ablations", default="configs/ablations.yaml")
    run.set_defaults(function=run_command)
    baseline = commands.add_parser("baseline")
    receiver_arguments(baseline)
    baseline.add_argument("--sender", choices=["Qwen3", "Llama3", "Gemma3"], default="Qwen3")
    baseline.add_argument("--adapter")
    baseline.add_argument("--candidates", type=int, default=1)
    baseline.set_defaults(function=baseline_command)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--records", required=True)
    prepare.add_argument("--schema", required=True)
    prepare.add_argument("--resource", choices=["EoBench", "DuET-PD", "PMIYC", "SYCON-Bench", "RebuttalBench"], required=True)
    prepare.add_argument("--output", required=True)
    def prepare_command(args):
        from .data import prepare_file
        print(json.dumps(prepare_file(args.records, read_config(args.schema), args.resource, args.output)))
    prepare.set_defaults(function=prepare_command)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--sources", default="configs/sources.json")
    fetch.add_argument("--names", nargs="+")
    fetch.add_argument("--output", default="external")
    def fetch_command(args):
        entries = json.loads(Path(args.sources).read_text())
        root = Path(args.output)
        root.mkdir(parents=True, exist_ok=True)
        for name in args.names or sorted(entries):
            item = entries[name]
            destination = root / item["directory"]
            if not destination.exists():
                subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", item["repository"], str(destination)], check=True)
            subprocess.run(["git", "fetch", "origin", item["revision"]], cwd=destination, check=True)
            subprocess.run(["git", "checkout", "--detach", item["revision"]], cwd=destination, check=True)
            if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=destination, text=True).strip() != item["revision"]:
                raise ValueError("Benchmark source has a different pinned revision")
    fetch.set_defaults(function=fetch_command)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--channel", required=True)
    evaluate.add_argument("--policy", required=True)
    evaluate.add_argument("--output", required=True)
    def evaluate_command(args):
        from .evaluation import evaluate_policy
        channel, record = read_channel(args.channel)
        policy = json.loads(Path(args.policy).read_text())
        if isinstance(policy, dict):
            policy = policy["policy"]
        result = evaluate_policy(channel, policy)
        write_json(args.output, result)
        print(json.dumps(json_value(result)))
    evaluate.set_defaults(function=evaluate_command)
    native = commands.add_parser("native-evaluate")
    native.add_argument("--config", default="configs/paper.yaml")
    native.add_argument("--game", required=True)
    native.add_argument("--records", required=True)
    native.add_argument("--evaluator", required=True)
    native.add_argument("--procedure", choices=["mixture", "generator"], default="mixture")
    native.add_argument("--messages")
    native.add_argument("--policy")
    native.add_argument("--sender", choices=["Qwen3", "Llama3", "Gemma3"], default="Qwen3")
    native.add_argument("--adapter")
    native.add_argument("--device-map", default="auto")
    native.add_argument("--seed", type=int, default=0)
    native.add_argument("--output", required=True)
    def native_command(args):
        from .data import read_records
        from .evaluation import GroupedMixtureSender, MixtureSender, evaluate_native, load_native_evaluator
        configuration = read_config(args.evaluator)
        game = read_game(args.game)
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
                if isinstance(policy, dict):
                    policy = policy["policy"]
                sender = MixtureSender(game, read_messages(args.messages), policy)
        else:
            sender = create_generator(args, read_config(args.config), game)
        try:
            evaluator = load_native_evaluator(configuration["factory"], configuration.get("options", {}))
            rows = evaluate_native(evaluator, read_records(args.records), sender, game, args.seed)
            write_json(args.output, rows)
        finally:
            if args.procedure == "generator":
                sender.backend.close()
    native.set_defaults(function=native_command)
    native_summary_parser = commands.add_parser("native-summary")
    native_summary_parser.add_argument("--resource", choices=["EoBench", "DuET-PD", "PMIYC", "SYCON-Bench", "RebuttalBench"], required=True)
    native_summary_parser.add_argument("--records", required=True)
    native_summary_parser.add_argument("--expected-items", type=int)
    native_summary_parser.add_argument("--output", required=True)
    def native_summary_command(args):
        from .data import read_records
        from .native import native_summary
        result = native_summary(args.resource, read_records(args.records), args.expected_items)
        write_json(args.output, result)
        print(json.dumps(result))
    native_summary_parser.set_defaults(function=native_summary_command)
    comparison = commands.add_parser("statistics")
    comparison.add_argument("--pairs", required=True)
    comparison.add_argument("--output", required=True)
    def statistics_command(args):
        from .statistics import holm_correction, paired_comparison
        inputs = json.loads(Path(args.pairs).read_text())
        if inputs.get("data_kind") != "measured_run":
            raise ValueError("Statistical summaries use measured run records")
        result = {cell: paired_comparison(values["first"], values["second"]) for cell, values in inputs["cells"].items()}
        correction = holm_correction({cell: value["p_value"] for cell, value in result.items()})
        write_json(args.output, {"paired": result, "holm": correction})
    comparison.set_defaults(function=statistics_command)
    arguments = parser.parse_args(argv)
    arguments.function(arguments)
