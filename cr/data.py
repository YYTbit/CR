import csv
import json
from pathlib import Path

import numpy as np

from .game import Game
from .messages import Message, partition_messages
from .storage import file_digest, signature, write_json


RESOURCES = {"EoBench", "DuET-PD", "PMIYC", "SYCON-Bench", "RebuttalBench"}


def field(record, path):
    value = record
    for part in path.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def read_records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if path.suffix == ".csv":
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        if not isinstance(value, list):
            raise ValueError("Released JSON record input is a list of items")
        return value
    if path.suffix == ".parquet":
        from datasets import Dataset
        return list(Dataset.from_parquet(str(path)))
    raise ValueError("Released records use JSON, JSONL, CSV, or Parquet")


def ternary_action(value, lower, upper):
    number = float(value)
    if not np.isfinite(number) or not lower < upper or not lower <= number <= upper:
        raise ValueError("Native graded action lies within its declared range")
    return 0 if number == lower else 2 if number == upper else 1


def prepare_records(records, schema, resource):
    if resource not in RESOURCES or not records:
        raise ValueError("Resource preparation requires a declared benchmark and released records")
    messages, labels, seen = [], [], set()
    mapping = schema["positive_labels"]
    for index, row in enumerate(records):
        identifier = str(field(row, schema["identifier"])) if schema.get("identifier") else resource + "_" + str(index)
        if identifier in seen:
            raise ValueError("Released resource repeats an item identifier")
        seen.add(identifier)
        text = str(field(row, schema["message"]))
        contexts = [(path, field(row, path)) for path in schema["context_fields"]]
        context = "\n".join(path + "\n" + (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
                            for path, value in contexts)
        proposition = str(field(row, schema["proposition"])) if schema.get("proposition") else identifier
        label = str(field(row, schema["state_label"]))
        if label not in mapping:
            raise ValueError("A released state label has no declared binary mapping")
        state = mapping[label]
        if state not in (0, 1) or isinstance(state, bool):
            raise ValueError("State-label mapping uses integer zero and one")
        labels.append(state)
        messages.append(Message(identifier, text, context, proposition,
                                 {"resource": resource, "source_index": index, "split": schema["split"]}))
    positive = float(np.mean(labels))
    game = Game.from_record({"prior": [1 - positive, positive], "states": schema.get("states", ["claim_false", "claim_true"]),
                             "actions": ["against", "undecided", "for"],
                             "sender_utility": [[0, 0], [0.5, 0.5], [1, 1]],
                             "receiver_utility": [[1, 0], [0.6, 0.6], [0, 1]]})
    return game, messages, {"resource": resource, "items": len(records), "positive_label_frequency": positive,
                            "schema_signature": signature(schema), "split": schema["split"]}


def prepare_file(source, schema, resource, output):
    game, messages, metadata = prepare_records(read_records(source), schema, resource)
    metadata["released_file_sha256"] = file_digest(source)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "game.json", game.record())
    write_json(root / "messages.json", messages)
    write_json(root / "groups.json", {key: {"context": pool[0].context, "proposition": pool[0].proposition,
                                            "message_ids": [message.identifier for message in pool]}
                                       for key, pool in partition_messages(messages).items()})
    write_json(root / "resource.json", metadata)
    write_json(root / "schema.json", schema)
    return metadata


def proposition_split(messages, fraction=0.8, seed=0):
    if not 0 < fraction < 1:
        raise ValueError("Proposition training fraction lies between zero and one")
    names = sorted({message.proposition for message in messages})
    if len(names) < 2:
        raise ValueError("Proposition split requires at least two distinct topics")
    order = np.random.default_rng(seed).permutation(len(names))
    count = min(max(int(len(names) * fraction), 1), len(names) - 1)
    training = {names[index] for index in order[:count]}
    return ([message for message in messages if message.proposition in training],
            [message for message in messages if message.proposition not in training])
