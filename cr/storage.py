import hashlib
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def signature(value):
    text = json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix="record-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(json_value(value), stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(json_value(value), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_channel(path, game, messages, measurements, metadata=None):
    write_json(path, {"format": "cr_measured_channel_v1", "game": game.record(),
                      "messages": messages, "measurements": measurements, "metadata": metadata or {}})


def read_channel(path):
    from .game import Game, MeasuredChannel
    from .messages import Message, common_context
    record = json.loads(Path(path).read_text())
    if record.get("format") != "cr_measured_channel_v1":
        raise ValueError("Unsupported measured-channel archive format")
    game = Game.from_record(record["game"])
    messages = [Message.from_record(row) for row in record["messages"]]
    common_context(messages)
    observations = record["measurements"]
    identities = [row["message_id"] for row in observations]
    signatures = {row["receiver_signature"] for row in observations}
    if len(signatures) != 1 or set(identities) != {message["identifier"] for message in record["messages"]}:
        raise ValueError("Channel archive mixes response protocols or message sets")
    lookup = {message.identifier: message for message in messages}
    if any(row["message_fingerprint"] != lookup[row["message_id"]].fingerprint for row in observations):
        raise ValueError("Channel archive has mismatched message fingerprints")
    channel = MeasuredChannel(game, tuple(identities), [row["posterior"] for row in observations],
                              np.asarray([row["action"] for row in observations]), next(iter(signatures)))
    return channel, record


def read_channel_collection(path):
    path = Path(path)
    record = json.loads(path.read_text())
    if record.get("format") != "cr_group_channels_v1":
        channel, archive = read_channel(path)
        from .messages import Message
        identifier = Message.from_record(archive["messages"][0]).group_id
        return {identifier: (channel, archive)}
    result = {}
    for identifier, filename in record["groups"].items():
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Channel manifests use relative paths within their directory")
        channel, archive = read_channel(path.parent / relative)
        from .messages import Message
        if channel.game.record() != record["game"] or Message.from_record(archive["messages"][0]).group_id != identifier:
            raise ValueError("Channel manifest has inconsistent game or context identifiers")
        result[identifier] = (channel, archive)
    if not result:
        raise ValueError("A resource channel manifest has measured groups")
    return result
