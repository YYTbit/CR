from dataclasses import dataclass, field

from .storage import signature


@dataclass(frozen=True)
class Message:
    identifier: str
    text: str
    context: str
    proposition: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.identifier, str) or not self.identifier or not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("A message has a nonempty identifier and text")
        if not isinstance(self.context, str) or not isinstance(self.proposition, str):
            raise ValueError("Message context and proposition identifiers are strings")

    @property
    def fingerprint(self):
        return signature({"text": self.text, "context": self.context, "proposition": self.proposition})

    @property
    def group_id(self):
        return context_group(self.context, self.proposition)

    @classmethod
    def generated(cls, text, context, proposition="", metadata=None):
        digest = signature({"text": text, "context": context, "proposition": proposition})
        return cls("generated_" + digest[:24], text, context, proposition, metadata or {})

    @classmethod
    def from_record(cls, record):
        return cls(record["identifier"], record["text"], record["context"],
                   record.get("proposition", ""), record.get("metadata", {}))


def context_group(context, proposition):
    return signature({"context": context, "proposition": proposition})


def partition_messages(messages):
    groups = {}
    for message in messages:
        groups.setdefault(message.group_id, []).append(message)
    return groups


def common_context(messages):
    rows = list(messages)
    if not rows or len({message.group_id for message in rows}) != 1:
        raise ValueError("A measured channel conditions on one context and proposition")
    return rows[0].context, rows[0].proposition
