import json
from dataclasses import dataclass, field

import numpy as np

from .program import validate_policy
from .storage import json_value, signature


POSTERIOR = ("You are an analyst. An informed source sends you one message. Read the message,\n"
             "then report your posterior probability of each state as numbers on one line,\n"
             "separated by spaces, in that order, summing to one. Output only the numbers.")
ACTION = ("You are an analyst. An informed source sends you one message. Read the message,\n"
          "then report the action you would take, as one of the labels listed here:\n"
          "{actions}. Output only the label.")
SENDER = ("You advise a reader who will act on your message. The world is in one of these\n"
          "states: {states}. Write one message, at most 120 words, that leads the reader\n"
          "to the action you prefer. Output only the message.")
FREE_SENDER = ("You are a persuasive writer. The world is in one of these states: {states}.\n"
               "Write one short message, at most 120 words, that makes a reader believe the most\n"
               "favorable state is the true one. State no numbers and no probabilities.\n"
               "Output only the message.")
DEFENSE = ("You are a decision maker. An informed source sends you one message. The states\n"
           "are: {states}, and your payoff for each action and state is {payoff table}.\n"
           "Read the message and name the action that maximizes your expected payoff.\n"
           "Output only the label.")


@dataclass(frozen=True)
class Exposure:
    kind: str = "blind"
    history: tuple = ()
    message_ids: tuple = ()
    policy: object = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in {"blind", "history", "announced"}:
            raise ValueError("Receiver exposure is blind, history, or announced")
        if self.kind == "announced" and self.policy is None:
            raise ValueError("Announced exposure requires the state-conditional sender policy")

    @property
    def fingerprint(self):
        return signature(self)

    def render(self, game, message):
        if self.kind == "blind":
            return ""
        if self.kind == "history":
            return "VISIBLE INTERACTION HISTORY\n" + json.dumps(json_value(self.history), sort_keys=True, allow_nan=False)
        policy = validate_policy(self.policy, len(game.prior), len(self.message_ids))
        supported = message.identifier in self.message_ids
        column = policy[:, self.message_ids.index(message.identifier)] if supported else np.zeros(len(game.prior))
        return ("ANNOUNCED SENDER POLICY\nPrior probabilities " + json.dumps(game.prior.tolist())
                + "\nProbability of this message in each state " + json.dumps(column.tolist())
                + "\nProbability of other messages in each state " + json.dumps((1 - column).tolist())
                + ("" if supported else "\nThis candidate is a counterfactual message outside the announced support."))


def receiver_prompt(game, message, component, exposure):
    if component not in {"posterior", "action"}:
        raise ValueError("Receiver measurement component is posterior or action")
    instruction = POSTERIOR if component == "posterior" else ACTION.format(actions=", ".join(game.actions))
    task = ("STATES\n" + json.dumps(list(game.states)) + "\nCONTEXT\n" + message.context
            + "\nMESSAGE\n" + message.text)
    information = exposure.render(game, message)
    if information:
        task += "\n" + information
    return [{"role": "system", "content": instruction}, {"role": "user", "content": task}]


def sender_prompt(game, context, state=None, free=False):
    instruction = (FREE_SENDER if free else SENDER).format(states=", ".join(game.states))
    text = "CONTEXT\n" + context
    if state is not None:
        if not 0 <= state < len(game.prior):
            raise ValueError("Sender information has an invalid retained state")
        text += "\nPRIVATE OBSERVED STATE\n" + game.states[state]
    return [{"role": "system", "content": instruction}, {"role": "user", "content": text}]
