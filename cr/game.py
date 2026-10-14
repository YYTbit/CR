from dataclasses import dataclass

import numpy as np


def probability_vector(value, size=None, tolerance=1e-9):
    vector = np.array(value, dtype=float, copy=True)
    if vector.ndim != 1 or not len(vector) or (size is not None and len(vector) != size):
        raise ValueError("Probability vector has incompatible dimensions")
    if not np.isfinite(vector).all() or np.any(vector < 0) or abs(vector.sum() - 1) > tolerance:
        raise ValueError("Probability vector is finite, nonnegative, and normalized")
    return vector


@dataclass(frozen=True)
class Game:
    prior: np.ndarray
    sender_utility: np.ndarray
    receiver_utility: np.ndarray
    states: tuple
    actions: tuple

    def __post_init__(self):
        prior = probability_vector(self.prior)
        sender = np.array(self.sender_utility, dtype=float, copy=True)
        receiver = np.array(self.receiver_utility, dtype=float, copy=True)
        if np.any(prior <= 0):
            raise ValueError("Retained states have positive prior mass")
        if sender.shape != (len(self.actions), len(prior)) or receiver.shape != sender.shape:
            raise ValueError("Utility tables have action-by-state dimensions")
        if len(self.states) != len(prior) or len(set(self.states)) != len(self.states):
            raise ValueError("State names have distinct retained coordinates")
        if not self.actions or len(set(self.actions)) != len(self.actions):
            raise ValueError("Action names are nonempty and distinct")
        if not np.isfinite(sender).all() or not np.isfinite(receiver).all():
            raise ValueError("Utility tables are finite")
        for name, value in (("prior", prior), ("sender_utility", sender), ("receiver_utility", receiver)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "actions", tuple(self.actions))

    @classmethod
    def from_record(cls, record):
        prior = probability_vector(record["prior"])
        active = prior > 0
        return cls(prior[active], np.asarray(record["sender_utility"])[:, active],
                   np.asarray(record["receiver_utility"])[:, active],
                   tuple(np.asarray(record["states"])[active]), tuple(record["actions"]))

    def best_responses(self, posterior, tolerance=1e-9):
        posterior = probability_vector(posterior, len(self.prior))
        values = self.receiver_utility @ posterior
        return tuple(np.flatnonzero(values >= np.max(values) - tolerance).tolist())

    def classical_value(self, posterior):
        posterior = probability_vector(posterior, len(self.prior))
        return float(max((self.sender_utility @ posterior)[list(self.best_responses(posterior))]))

    def record(self):
        return {"prior": self.prior.tolist(), "sender_utility": self.sender_utility.tolist(),
                "receiver_utility": self.receiver_utility.tolist(),
                "states": list(self.states), "actions": list(self.actions)}


@dataclass(frozen=True)
class MeasuredChannel:
    game: Game
    message_ids: tuple
    posteriors: np.ndarray
    actions: np.ndarray
    receiver_signature: str = ""

    def __post_init__(self):
        beliefs = np.array(self.posteriors, dtype=float, copy=True)
        actions = np.asarray(self.actions)
        if beliefs.shape != (len(self.message_ids), len(self.game.prior)) or not len(self.message_ids):
            raise ValueError("Measured channel has message-by-state posterior dimensions")
        if len(set(self.message_ids)) != len(self.message_ids):
            raise ValueError("Measured message identifiers are distinct")
        if actions.shape != (len(self.message_ids),) or actions.dtype.kind not in "iu":
            raise ValueError("Measured actions are integer indices")
        if np.any(actions < 0) or np.any(actions >= len(self.game.actions)):
            raise ValueError("Measured action lies outside the declared action set")
        if not np.isfinite(beliefs).all() or np.any(beliefs < 0) or not np.allclose(beliefs.sum(axis=1), 1, atol=1e-9, rtol=0):
            raise ValueError("Measured posterior rows are probability vectors")
        actions = np.array(actions, dtype=int, copy=True)
        beliefs.setflags(write=False)
        actions.setflags(write=False)
        object.__setattr__(self, "posteriors", beliefs)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "message_ids", tuple(self.message_ids))

    @property
    def payoffs(self):
        return np.einsum("ms,ms->m", self.game.sender_utility[self.actions], self.posteriors)

    @property
    def classical_payoffs(self):
        return np.asarray([self.game.classical_value(posterior) for posterior in self.posteriors])

    def subset(self, identifiers):
        indices = [self.message_ids.index(identifier) for identifier in identifiers]
        return MeasuredChannel(self.game, tuple(identifiers), self.posteriors[indices], self.actions[indices], self.receiver_signature)
