from dataclasses import dataclass

import numpy as np
import torch

from .measurement import BudgetExhausted, ReceiverChanged
from .messages import Message, context_group
from .prompts import sender_prompt
from .storage import signature


def acquisition_reward(game, posterior, action, state_prices=None):
    belief = np.asarray(posterior, dtype=float)
    payoff = float(game.sender_utility[action] @ belief)
    if state_prices is None:
        return payoff
    prices = np.asarray(state_prices, dtype=float)
    if prices.shape != belief.shape or not np.isfinite(prices).all():
        raise ValueError("Acquisition prices have the measured posterior dimensions")
    return payoff - float(prices @ belief)


def group_advantages(rewards, epsilon=1e-4):
    rewards = np.asarray(rewards, dtype=float)
    if rewards.ndim != 1 or len(rewards) < 2 or not np.isfinite(rewards).all() or epsilon <= 0:
        raise ValueError("Group-relative rewards have at least two finite samples")
    return (rewards - rewards.mean()) / (rewards.std(ddof=0) + epsilon)


def grpo_loss(current, preceding, reference, advantage, clip=0.2, kl_weight=0.04):
    if current.shape != preceding.shape or current.shape != reference.shape or current.ndim != 1 or not len(current):
        raise ValueError("GRPO likelihoods have matching completion-token dimensions")
    if not 0 < clip < 1 or kl_weight < 0:
        raise ValueError("GRPO clipping and KL settings are valid")
    # Tokenwise ratios align clipping and the KL penalty with generated completion tokens.
    ratio = torch.exp(current - preceding)
    surrogate = torch.minimum(ratio * advantage, torch.clamp(ratio, 1 - clip, 1 + clip) * advantage)
    difference = reference - current
    divergence = torch.exp(difference) - difference - 1
    loss = -(surrogate - kl_weight * divergence).mean()
    if not torch.isfinite(loss):
        raise ValueError("GRPO loss is nonfinite")
    return loss


@dataclass(frozen=True)
class RewardedCompletion:
    completion: object
    old_log_probabilities: torch.Tensor
    reference_log_probabilities: torch.Tensor
    reward: float


class GRPOTrainer:
    def __init__(self, generator, engine, settings, seed=0):
        self.generator, self.engine, self.settings, self.seed = generator, engine, dict(settings), int(seed)
        self.backend = generator.backend
        if self.backend is engine.backend or getattr(engine.backend, "model", None) is self.backend.model:
            raise ValueError("Sender training and receiver measurement use separate model instances")
        if not self.backend.adapters:
            self.backend.enable_lora(settings.get("rank", 16), settings.get("scale", 32), seed,
                                     settings.get("activation_checkpointing", True))
        parameters = [parameter for parameter in self.backend.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.AdamW(parameters, lr=settings.get("learning_rate", 2e-5),
                                          betas=(0.9, 0.999), weight_decay=settings.get("weight_decay", 0.0))
        self.completed_steps = 0

    def group(self, context, proposition, solution, exposure, round_id, seed, payoff=False, classical=False, repeats=1):
        count = int(self.settings.get("group_size", 8))
        if count < 2:
            raise ValueError("GRPO uses at least two generations per context")
        samples = []
        prices = None if payoff else solution.state_prices.copy()
        # Sharing one state across a prompt group makes sampled-message rewards comparable.
        state = int(np.random.default_rng(seed).choice(len(self.generator.game.prior), p=self.generator.game.prior))
        for index in range(count):
            if solution.receiver_signature != self.engine.response_signature(exposure, repeats):
                raise ReceiverChanged("GRPO prices differ from the active measurement protocol")
            completion = self.backend.sample(sender_prompt(self.generator.game, context, state), 1.0, seed + index)
            if not completion.text:
                continue
            message = Message.generated(completion.text, context, proposition)
            reading = self.engine.measure(message, "acquisition", repeats, round_id, exposure)
            if reading is None:
                continue
            if reading.receiver_signature != solution.receiver_signature:
                raise ReceiverChanged("GRPO reward reading differs from its frozen dual protocol")
            reward = acquisition_reward(self.generator.game, reading.posterior, reading.action, prices)
            if classical:
                reward = self.generator.game.classical_value(reading.posterior)
                if prices is not None:
                    reward -= float(prices @ reading.posterior)
            with torch.no_grad():
                old = self.backend.token_log_probabilities(completion).detach()
                reference = self.backend.reference_log_probabilities(completion)
            samples.append(RewardedCompletion(completion, old, reference, reward))
        return samples

    def update(self, contexts, solution, exposure, round_id, payoff=False, stage_limit=None, classical=False, pool_repeats=1):
        if not contexts:
            raise ValueError("Sender update requires training contexts")
        if stage_limit is not None:
            self.engine.ledger.set_stage_limit("acquisition", round_id, stage_limit)
        groups_per_batch = int(self.settings.get("rollout_batch", 8))
        accumulation = int(self.settings.get("gradient_accumulation", 2))
        updates = int(self.settings.get("updates_per_round", 2))
        if min(groups_per_batch, accumulation, updates) < 1:
            raise ValueError("GRPO batch, accumulation, and update counts are positive")
        total_steps = int(self.settings.get("total_optimizer_steps", updates * 4))
        records = []
        for update in range(updates):
            self.optimizer.zero_grad(set_to_none=True)
            batch_groups = []
            exhausted = False
            try:
                for index in range(groups_per_batch * accumulation):
                    context, proposition = contexts[(round_id * groups_per_batch * accumulation + index) % len(contexts)]
                    identifier = context_group(context, proposition)
                    prices = solution[identifier] if isinstance(solution, dict) else solution
                    information = exposure[identifier] if isinstance(exposure, dict) else exposure
                    repeats = pool_repeats[identifier] if isinstance(pool_repeats, dict) else pool_repeats
                    group = self.group(context, proposition, prices, information, round_id,
                                       self.seed + 100000 * round_id + 10000 * update + 100 * index, payoff, classical, repeats)
                    if len(group) >= 2:
                        batch_groups.append(group)
            except BudgetExhausted:
                exhausted = True
            if not batch_groups:
                return records
            self.backend.model.train()
            losses, rewards = [], []
            for group in batch_groups:
                advantages = group_advantages([sample.reward for sample in group], self.settings.get("advantage_epsilon", 1e-4))
                for sample, advantage in zip(group, advantages):
                    current = self.backend.token_log_probabilities(sample.completion)
                    loss = grpo_loss(current, sample.old_log_probabilities, sample.reference_log_probabilities,
                                     float(advantage), self.settings.get("clip", 0.2), self.settings.get("kl_weight", 0.04))
                    # Scaling by group size gives each sampled group equal weight in the batch loss.
                    (loss / (len(batch_groups) * len(group))).backward()
                    losses.append(float(loss.detach()))
                    rewards.append(sample.reward)
            parameters = [parameter for parameter in self.backend.model.parameters() if parameter.requires_grad]
            torch.nn.utils.clip_grad_norm_(parameters, self.settings.get("gradient_norm", 1.0), error_if_nonfinite=True)
            progress = min(self.completed_steps / max(total_steps, 1), 1)
            rate = self.settings.get("learning_rate", 2e-5) * 0.5 * (1 + np.cos(np.pi * progress))
            for group in self.optimizer.param_groups:
                group["lr"] = float(rate)
            self.optimizer.step()
            self.completed_steps += 1
            self.backend.version += 1
            self.backend.model.eval()
            records.append({"optimizer_step": self.completed_steps, "groups": len(batch_groups),
                            "completions": len(rewards), "mean_reward": float(np.mean(rewards)),
                            "mean_loss": float(np.mean(losses)), "learning_rate": float(rate),
                            "frozen_dual_signature": signature({key: {"lambda": item.state_prices, "nu": item.normalization_price}
                                                               for key, item in solution.items()} if isinstance(solution, dict)
                                                              else {"lambda": solution.state_prices, "nu": solution.normalization_price}),
                            "receiver_checkpoint_signature": self.engine.checkpoint_signature})
            if exhausted:
                return records
        return records
