import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .lora import disabled_adapters, install_lora
from .messages import Message, common_context
from .evaluation import sender_state
from .program import validate_policy
from .prompts import sender_prompt
from .storage import signature


@dataclass(frozen=True)
class Completion:
    input_ids: tuple
    completion_ids: tuple
    text: str
    seed: int


class CausalBackend:
    def __init__(self, model, tokenizer, checkpoint, revision=None, max_input_tokens=1024,
                 max_completion_tokens=256, enable_thinking=False):
        if max_input_tokens < 1 or max_completion_tokens < 1:
            raise ValueError("Language-model token budgets are positive")
        self.model, self.tokenizer = model, tokenizer
        self.checkpoint, self.revision = checkpoint, revision
        self.max_input_tokens, self.max_completion_tokens = int(max_input_tokens), int(max_completion_tokens)
        self.enable_thinking = bool(enable_thinking)
        self.adapters = {}
        self.data_kind = "measured_receiver"
        self.version = 0
        self.model.eval()
        self.model.requires_grad_(False)

    @classmethod
    def from_pretrained(cls, checkpoint, revision=None, device_map="auto", dtype="bfloat16", **options):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
        configuration = AutoConfig.from_pretrained(checkpoint, revision=revision)
        model_class = AutoModelForImageTextToText if configuration.model_type == "gemma3" else AutoModelForCausalLM
        model = model_class.from_pretrained(checkpoint, revision=revision, device_map=device_map,
                                            torch_dtype=getattr(torch, dtype), attn_implementation="eager")
        if any(parameter.is_meta for parameter in model.parameters()):
            raise ValueError("Sender updates require resident model tensors")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
        resolved = getattr(model.config, "_commit_hash", None) or revision
        return cls(model, tokenizer, checkpoint, resolved, **options)

    @property
    def identity(self):
        return {"kind": "local_causal", "checkpoint": self.checkpoint, "revision": self.revision,
                "version": self.version, "max_input_tokens": self.max_input_tokens,
                "max_completion_tokens": self.max_completion_tokens, "enable_thinking": self.enable_thinking}

    def encode(self, messages):
        messages = list(messages)
        kind = getattr(self.model.config, "model_type", "")
        if kind in {"gemma3", "gemma3_text"} and messages and messages[0]["role"] == "system":
            instruction = messages[0]["content"]
            messages = messages[1:]
            if not messages or messages[0]["role"] != "user":
                raise ValueError("Gemma text prompts begin with a user message")
            messages[0] = dict(messages[0], content=instruction + "\n\n" + messages[0]["content"])
        options = {"enable_thinking": self.enable_thinking} if kind == "qwen3" else {}
        identifiers = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **options)
        if len(identifiers) > self.max_input_tokens:
            raise ValueError("Prompt exceeds the declared input-token budget")
        if not identifiers:
            raise ValueError("Chat template returned an empty prompt")
        return tuple(identifiers)

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    def sample(self, messages, temperature=1.0, seed=0, max_tokens=None):
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError("Generation temperature is finite and nonnegative")
        prompt = self.encode(messages)
        identifiers = torch.tensor([prompt], dtype=torch.long, device=self.device)
        devices = sorted({parameter.device.index for parameter in self.model.parameters() if parameter.device.type == "cuda"})
        training = self.model.training
        self.model.eval()
        try:
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                torch.random.default_generator.manual_seed(seed)
                for device in devices:
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(seed)
                options = {"input_ids": identifiers, "attention_mask": torch.ones_like(identifiers),
                           "max_new_tokens": int(max_tokens or self.max_completion_tokens),
                           "do_sample": temperature > 0, "use_cache": True, "num_beams": 1,
                           "repetition_penalty": 1.0}
                if temperature > 0:
                    options.update(temperature=temperature, top_p=1.0, top_k=0)
                if self.tokenizer.pad_token_id is not None:
                    options["pad_token_id"] = self.tokenizer.pad_token_id
                output = self.model.generate(**options)[0, len(prompt):].tolist()
        finally:
            self.model.train(training)
        if not output:
            raise ValueError("Sender generation returned an empty completion")
        text = self.tokenizer.decode(output, skip_special_tokens=True).strip()
        return Completion(prompt, tuple(output), text, int(seed))

    def complete(self, messages, temperature=0.0, seed=0, max_tokens=None):
        completion = self.sample(messages, temperature, seed, max_tokens)
        return {"text": completion.text, "usage": {"prompt_tokens": len(completion.input_ids),
                                                     "completion_tokens": len(completion.completion_ids)}}

    def token_log_probabilities(self, completion):
        sequence = completion.input_ids + completion.completion_ids
        identifiers = torch.tensor([sequence], dtype=torch.long, device=self.device)
        arguments = {"input_ids": identifiers, "attention_mask": torch.ones_like(identifiers), "use_cache": False}
        parameters = inspect.signature(self.model.forward).parameters
        count = len(completion.completion_ids)
        if "logits_to_keep" in parameters:
            arguments["logits_to_keep"] = count + 1
        elif "num_logits_to_keep" in parameters:
            arguments["num_logits_to_keep"] = count + 1
        logits = self.model(**arguments).logits[0].float()
        start = logits.shape[0] - count - 1
        if start < 0:
            raise ValueError("Model logits omitted a completion prediction")
        # The prompt's final-position logit predicts the first completion token.
        logits = logits[start:start + count]
        target = torch.tensor(completion.completion_ids, dtype=torch.long, device=logits.device)
        return torch.log_softmax(logits, dim=-1).gather(1, target[:, None]).squeeze(1)

    def enable_lora(self, rank=16, scale=32.0, seed=0, checkpointing=True):
        if self.adapters:
            raise ValueError("Sender adapters have already been installed")
        devices = sorted({parameter.device.index for parameter in self.model.parameters() if parameter.device.type == "cuda"})
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            for device in devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            self.adapters = install_lora(self.model, rank, scale)
        if checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return self.adapters

    def reference_log_probabilities(self, completion):
        # The frozen base policy supplies the KL anchor with sender adapters disabled.
        with disabled_adapters(self.adapters), torch.no_grad():
            return self.token_log_probabilities(completion).detach()

    def save_adapter(self, destination):
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for index, (name, module) in enumerate(sorted(self.adapters.items())):
            arrays["A_" + str(index)] = module.lora_A.detach().float().cpu().numpy()
            arrays["B_" + str(index)] = module.lora_B.detach().float().cpu().numpy()
        metadata = {"format": "cr_lora_v1", "backend": self.identity,
                    "blocks": list(sorted(self.adapters)), "rank": next(iter(self.adapters.values())).rank,
                    "scale": next(iter(self.adapters.values())).scale,
                    "vocabulary_sha256": signature(self.tokenizer.get_vocab())}
        np.savez_compressed(path, **arrays, metadata=np.asarray(json.dumps(metadata)))

    def load_adapter(self, source):
        with np.load(source, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("format") != "cr_lora_v1" or metadata["backend"]["checkpoint"] != self.checkpoint:
                raise ValueError("Sender adapter has a different format or base checkpoint")
            if metadata["backend"].get("revision") != self.revision:
                raise ValueError("Sender adapter has a different checkpoint revision")
            if metadata["vocabulary_sha256"] != signature(self.tokenizer.get_vocab()):
                raise ValueError("Sender adapter has a different entry vocabulary")
            if not self.adapters:
                self.enable_lora(metadata["rank"], metadata["scale"])
            if sorted(self.adapters) != metadata["blocks"]:
                raise ValueError("Sender adapter has a different attention block set")
            with torch.no_grad():
                for index, (name, module) in enumerate(sorted(self.adapters.items())):
                    for prefix, parameter in (("A_", module.lora_A), ("B_", module.lora_B)):
                        array = archive[prefix + str(index)]
                        if array.shape != tuple(parameter.shape) or not np.isfinite(array).all():
                            raise ValueError("Sender adapter contains invalid parameter arrays")
                        parameter.copy_(torch.tensor(array, device=parameter.device, dtype=parameter.dtype))
            self.version = int(metadata["backend"]["version"])

    def close(self):
        pass


class LanguageGenerator:
    def __init__(self, backend, game, temperature=1.0):
        if temperature != 1.0:
            raise ValueError("The sender's scored sampling distribution uses temperature one")
        self.backend, self.game = backend, game
        self.temperature = float(temperature)

    def propose(self, contexts, count, seed, free=False, states=None):
        if not contexts or count < 1:
            raise ValueError("Proposal generation requires contexts and a positive batch size")
        result = []
        if states is not None and len(states) != count:
            raise ValueError("Private sender states align with the proposed messages")
        sampled_states = np.random.default_rng(seed).choice(len(self.game.prior), size=count, p=self.game.prior)
        for index in range(count):
            context, proposition = contexts[index % len(contexts)]
            state = int(sampled_states[index]) if states is None else sender_state(self.game, states[index])
            completion = self.backend.sample(sender_prompt(self.game, context, state, free), self.temperature, seed + index)
            if completion.text:
                result.append(Message.generated(completion.text, context, proposition,
                                                {"sampled_state": self.game.states[state], "sender_seed": seed + index,
                                                 "completion_ids": completion.completion_ids,
                                                 "sender_identity": self.backend.identity}))
        return result

    def proposed_policy(self, messages):
        context, proposition = common_context(messages)
        eos = self.backend.tokenizer.eos_token_id
        if eos is None:
            raise ValueError("Finite-pool sequence scoring requires an explicit termination token")
        scores = np.empty((len(self.game.prior), len(messages)))
        with torch.no_grad():
            for state in range(len(self.game.prior)):
                for index, message in enumerate(messages):
                    prompt = self.backend.encode(sender_prompt(self.game, context, state))
                    content = tuple(self.backend.tokenizer.encode(message.text, add_special_tokens=False)) + (int(eos),)
                    if len(content) < 2 or len(content) > self.backend.max_completion_tokens:
                        raise ValueError("Finite-pool sequence exceeds the declared completion-token budget")
                    scores[state, index] = float(self.backend.token_log_probabilities(Completion(prompt, content, message.text, 0)).sum())
        scores -= scores.max(axis=1, keepdims=True)
        weights = np.exp(scores)
        return validate_policy(weights / weights.sum(axis=1, keepdims=True), len(self.game.prior), len(messages))
