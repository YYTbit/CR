import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, Gemma3TextConfig, LlamaConfig, Qwen3Config

from cr.models import CausalBackend, Completion
from cr.learning import GRPOTrainer
from cr.measurement import MeasurementEngine, QueryLedger
from cr.models import LanguageGenerator
from cr.program import solve_sender
from cr.prompts import Exposure

from test_adaptive import ReceiverFixture, catalog
from test_program import game


class TokenizerFixture:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        return [1, 3, 4]

    def decode(self, identifiers, skip_special_tokens):
        return " ".join("token" + str(value) for value in identifiers)

    def get_vocab(self):
        return {"token" + str(index): index for index in range(32)}


def model_config(kind):
    common = {"vocab_size": 32, "hidden_size": 8, "intermediate_size": 12,
              "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
              "max_position_embeddings": 64, "head_dim": 4,
              "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0}
    classes = {"qwen3": Qwen3Config, "llama": LlamaConfig, "gemma3": Gemma3TextConfig}
    config = classes[kind](**common)
    config._attn_implementation = "eager"
    return config


@pytest.mark.parametrize("kind", ["qwen3", "llama", "gemma3"])
def test_completion_likelihood_matches_full_teacher_forcing_and_adapter_roundtrip(kind, tmp_path):
    torch.manual_seed(7)
    model = AutoModelForCausalLM.from_config(model_config(kind))
    backend = CausalBackend(model, TokenizerFixture(), "software_fixture", max_input_tokens=32, max_completion_tokens=2)
    completion = Completion((1, 3, 4), (5, 6, 2), "fixture", 0)
    sequence = torch.tensor([completion.input_ids + completion.completion_ids])
    with torch.no_grad():
        full = model(input_ids=sequence, use_cache=False).logits[0, 2:5].float()
        expected = torch.log_softmax(full, -1).gather(1, torch.tensor([5, 6, 2])[:, None]).squeeze()
    torch.testing.assert_close(backend.token_log_probabilities(completion), expected)
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    backend.enable_lora(rank=2, scale=4, checkpointing=False)
    assert len(backend.adapters) == 4
    with torch.no_grad():
        for wrapper in backend.adapters.values():
            wrapper.lora_B.normal_(std=.05)
    loss = -backend.token_log_probabilities(completion).mean()
    loss.backward()
    assert all(parameter.requires_grad == ("lora_" in name) for name, parameter in model.named_parameters())
    for name, parameter in model.named_parameters():
        if ".base." in name:
            source = name.replace(".base.", ".")
            torch.testing.assert_close(parameter, original[source])
    reference = backend.reference_log_probabilities(completion)
    torch.testing.assert_close(reference, expected)
    before = backend.token_log_probabilities(completion).detach()
    backend.save_adapter(tmp_path / "adapter.npz")
    with torch.no_grad():
        for wrapper in backend.adapters.values():
            wrapper.lora_B.zero_()
    backend.load_adapter(tmp_path / "adapter.npz")
    torch.testing.assert_close(backend.token_log_probabilities(completion), before)


def test_grpo_optimizer_uses_real_adapter_gradients_and_retains_completed_groups_at_query_boundary():
    torch.manual_seed(3)
    model = AutoModelForCausalLM.from_config(model_config("qwen3"))
    backend = CausalBackend(model, TokenizerFixture(), "software_fixture", max_input_tokens=32, max_completion_tokens=2)
    def sample(prompt, temperature, seed):
        assert "PRIVATE OBSERVED STATE" in prompt[1]["content"]
        return Completion((1, 3, 4), (5 if seed % 2 == 0 else 6, 2), "candidate" if seed % 2 == 0 else "anchor_against", seed)
    backend.sample = sample
    instance = game()
    engine = MeasurementEngine(instance, ReceiverFixture(), QueryLedger(100))
    retained, readings = engine.pool(catalog(), "shared_pool")
    solution = solve_sender(engine.channel(readings))
    settings = {"rank": 2, "scale": 4, "group_size": 2, "rollout_batch": 3,
                "gradient_accumulation": 1, "updates_per_round": 2, "activation_checkpointing": False,
                "total_optimizer_steps": 2, "learning_rate": .01}
    trainer = GRPOTrainer(LanguageGenerator(backend, instance), engine, settings)
    updates = trainer.update([("context", "topic")], solution, Exposure(), 0, stage_limit=10)
    assert len(updates) == 1 and updates[0]["completions"] == 4
    assert trainer.completed_steps == 1 and backend.version == 1
    assert engine.ledger.charged == 10
    assert any(torch.count_nonzero(module.lora_B).item() > 0 for module in backend.adapters.values())
    assert all(not parameter.requires_grad for module in backend.adapters.values() for parameter in module.base.parameters())
