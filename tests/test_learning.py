import numpy as np
import pytest
import torch

from cr.learning import acquisition_reward, group_advantages, grpo_loss
from cr.lora import LoRALinear, disabled_adapters

from test_program import game


def test_dual_reward_has_negative_price_sign_and_group_offsets_cancel():
    instance = game()
    prices = np.array([1.0, 2.0])
    assert acquisition_reward(instance, [.2, .8], 2, prices) == pytest.approx(-.8)
    rewards = np.array([1.0, .7, -.2, .4])
    np.testing.assert_allclose(group_advantages(rewards), group_advantages(rewards - 20), atol=1e-12)
    np.testing.assert_array_equal(group_advantages([1, 1, 1, 1]), [0, 0, 0, 0])


def test_grpo_clipping_and_reference_kl_have_completion_token_gradients():
    old = torch.tensor([-.3, -.4])
    current = torch.tensor([-.2, -.5], requires_grad=True)
    reference = torch.tensor([-.35, -.45])
    ratio = torch.exp(current - old)
    surrogate = torch.minimum(ratio * 1.2, torch.clamp(ratio, .8, 1.2) * 1.2)
    gap = reference - current
    expected = -(surrogate - .04 * (torch.exp(gap) - gap - 1)).mean()
    actual = grpo_loss(current, old, reference, 1.2)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert current.grad is not None and torch.isfinite(current.grad).all()


def test_lora_matches_merged_dense_weights_and_freezes_base():
    torch.manual_seed(3)
    base = torch.nn.Linear(4, 3)
    wrapper = LoRALinear(base, rank=2, scale=4)
    with torch.no_grad():
        wrapper.lora_B.normal_()
    inputs = torch.randn(5, 4)
    expected = torch.nn.functional.linear(inputs, base.weight + 2 * wrapper.lora_B @ wrapper.lora_A, base.bias)
    torch.testing.assert_close(wrapper(inputs), expected)
    wrapper(inputs).sum().backward()
    assert wrapper.lora_A.grad is not None and wrapper.lora_B.grad is not None
    assert base.weight.grad is None and base.bias.grad is None
    with disabled_adapters({"a": wrapper}):
        torch.testing.assert_close(wrapper(inputs), base(inputs))
    assert wrapper.enabled
