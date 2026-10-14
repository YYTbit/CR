from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, scale=32.0):
        super().__init__()
        if not isinstance(base, nn.Linear) or isinstance(rank, bool) or int(rank) != rank or rank < 1 or scale <= 0:
            raise ValueError("LoRA projection has a linear base, positive rank, and positive scale")
        self.base = base
        self.base.requires_grad_(False)
        self.rank, self.scale = int(rank), float(scale)
        self.lora_A = nn.Parameter(base.weight.new_empty((rank, base.in_features)))
        self.lora_B = nn.Parameter(base.weight.new_zeros((base.out_features, rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.enabled = True

    def forward(self, inputs):
        output = self.base(inputs)
        if self.enabled:
            output = output + self.scale / self.rank * functional.linear(functional.linear(inputs, self.lora_A), self.lora_B)
        return output


def install_lora(model, rank=16, scale=32.0):
    model.requires_grad_(False)
    modules = {}
    for name, layer in list(model.named_modules()):
        if isinstance(layer, nn.Linear) and name.endswith(("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")):
            parent, child = name.rsplit(".", 1)
            wrapper = LoRALinear(layer, rank, scale)
            setattr(model.get_submodule(parent), child, wrapper)
            modules[name] = wrapper
    if not modules:
        raise ValueError("Sender model exposes no declared attention projection")
    return modules


@contextmanager
def disabled_adapters(modules):
    previous = {name: module.enabled for name, module in modules.items()}
    try:
        for module in modules.values():
            module.enabled = False
        yield
    finally:
        for name, module in modules.items():
            module.enabled = previous[name]
