from contextlib import contextmanager

from torch import nn
from peft.tuners.lora.layer import Linear


class LoRALinear(nn.Module):
    """PEFT projection with the sender archive's matrix accessors."""

    def __init__(self, base, rank=16, scale=32.0):
        super().__init__()
        if not isinstance(base, nn.Linear) or isinstance(rank, bool) or int(rank) != rank or rank < 1 or scale <= 0:
            raise ValueError("LoRA projection has a linear base, positive rank, and positive scale")
        base.requires_grad_(False)
        self.rank, self.scale = int(rank), float(scale)
        self.layer = Linear(base, adapter_name="default", r=self.rank, lora_alpha=self.scale,
                            lora_dropout=0.0, init_lora_weights=True, lora_bias=False)

    @property
    def base(self):
        return self.layer.get_base_layer()

    @property
    def lora_A(self):
        return self.layer.lora_A["default"].weight

    @property
    def lora_B(self):
        return self.layer.lora_B["default"].weight

    @property
    def enabled(self):
        return not self.layer.disable_adapters

    @enabled.setter
    def enabled(self, value):
        self.layer.enable_adapters(bool(value))

    def forward(self, inputs):
        return self.layer(inputs)


def install_lora(model, rank=16, scale=32.0):
    model.requires_grad_(False)
    modules = {}
    for name, layer in list(model.named_modules()):
        if getattr(model.config, "model_type", "") == "gemma3" and "language_model." not in name:
            continue
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
