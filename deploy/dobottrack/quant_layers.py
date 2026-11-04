import torch
from torch import nn


class QDQLinear(nn.Module):
    def __init__(self, source, amax):
        super().__init__()
        self.weight = source.weight
        self.bias = source.bias
        self.register_buffer(
            "weight_scale", source.weight.detach().abs().amax(1).clamp_min(1e-8) / 127
        )
        self.register_buffer(
            "weight_zero",
            torch.zeros(
                source.weight.shape[0], dtype=torch.int32, device=source.weight.device
            ),
        )
        self.register_buffer(
            "activation_scale",
            torch.tensor(max(amax, 1e-8) / 127, device=source.weight.device),
        )
        self.register_buffer(
            "activation_zero",
            torch.zeros((), dtype=torch.int32, device=source.weight.device),
        )

    def forward(self, x):
        x = torch.fake_quantize_per_tensor_affine(
            x, self.activation_scale, self.activation_zero, -128, 127
        )
        w = torch.fake_quantize_per_channel_affine(
            self.weight, self.weight_scale, self.weight_zero, 0, -128, 127
        )
        return nn.functional.linear(x, w, self.bias)


def install_qdq(model, ranges):
    for name, value in ranges.items():
        parent, key = name.rsplit(".", 1)
        container = model.get_submodule(parent)
        source = getattr(container, key)
        setattr(container, key, QDQLinear(source, value))
