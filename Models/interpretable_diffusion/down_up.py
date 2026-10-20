from Models.interpretable_diffusion import up
import torch

class Down_Up(up.Adaptation_Up):
    def __init__(self, in_features: int, *args, **kwargs):
        up.Adaptation_Up.__init__(self, *args, **kwargs)
        self.in_features = in_features
        self.register_buffer('scale2', None, persistent=False)

    def assign_adaptation(self, adaptation):
        if adaptation is None:
            self.scale, self.scale2, self.shift = None, None, None
        else:
            self.scale = adaptation[..., :self.out_features] + 1
            self.scale2 = adaptation[..., self.out_features:self.out_features + self.in_features] + 1
            if self.flag_adapt_bias:
                self.shift = adaptation[..., -self.out_features:]
            if self.scale.dim() == 2:
                self.scale = self.scale.unsqueeze(1)
                self.scale2 = self.scale2.unsqueeze(1)
                self.shift = self.shift.unsqueeze(1)

    def _merge(self, weight, bias):
        weight, bias = super()._merge(weight, bias)
        if weight is not None and self.flag_adapt_weight:
            scale2 = self.scale2.squeeze()
            if self.fan_in_fan_out:
                weight = weight * scale2.reshape(scale2.shape[-1:] + (1,) * (weight.dim() - 1))
            else:
                weight = weight * scale2.reshape((1, scale2.shape[-1]) + (1,) * (weight.dim() - 2))
        return weight, bias

    def _ssf_input(self, x: torch.Tensor):
        batch_size = x.size()[:-1]
        x = x.reshape(self.scale2.shape[0], -1, x.shape[-1])
        return (x * self.scale2).view(*batch_size, x.shape[-1])

    def _ssf(self, res: torch.Tensor):
        batch_size = res.size()[:-1]
        res = res.view(self.scale.shape[0], -1, res.shape[-1])
        if self.bias is not None:
            res = res * self.scale + (self.shift + self.bias)
        else:
            res = res * self.scale
        return res.view(*batch_size, res.shape[-1])