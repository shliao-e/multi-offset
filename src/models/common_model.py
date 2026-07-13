# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math

import torch
from torch import nn

from .entropy_models import BitEstimator, GaussianEncoder, EntropyCoder
from ..utils.stream_helper import get_padding_size
from torch.autograd import Function

class LowerBoundFunction(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""

    @staticmethod
    def forward(ctx, input_, bound):
        ctx.save_for_backward(input_, bound)
        return torch.max(input_, bound)

    @staticmethod
    def backward(ctx, grad_output):
        input_, bound = ctx.saved_tensors
        pass_through_if = (input_ >= bound) | (grad_output < 0)
        return pass_through_if.type(grad_output.dtype) * grad_output, None


class LowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.

    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """

    def __init__(self, bound):
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))

    @torch.jit.unused
    def lower_bound(self, x):
        return LowerBoundFunction.apply(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)
class QuantizeFunction(Function):
    @staticmethod
    def forward(ctx, inputs):
        noise = torch.zeros(1, dtype=inputs.dtype, device=inputs.device).uniform_(-0.5, 0.5)
        #another method different from directly add the noise
        outputs = torch.round(inputs + noise) - noise

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs

class CompressionModel(nn.Module):
    def __init__(self, y_distribution, z_channel, mv_z_channel=None,
                 ec_thread=False, stream_part=1):
        super().__init__()
        self.scale_bound = LowerBound(0.01)
        self.quant_bound = LowerBound(0.5)
        self.y_distribution = y_distribution
        self.z_channel = z_channel
        self.mv_z_channel = mv_z_channel
        self.entropy_coder = None
        self.bit_estimator_z = BitEstimator(z_channel)
        self.bit_estimator_offset = BitEstimator(z_channel)
        self.bit_estimator_z_mv = None
        if mv_z_channel is not None:
            self.bit_estimator_z_mv = BitEstimator(mv_z_channel)
        self.gaussian_encoder = GaussianEncoder(distribution=y_distribution)
        self.ec_thread = ec_thread
        self.stream_part = stream_part

        self.masks = {}

    def quant(self,x, offset=0):
        if self.training:
            y = QuantizeFunction.apply(x)
        else:
            y = torch.round(x - offset) + offset

        return y
    def quantize(self, x, mean=None, noise=False):
        if mean is not None:
            x = x - mean
        if not self.training:
            x = x.round()
            return x + mean if mean is not None else x
        if noise:
            noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.4, 0.4)
            noise = noise.clone().detach()
            x = x + noise
            # x = x + torch.empty_like(x).uniform_(-0.4, 0.4)
        else:
            n = torch.round(x) - x
            n = n.clone().detach()
            x = x + n
        return x + mean if mean is not None else x
    def get_curr_q(self, q_scale, q_basic, q_index=None):
        q_scale = q_scale[q_index]
        return q_basic * q_scale

    @staticmethod
    def probs_to_bits(probs):
        bits = -1.0 * torch.log(probs + 1e-5) / math.log(2.0)
        bits = torch.clamp_min(bits, 0)
        return bits

    def get_y_gaussian_bits(self, y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.normal.Normal(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        return CompressionModel.probs_to_bits(probs)

    def get_y_laplace_bits(self, y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        return CompressionModel.probs_to_bits(probs)

    def get_z_bits(self, z, bit_estimator):
        probs = bit_estimator.get_cdf(z + 0.5) - bit_estimator.get_cdf(z - 0.5)
        return CompressionModel.probs_to_bits(probs)

    def update(self, force=False):
        self.entropy_coder = EntropyCoder(self.ec_thread, self.stream_part)
        self.gaussian_encoder.update(force=force, entropy_coder=self.entropy_coder)
        self.bit_estimator_z.update(force=force, entropy_coder=self.entropy_coder)

    def pad_for_y(self, y):
        _, _, H, W = y.size()
        padding_l, padding_r, padding_t, padding_b = get_padding_size(H, W, 4)
        y_pad = torch.nn.functional.pad(
            y,
            (padding_l, padding_r, padding_t, padding_b),
            mode="replicate",
        )
        return y_pad, (-padding_l, -padding_r, -padding_t, -padding_b)

    @staticmethod
    def get_to_y_slice_shape(height, width):
        padding_l, padding_r, padding_t, padding_b = get_padding_size(height, width, 4)
        return (-padding_l, -padding_r, -padding_t, -padding_b)

    def slice_to_y(self, param, slice_shape):
        return torch.nn.functional.pad(param, slice_shape)

    @staticmethod
    def separate_prior(params):
        return params.chunk(3, 1)

    def forward_four_part_prior(self, y, common_params,write=False):
        
        quant_step, scales, means = self.separate_prior(common_params)
        
        _, _, H, W = y.size()
        quant_step = torch.clamp_min(quant_step, 0.5)
        y = y / quant_step
        y_res = y-means
        y_q = self.quant(y_res)
        y_hat = y_q + means
        y_hat = y_hat * quant_step
        return y_res, y_q, y_hat, scales
    