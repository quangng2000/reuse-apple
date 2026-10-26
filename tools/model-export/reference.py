"""CPU-only FP32 oracle for the pinned commercial 30-layer RE-USE checkpoint.

Architecture adapted from NVIDIA-NemotronLabs-RE-USE revision
31f82f2efad1095034492f236e8dac9239ba713b, Copyright (c) 2026 NVIDIA
CORPORATION & AFFILIATES. SPDX-License-Identifier: OpenMDW-1.1.
Mamba recurrence follows the Apache-2.0 reference by Tri Dao / Albert Gu.
No CUDA extension, model download, reduced precision, or learned-weight changes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

CHECKPOINT_SHA256 = "06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1"
SOURCE_REVISION = "31f82f2efad1095034492f236e8dac9239ba713b"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "models/model.safetensors"


def verified_weights(path: Path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("Unsupported checkpoint: SHA-256 must match the commercial 30-layer model")
    weights = load_file(str(path), device="cpu")
    if sum(t.numel() for t in weights.values()) != 9_607_747:
        raise ValueError("Unexpected parameter count")
    if any(t.dtype != torch.float32 for t in weights.values()):
        raise ValueError("Every original weight must be FP32")
    if weights["TSMamba.29.freq_mamba.backward_blocks.A_log"].shape != (256, 16):
        raise ValueError("Incomplete 30-layer checkpoint")
    return weights


class Reference:
    def __init__(self, weights):
        self.w = weights

    def linear(self, x, p):
        return F.linear(x, self.w[p + ".weight"], self.w.get(p + ".bias"))

    def conv(self, x, p, *, padding=0, stride=1, dilation=1):
        return F.conv2d(x, self.w[p + ".weight"], self.w[p + ".bias"],
                        stride=stride, padding=padding, dilation=dilation)

    def norm_prelu(self, x, p):
        return F.prelu(F.instance_norm(x, weight=self.w[p + ".1.weight"],
                                      bias=self.w[p + ".1.bias"], eps=1e-5),
                       self.w[p + ".2.weight"])

    def dense(self, x, p):
        skip = x
        for i in range(4):
            q = f"{p}.dense_block.{i}"
            x = self.conv(skip, q + ".0", padding=(1 << i, 1), dilation=(1 << i, 1))
            x = self.norm_prelu(x, q)
            skip = torch.cat((x, skip), dim=1)
        return x

    def up(self, x, p, ratio):
        x = self.conv(F.pad(x, (1, 1, 0, 0)), p + ".0.conv")
        b, cr, h, w = x.shape
        x = x.reshape(b, ratio, cr // ratio, h, w).permute(0, 2, 3, 4, 1)
        return self.norm_prelu(x.reshape(b, cr // ratio, h, w * ratio), p)

    def decoder(self, x, p):
        x = self.dense(x, p + ".dense_block")
        x = self.up(x, p + ".up_conv1", 2)
        return self.up(x.transpose(2, 3), p + ".up_conv2", 4).transpose(2, 3)

    def scan(self, u, delta, b, c, z, p):
        a = -torch.exp(self.w[p + ".A_log"])
        dt = F.softplus(delta + self.w[p + ".dt_proj.bias"])
        state = torch.zeros(u.shape[0], 256, 16, dtype=torch.float32)
        outputs = []
        # Sequential time updates, never parallel prefix or truncated context.
        for t in range(u.shape[1]):
            d = dt[:, t, :, None]
            state = torch.exp(d * a) * state + (d * b[:, t, None, :]) * u[:, t, :, None]
            y = (state * c[:, t, None, :]).sum(dim=-1)
            outputs.append((y + self.w[p + ".D"] * u[:, t]) * F.silu(z[:, t]))
        return torch.stack(outputs, dim=1)

    def mamba(self, x, p):
        xz = self.linear(x, p + ".in_proj")
        u, z = xz.split(256, dim=-1)
        u = F.conv1d(u.transpose(1, 2), self.w[p + ".conv1d.weight"],
                     self.w[p + ".conv1d.bias"], padding=3, groups=256)[:, :, :x.shape[1]]
        u = F.silu(u.transpose(1, 2))
        proj = self.linear(u, p + ".x_proj")
        dt, b, c = torch.split(proj, (4, 16, 16), dim=-1)
        dt = F.linear(dt, self.w[p + ".dt_proj.weight"])
        return self.linear(self.scan(u, dt, b, c, z, p), p + ".out_proj")

    def bidirectional(self, x, p):
        fw = self.mamba(x, p + ".forward_blocks") + x
        rev = x.flip((1,))
        bw = (self.mamba(rev, p + ".backward_blocks") + rev).flip((1,))
        y = self.linear(torch.cat((fw, bw), dim=-1), p + ".output_proj")
        return F.layer_norm(y, (64,), self.w[p + ".norm.weight"], self.w[p + ".norm.bias"], 1e-5)

    @torch.inference_mode()
    def __call__(self, magnitude, phase, inspect=None):
        frames = magnitude.shape[2]
        x = torch.stack((magnitude.transpose(1, 2), phase.transpose(1, 2)), dim=1)
        x = F.pad(x, (0, 2, 0, 2))
        x = self.norm_prelu(self.conv(x, "dense_encoder.dense_conv_1.0"), "dense_encoder.dense_conv_1")
        x = self.dense(x, "dense_encoder.dense_block")
        x = self.norm_prelu(self.conv(x, "dense_encoder.dense_conv_2.0", stride=(4, 2)), "dense_encoder.dense_conv_2")
        if inspect: inspect("encoder", x)
        for i in range(30):
            b, ch, t, f = x.shape
            tx = x.permute(0, 3, 2, 1).reshape(b * f, t, ch)
            tx = self.bidirectional(tx, f"TSMamba.{i}.time_mamba") + tx
            fx = tx.reshape(b, f, t, ch).permute(0, 2, 1, 3).reshape(b * t, f, ch)
            fx = self.bidirectional(fx, f"TSMamba.{i}.freq_mamba") + fx
            x = fx.reshape(b, t, f, ch).permute(0, 3, 1, 2)
            if inspect and i in (0, 29): inspect(f"block{i+1}", x)
        mag = self.conv(self.decoder(x, "mask_decoder"), "mask_decoder.final_conv")[:, 0, :frames, :321].transpose(1, 2)
        features = self.decoder(x, "phase_decoder")
        real = self.conv(features, "phase_decoder.phase_conv_r")[:, 0, :frames, :321].transpose(1, 2)
        imag = self.conv(features, "phase_decoder.phase_conv_i")[:, 0, :frames, :321].transpose(1, 2)
        return mag, torch.atan2(imag, real)
