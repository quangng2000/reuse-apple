"""Bounded-memory CPU implementation of Mamba-1's selective scan.

Implements the real-valued, input-dependent B/C path used by RE-USE.
The official reference is checked independently in run_cpu.py.
"""

import torch
import torch.nn.functional as F


def selective_scan_cpu(
    u, delta, A, B, C, D=None, z=None, delta_bias=None,
    delta_softplus=False, return_last_state=False,
):
    if A.is_complex() or B.ndim != 3 or C.ndim != 3:
        raise ValueError("This CPU adapter supports only the real variable-B/C RE-USE path")
    original_dtype = u.dtype
    u, delta, B, C = u.float(), delta.float(), B.float(), C.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    state = A.new_zeros((u.shape[0], u.shape[1], A.shape[1]))
    output = torch.empty_like(u)
    # Bound intermediates independently of the full recording length.
    for start in range(0, u.shape[-1], 32):
        end = min(start + 32, u.shape[-1])
        dt = delta[:, :, start:end]
        transition = torch.exp(dt[..., None] * A[None, :, None, :])
        drive = dt[..., None] * B[:, None, :, start:end].transpose(-1, -2)
        drive = drive * u[:, :, start:end, None]
        for offset, index in enumerate(range(start, end)):
            state = transition[:, :, offset] * state + drive[:, :, offset]
            output[:, :, index] = (state * C[:, None, :, index]).sum(-1)
    if D is not None:
        output = output + u * D[None, :, None]
    if z is not None:
        output = output * F.silu(z)
    output = output.to(original_dtype)
    return (output, state) if return_last_state else output
