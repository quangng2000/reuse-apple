"""Unchanged pinned NVIDIA model + online wrapper, with CPU Mamba adapter.

The CUDA/download CLI is never executed. Upstream source bytes are SHA verified
before importing or extracting its class definitions. NVIDIA NSCLv1 applies.
"""
import ast
import hashlib
import json
import sys
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import SOURCE, GENERATED, selected_weights


class Oracle:
    def __init__(self, checkpoint, layers):
        root = GENERATED / "upstream"
        for item in SOURCE["reference_sources"]:
            p = root / item["upstream_path"]
            if hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Upstream source integrity failure:{p.name}")
        import cpu_mamba
        # Always the independently derived eager CPU path, even if a user has a
        # CUDA mamba_ssm installation in their environment.
        cpu_mamba.selective_state_update = None
        cpu_mamba.causal_conv1d_fn = None
        shim = types.ModuleType("mamba_ssm")
        shim.Mamba = cpu_mamba.Mamba
        sys.modules["mamba_ssm"] = shim
        hub = types.ModuleType("huggingface_hub")
        hub.PyTorchModelHubMixin = type("PyTorchModelHubMixin", (), {})
        sys.modules["huggingface_hub"] = hub
        sys.path.insert(0, str(root))
        from models.streaming_generator_SEMamba_time_d1_random_layer_ahead_sep_conv import SEMamba_decoder_list
        full, _ = selected_weights(checkpoint, layers)
        self.model = SEMamba_decoder_list(json.loads((root / "config.json").read_text())).eval()
        self.model.load_state_dict(full, strict=True)
        source = root / "online_inference.py"
        tree = ast.parse(source.read_text())
        selected = [n for n in tree.body if isinstance(n, ast.ClassDef) or
                    isinstance(n, ast.FunctionDef) and n.name == "get_causal_padding_2d"]
        namespace = {"torch": torch, "nn": nn, "F": F}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
        self.stream = namespace["StreamingSEMamba"](self.model).eval()
        self.layers = layers
        self.reset()

    def reset(self):
        self.stream.reset(161)

    @torch.inference_mode()
    def step(self, magnitude, phase):
        mag, pha = self.stream(torch.tensor(magnitude), torch.tensor(phase), self.layers, 0)
        return mag[..., None].numpy().copy(), pha[..., None].numpy().copy()

    def states(self):
        stream, depth = self.stream, self.layers
        values = {"dense_encoder.dense_conv_1_1.0": stream.encoder.conv1_1[0].cache}
        for i in range(4):
            values[f"dense_encoder.dense_block.dense_block.{i}.1"] = stream.encoder.dense_block.blocks[i][0].cache
        for i in range(depth):
            p = f"TSMamba.{i}.time_mamba.forward_blocks"
            # The published Mamba.step stores four filter samples, whereas the
            # minimal next-hop input needs only its latest three samples.
            values[p + ".conv1d"] = stream.mamba[i].time_mamba.conv_state[:, :, -3:]
            values[p + ".ssm_state"] = stream.mamba[i].time_mamba.ssm_state
        for name, decoder in [("mask_decoder_list", stream.mag_decoder_list[depth-1]),
                              ("phase_decoder_list", stream.pha_decoder_list[depth-1])]:
            for i in range(4):
                values[f"{name}.{depth-1}.dense_block.dense_block.{i}.1"] = decoder.dense_block.blocks[i][0].cache
            values[f"{name}.{depth-1}.up_conv2.0.conv"] = decoder.up_conv2[0].cache.transpose(2, 3)
        return {k: v.detach().numpy().copy() for k, v in values.items()}
