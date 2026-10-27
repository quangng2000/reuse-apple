#!/usr/bin/env python3
"""Experimental L8-only FP32 frequency Scan fusion; NVIDIA NSCLv1.

Every original learned initializer and external cache remains unchanged. The
native kernel replaces only bounded frequency recurrence dispatch. Baseline
artifacts and manifests are never overwritten or silently substituted.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[2]
DOMAIN = "ai.reuse.streaming"
SOURCE_SHA = "18db8d32b638702909de484d2a9cffc97ed09e6454c4caed3f0008fb0aab53e8"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fuse(graph, parallel=True):
    transformed = copy.deepcopy(graph)
    nodes, count = [], 0
    initializers = {item.name: item for item in graph.graph.initializer}
    for node in graph.graph.node:
        if node.op_type != "Scan":
            nodes.append(copy.deepcopy(node))
            continue
        attributes = {item.name: helper.get_attribute_value(item) for item in node.attribute}
        body = attributes["body"]
        assert len(node.input) == 6 and len(node.output) == 2
        assert attributes["num_scan_inputs"] == 5
        assert attributes["scan_input_axes"] == [1] * 5 and attributes["scan_output_axes"] == [1]
        assert [item.name for item in body.input] == ["state_in", "u_t", "delta_t", "b_t", "c_t", "z_t"]
        assert [item.op_type for item in body.node] == [
            "Unsqueeze", "Unsqueeze", "Unsqueeze", "Unsqueeze", "Mul", "Exp", "Mul", "Mul", "Mul",
            "Add", "Mul", "ReduceSum", "Mul", "Sigmoid", "Mul", "Add", "Mul"]
        a = body.node[4].input[1]
        d = body.node[12].input[0]
        assert d in initializers and list(initializers[d].dims) == [256]
        prefix = f"__reuse_native_frequency_v1_{count}"
        axes = prefix + "_axes3"
        transformed.graph.initializer.append(numpy_helper.from_array(np.asarray([3], np.int64), axes))
        # The original body computes these same FP32 operations. Bulk Exp and
        # gate preserve ORT's vector math implementation and remove per-bin
        # interpreter dispatch. Native recurrence keeps its dependence order.
        delta_expanded, product, decay = (prefix + suffix for suffix in ("_delta4d", "_product", "_decay"))
        sigmoid, gate = prefix + "_sigmoid", prefix + "_gate"
        nodes += [helper.make_node("Unsqueeze", [node.input[2], axes], [delta_expanded]),
                  helper.make_node("Mul", [delta_expanded, a], [product]),
                  helper.make_node("Exp", [product], [decay]),
                  helper.make_node("Sigmoid", [node.input[5]], [sigmoid]),
                  helper.make_node("Mul", [node.input[5], sigmoid], [gate]),
                  helper.make_node("FrequencyScanV1", [*node.input[:5], gate, decay, d], list(node.output),
                                   domain=DOMAIN, name=node.name + ".native_v1", abi_version=1,
                                   channels=256, state_size=16, frequency_bins=161, parallel=int(parallel))]
        count += 1
    del transformed.graph.node[:]
    transformed.graph.node.extend(nodes)
    if count:
        assert not any(item.domain == DOMAIN for item in transformed.opset_import)
        transformed.opset_import.append(helper.make_opsetid(DOMAIN, 1))
    return transformed, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "tools/streaming-export/generated/models/reuse-streaming-l8-fp32.onnx")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "generated/reuse-streaming-l8-native-frequency-v1.onnx")
    parser.add_argument("--serial", action="store_true")
    args = parser.parse_args()
    assert args.source.resolve() != args.output.resolve()
    assert digest(args.source) == SOURCE_SHA, "Only the exact released L8 graph is accepted"
    model = onnx.load(args.source)
    assert sum(node.op_type == "Scan" for node in model.graph.node) == 16
    transformed, count = fuse(model, not args.serial)
    assert count == 16
    original = {item.name: item.SerializeToString() for item in model.graph.initializer}
    actual = {item.name: item.SerializeToString() for item in transformed.graph.initializer}
    assert all(actual[key] == value for key, value in original.items()), "Learned weights must be byte-identical"
    for kind in ("input", "output"):
        assert [item.SerializeToString() for item in getattr(model.graph, kind)] == [
            item.SerializeToString() for item in getattr(transformed.graph, kind)], "External schemas must be unchanged"
    props = dict((item.key, item.value) for item in model.metadata_props)
    props.update(execution_variant="native-frequency-scan-v1", baseline_model_sha256=SOURCE_SHA,
                 native_kernel_abi="1", native_frequency_scans="16", standard_frequency_scans="0")
    helper.set_model_props(transformed, props)
    onnx.checker.check_model(transformed, full_check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(transformed, args.output)
    manifest = json.loads(args.source.with_suffix(".manifest.json").read_text())
    manifest.update(modelFile=args.output.name, modelBytes=args.output.stat().st_size, modelSha256=digest(args.output),
                    executionVariant="native-frequency-scan-v1", baselineModelSha256=SOURCE_SHA,
                    customOpDomain=DOMAIN, customOpName="FrequencyScanV1", kernelAbi=1,
                    nativeFrequencyScanCount=16, standardFrequencyScanCount=0,
                    preparedDecay=True, preparedGate=True, recurrenceReduction="serial16",
                    nativeChannelParallel=not args.serial)
    here = Path(__file__).resolve().parent
    manifest["nativeSources"] = {name: digest(here / name) for name in (
        "frequency_scan.cpp", "CMakeLists.txt", "build_native.py", "transform.py",
        "vendor/onnxruntime-1.23.2/provenance.json")}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"sourceSha256": SOURCE_SHA, "modelFile": args.output.name, "modelBytes": manifest["modelBytes"],
                      "modelSha256": manifest["modelSha256"], "fusedScans": count, "initializersUnchanged": True}, indent=2))


if __name__ == "__main__":
    main()
