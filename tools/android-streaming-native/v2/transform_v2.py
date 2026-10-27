"""Explicit L8 V2 graph: verified V1 frequency fusion plus temporal pointwise fusion."""
import copy
import json
from pathlib import Path
import sys

import onnx
from onnx import helper

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from transform import SOURCE_SHA, DOMAIN, digest, fuse
from build_v2 import source_receipt

V1_SHA = "6b6cbd4117146f3137c1a0db71ecf04c23b8ba97bab7d973ece71c8739aa7756"
VARIANT = "native-frequency-temporal-v2"


def replace_temporal(model, states):
    graph = copy.deepcopy(model)
    producer = {value: node for node in graph.graph.node for value in node.output}
    consumers = {}
    for node in graph.graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node)

    def op(value, kind):
        node = producer[value]
        assert node.op_type == kind, (value, node.op_type, kind)
        return node

    def unique(value, kind):
        matches = [node for node in consumers[value] if node.op_type == kind]
        assert len(matches) == 1, (value, kind, len(matches))
        return matches[0]

    removed, inserted = set(), []
    for spec in states:
        if not spec["key"].endswith(".ssm_state"):
            continue
        old = spec["inputName"]
        saved = op(spec["outputName"], "Identity")
        added = op(saved.input[0], "Add")
        decayed, update = op(added.input[0], "Mul"), op(added.input[1], "Mul")
        decay = op(decayed.input[0], "Exp")
        assert decayed.input[1] == old
        dtb = op(update.input[0], "Mul")
        unsqueezed = op(update.input[1], "Unsqueeze")
        u, dt, b = unsqueezed.input[0], dtb.input[0], dtb.input[1]
        # Keep the existing ORT Exp(Mul(dt,A)) and prepared gate unchanged.
        assert op(decay.input[0], "Mul").input[0] == dt
        product = unique(added.output[0], "Mul")
        assert product.input[0] == added.output[0]
        c = product.input[1]
        reduced = unique(product.output[0], "ReduceSum")
        summed = unique(reduced.output[0], "Add")
        assert summed.input[0] == reduced.output[0]
        skipped = op(summed.input[1], "Mul")
        assert skipped.input[1] == u
        d = skipped.input[0]
        output = unique(summed.output[0], "Mul")
        assert output.input[0] == summed.output[0]
        gate = output.input[1]
        gate_node = op(gate, "Mul")
        assert op(gate_node.input[1], "Sigmoid").input[0] == gate_node.input[0]
        prefix = f"__reuse_native_temporal_v2_{len(inserted)}"
        inserted.append(helper.make_node("TemporalStepV2", [old, u, dt, b, c, gate, decay.output[0], d],
                                        [added.output[0], output.output[0]], domain=DOMAIN, name=prefix,
                                        abi_version=2, channels=256, state_size=16, frequency_bins=161))
        removed.update(node.output[0] for node in (added, decayed, update, dtb, product, reduced, summed, skipped, output))
    assert len(inserted) == 8 and len(removed) == 8 * 9
    pending = [node for node in graph.graph.node if node.output[0] not in removed] + inserted
    # Cache-output Identity nodes originally precede their prepared gate; restore
    # topological order without changing any remaining operator or its tensors.
    available = {value.name for value in graph.graph.input} | {value.name for value in graph.graph.initializer}
    ordered = []
    while pending:
        next_pending = []
        for node in pending:
            if all(not value or value in available for value in node.input):
                ordered.append(node)
                available.update(node.output)
            else:
                next_pending.append(node)
        assert len(next_pending) < len(pending), "Temporal fusion introduced an unresolved dependency"
        pending = next_pending
    del graph.graph.node[:]
    graph.graph.node.extend(ordered)
    return graph


def main():
    source = ROOT / "tools/streaming-export/generated/models/reuse-streaming-l8-fp32.onnx"
    assert digest(source) == SOURCE_SHA
    info = json.loads(source.with_suffix(".manifest.json").read_text())
    original = onnx.load(source)
    frequency, count = fuse(original)
    assert count == 16
    model = replace_temporal(frequency, info["states"])
    before = {item.name: item.SerializeToString() for item in original.graph.initializer}
    after = {item.name: item.SerializeToString() for item in model.graph.initializer}
    assert all(after[name] == value for name, value in before.items())
    for kind in ("input", "output"):
        assert [item.SerializeToString() for item in getattr(original.graph, kind)] == [
            item.SerializeToString() for item in getattr(model.graph, kind)]
    props = {item.key: item.value for item in original.metadata_props}
    props.update(execution_variant=VARIANT, baseline_model_sha256=SOURCE_SHA,
                 native_kernel_abi="2", native_frequency_scans="16", standard_frequency_scans="0",
                 native_temporal_steps="8", standard_temporal_steps="0")
    helper.set_model_props(model, props)
    onnx.checker.check_model(model, full_check=True)
    target = HERE / "generated/reuse-streaming-l8-native-frequency-temporal-v2.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, target)
    info.update(modelFile=target.name, modelBytes=target.stat().st_size, modelSha256=digest(target),
                executionVariant=VARIANT, baselineModelSha256=SOURCE_SHA, nativeFrequencyBaselineModelSha256=V1_SHA,
                customOpDomain=DOMAIN, customOpName="FrequencyScanV1", customTemporalOpName="TemporalStepV2",
                kernelAbi=2, nativeFrequencyScanCount=16, standardFrequencyScanCount=0,
                nativeTemporalStepCount=8, standardTemporalStepCount=0,
                preparedDecay=True, preparedGate=True, temporalPreparedDecay=True, temporalPreparedGate=True,
                recurrenceReduction="serial16", temporalReduction="serial16", nativeChannelParallel=True,
                nativeSources=source_receipt())
    target.with_suffix(".manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({"graph": str(target), "sha256": info["modelSha256"], "bytes": info["modelBytes"],
                      "frequencyScans": 16, "temporalSteps": 8, "originalInitializerBytesUnchanged": True}, indent=2))


if __name__ == "__main__":
    main()
