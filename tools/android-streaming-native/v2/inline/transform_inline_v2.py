"""V2 inline-MLAS variant; preserves the separate V1 and bulk V2 artifacts."""
import copy
import json
from pathlib import Path
import sys

import onnx
from onnx import helper

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from transform_v2 import SOURCE_SHA, V1_SHA, DOMAIN, digest
from build_inline_v2 import source_receipt


def main():
    source = HERE.parent / "generated/reuse-streaming-l8-native-frequency-temporal-v2.onnx"
    assert digest(source) == "1fbd91be9c90568c58ab40e9ce70e07c54dafc599fffb694a84c1f4d3905b11d"
    model = onnx.load(source)
    before = {item.name: item.SerializeToString() for item in model.graph.initializer}
    producers = {value: node for node in model.graph.node for value in node.output}
    removed, counts = set(), {"FrequencyScanV1": 0, "TemporalStepV2": 0}
    for node in model.graph.node:
        if node.domain != DOMAIN or node.op_type not in counts:
            continue
        counts[node.op_type] += 1
        decay = producers[node.input[6]]
        assert decay.op_type == "Exp"
        product = producers[decay.input[0]]
        assert product.op_type == "Mul"
        a = product.input[1]
        removed.update((decay.output[0], product.output[0]))
        if node.op_type == "FrequencyScanV1":
            expanded = producers[product.input[0]]
            assert expanded.op_type == "Unsqueeze" and expanded.input[0] == node.input[2]
            removed.add(expanded.output[0])
            node.op_type = "FrequencyScanV2"
            next(item for item in node.attribute if item.name == "abi_version").i = 2
        else:
            assert product.input[0] == node.input[2]
        node.input[6] = a
    assert counts == {"FrequencyScanV1": 16, "TemporalStepV2": 8}
    kept = [copy.deepcopy(node) for node in model.graph.node if node.output[0] not in removed]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    assert {item.name: item.SerializeToString() for item in model.graph.initializer} == before
    props = {item.key: item.value for item in model.metadata_props}
    props.update(execution_variant="native-inline-frequency-temporal-v2", inline_exp="ort-1.23.2-mlas-neon-fp32")
    helper.set_model_props(model, props)
    onnx.checker.check_model(model, full_check=True)
    target = HERE / "generated/reuse-streaming-l8-native-inline-v2.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, target)
    info = json.loads(source.with_suffix(".manifest.json").read_text())
    info.update(modelFile=target.name, modelBytes=target.stat().st_size, modelSha256=digest(target),
                executionVariant="native-inline-frequency-temporal-v2", customOpName="FrequencyScanV2",
                preparedDecay=False, temporalPreparedDecay=False, nativeSources=source_receipt(),
                inlineExp="ort-1.23.2-mlas-neon-fp32", nativeTargets=["arm64-v8a", "host-arm64"],
                nativeTemporalBaselineModelSha256=digest(source))
    target.with_suffix(".manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({"graphSha256": info["modelSha256"], "bytes": info["modelBytes"], "removedNodes": len(removed)}, indent=2))


if __name__ == "__main__":
    main()
