"""Separate native stable Softplus variant with unchanged inline V2 fallback."""
import json
from pathlib import Path
import sys

import onnx
from onnx import helper

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "inline"))
from transform_inline_v2 import SOURCE_SHA, V1_SHA, DOMAIN, digest
from build_stable_v2 import source_receipt


def main():
    source = HERE.parent / "inline/generated/reuse-streaming-l8-native-inline-v2.onnx"
    assert digest(source) == "55d910c30cd7144187e5f358dca6700ae6e7e1051eb8f7758d454f68e9144c11"
    model = onnx.load(source)
    producer = {value: node for node in model.graph.node for value in node.output}
    before = {item.name: item.SerializeToString() for item in model.graph.initializer}
    count = 0
    for node in model.graph.node:
        if node.op_type != "Softplus" or node.domain:
            continue
        added = producer[node.input[0]]
        assert added.op_type == "Add" and added.input[1].endswith(".dt_proj.bias")
        layout = int(".time_mamba." in added.input[1])
        assert layout or ".freq_mamba." in added.input[1]
        node.op_type = "StableSoftplusV2"
        node.domain = DOMAIN
        node.attribute.extend([helper.make_attribute("abi_version", 2), helper.make_attribute("layout", layout)])
        count += 1
    assert count == 24
    assert {item.name: item.SerializeToString() for item in model.graph.initializer} == before
    props = {item.key: item.value for item in model.metadata_props}
    props.update(execution_variant="native-stable-frequency-temporal-v2", native_softplus="24", standard_softplus="0",
                 softplus_implementation="ort-1.23.2-eigen-stable-packet-log1p")
    helper.set_model_props(model, props)
    onnx.checker.check_model(model, full_check=True)
    target = HERE / "generated/reuse-streaming-l8-native-stable-v2.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, target)
    info = json.loads(source.with_suffix(".manifest.json").read_text())
    info.update(modelFile=target.name, modelBytes=target.stat().st_size, modelSha256=digest(target),
                executionVariant="native-stable-frequency-temporal-v2", customSoftplusOpName="StableSoftplusV2",
                nativeSoftplusCount=24, standardSoftplusCount=0, nativeSources=source_receipt(),
                softplusImplementation="ort-1.23.2-eigen-stable-packet-log1p", nativeInlineBaselineModelSha256=digest(source))
    target.with_suffix(".manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({"graphSha256": info["modelSha256"], "bytes": info["modelBytes"], "nativeSoftplusCount": count}, indent=2))


if __name__ == "__main__":
    main()
