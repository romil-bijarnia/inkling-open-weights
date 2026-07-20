#!/usr/bin/env python3
"""Validate the generated Inkling checkpoint map against pinned invariants."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED_COMMIT = "86b4d430ab871652a707666b89203a866888c5e5"
EXPECTED_LAYERS = 66
EXPECTED_TENSORS = 1_552
EXPECTED_SAFETENSORS_FILES = 109
EXPECTED_PARAMETERS = 952_377_623_626
EXPECTED_TENSOR_BYTES = 1_904_755_280_148
EXPECTED_SPARSE_LAYERS = set(range(2, 66))
EXPECTED_MATRIX_RAW_VALUES = 6_144

LAYER_NAME_RE = re.compile(r"^model\.llm\.layers\.(\d+)\.")
DTYPE_BYTES = {"BF16": 2, "F32": 4}

SPARSE_TENSOR_SPECS: dict[str, tuple[str, list[int]]] = {
    "mlp.experts.w13_weight": ("BF16", [256, 6_144, 6_144]),
    "mlp.experts.w2_weight": ("BF16", [256, 6_144, 3_072]),
    "mlp.gate.bias": ("F32", [256]),
    "mlp.gate.global_scale": ("F32", [1]),
    "mlp.gate.weight": ("BF16", [258, 6_144]),
    "mlp.shared_experts.shared_w13_weight": ("BF16", [2, 6_144, 6_144]),
    "mlp.shared_experts.shared_w2_weight": ("BF16", [2, 6_144, 3_072]),
}

DENSE_TENSOR_SPECS: dict[str, tuple[str, list[int]]] = {
    "mlp.global_scale": ("BF16", [1]),
    "mlp.w13_dn.weight": ("BF16", [49_152, 6_144]),
    "mlp.w2_md.weight": ("BF16", [6_144, 24_576]),
}

EXPECTED_MATRIX_SAMPLES: dict[str, list[int]] = {
    "model.llm.embed.weight": [201_024, 6_144],
    "model.audio.encoder.weight": [1_280, 6_144],
    "model.visual.layers.linear_3.weight": [6_144, 9_600],
    "model.llm.layers.29.attn.wq_du.weight": [8_192, 6_144],
    "model.llm.layers.29.mlp.experts.w13_weight": [256, 6_144, 6_144],
    "model.llm.layers.29.mlp.shared_experts.shared_w13_weight": [2, 6_144, 6_144],
}


class Validation:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.checks = 0

    def check(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.errors.append(message)

    def equal(self, actual: Any, expected: Any, label: str) -> None:
        self.check(actual == expected, f"{label}: expected {expected!r}, got {actual!r}")


def product(dimensions: Iterable[int]) -> int:
    result = 1
    for dimension in dimensions:
        result *= dimension
    return result


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_finite_sample(
    validation: Validation,
    sample: Any,
    *,
    label: str,
    expected_tensor: str,
    expected_dtype: str,
    expected_shape: list[int],
    expected_values: int,
) -> None:
    validation.check(isinstance(sample, dict), f"{label}: sample must be an object")
    if not isinstance(sample, dict):
        return

    validation.equal(sample.get("tensor"), expected_tensor, f"{label}.tensor")
    validation.equal(sample.get("dtype"), expected_dtype, f"{label}.dtype")
    validation.equal(sample.get("shape"), expected_shape, f"{label}.shape")
    validation.equal(sample.get("sampledValues"), expected_values, f"{label}.sampledValues")

    values = sample.get("values")
    validation.check(isinstance(values, list), f"{label}.values must be an array")
    if not isinstance(values, list):
        return
    validation.equal(len(values), expected_values, f"{label}.values length")
    validation.check(all(finite_number(value) for value in values), f"{label}.values contain non-finite data")
    for statistic in ("minimum", "maximum", "mean"):
        validation.check(finite_number(sample.get(statistic)), f"{label}.{statistic} must be finite")


def validate_matrix_samples(
    validation: Validation,
    matrix_samples: Any,
    tensor_by_name: dict[str, dict[str, Any]],
) -> int:
    validation.check(isinstance(matrix_samples, dict), "matrixSamples must be an object")
    if not isinstance(matrix_samples, dict):
        return 0

    validation.equal(set(matrix_samples), set(EXPECTED_MATRIX_SAMPLES), "matrix sample tensor names")
    total_raw_values = 0

    for name, expected_shape in EXPECTED_MATRIX_SAMPLES.items():
        sample = matrix_samples.get(name)
        validation.check(isinstance(sample, dict), f"matrixSamples[{name!r}] must be an object")
        if not isinstance(sample, dict):
            continue

        tensor = tensor_by_name.get(name)
        validation.check(tensor is not None, f"matrix sample tensor {name!r} is missing from tensors")
        validation.equal(sample.get("tensor"), name, f"matrixSamples[{name!r}].tensor")
        validation.equal(sample.get("shape"), expected_shape, f"matrixSamples[{name!r}].shape")
        if tensor is not None:
            validation.equal(sample.get("shape"), tensor.get("shape"), f"matrixSamples[{name!r}] tensor shape")
            validation.equal(sample.get("dtype"), tensor.get("dtype"), f"matrixSamples[{name!r}] tensor dtype")

        shape = sample.get("shape")
        leading_indices = sample.get("leadingIndices")
        rows = sample.get("rowIndices")
        columns = sample.get("columnIndices")
        values = sample.get("values")
        validation.check(isinstance(shape, list) and len(shape) >= 2, f"matrixSamples[{name!r}].shape must have rank >= 2")
        validation.check(isinstance(leading_indices, list), f"matrixSamples[{name!r}].leadingIndices must be an array")
        validation.check(isinstance(rows, list), f"matrixSamples[{name!r}].rowIndices must be an array")
        validation.check(isinstance(columns, list), f"matrixSamples[{name!r}].columnIndices must be an array")
        validation.check(isinstance(values, list), f"matrixSamples[{name!r}].values must be an array")
        if not all(isinstance(item, list) for item in (shape, leading_indices, rows, columns, values)):
            continue

        validation.equal(len(leading_indices), len(shape) - 2, f"matrixSamples[{name!r}] leading rank")
        validation.equal(len(rows), 32, f"matrixSamples[{name!r}] sampled rows")
        validation.equal(len(columns), 32, f"matrixSamples[{name!r}] sampled columns")
        validation.equal(len(values), len(rows), f"matrixSamples[{name!r}] value rows")
        validation.check(rows == sorted(set(rows)), f"matrixSamples[{name!r}].rowIndices must be sorted and unique")
        validation.check(
            columns == sorted(set(columns)),
            f"matrixSamples[{name!r}].columnIndices must be sorted and unique",
        )
        if len(shape) >= 2:
            validation.check(
                all(isinstance(index, int) and 0 <= index < shape[-2] for index in rows),
                f"matrixSamples[{name!r}].rowIndices are out of bounds",
            )
            validation.check(
                all(isinstance(index, int) and 0 <= index < shape[-1] for index in columns),
                f"matrixSamples[{name!r}].columnIndices are out of bounds",
            )
            validation.check(
                all(
                    isinstance(index, int) and 0 <= index < dimension
                    for index, dimension in zip(leading_indices, shape[:-2])
                ),
                f"matrixSamples[{name!r}].leadingIndices are out of bounds",
            )

        row_widths = [len(row) for row in values if isinstance(row, list)]
        validation.equal(len(row_widths), len(values), f"matrixSamples[{name!r}] matrix row types")
        validation.check(
            all(width == len(columns) for width in row_widths),
            f"matrixSamples[{name!r}] matrix rows have the wrong width",
        )
        raw_values = [value for row in values if isinstance(row, list) for value in row]
        total_raw_values += len(raw_values)
        validation.equal(sample.get("sampledValues"), len(raw_values), f"matrixSamples[{name!r}] raw value count")
        validation.equal(len(raw_values), 1_024, f"matrixSamples[{name!r}] expected raw values")
        validation.check(
            all(finite_number(value) for value in raw_values),
            f"matrixSamples[{name!r}] contains non-finite values",
        )
        for statistic in ("minimum", "maximum", "mean"):
            validation.check(
                finite_number(sample.get(statistic)),
                f"matrixSamples[{name!r}].{statistic} must be finite",
            )

    validation.equal(total_raw_values, EXPECTED_MATRIX_RAW_VALUES, "matrix sample total raw value count")
    return total_raw_values


def validate_document(document: Any) -> tuple[Validation, int, int]:
    validation = Validation()
    validation.check(isinstance(document, dict), "document root must be an object")
    if not isinstance(document, dict):
        return validation, 0, 0

    source = document.get("source")
    architecture = document.get("architecture")
    checkpoint = document.get("checkpoint")
    layers = document.get("layers")
    tensors = document.get("tensors")
    validation.check(isinstance(source, dict), "source must be an object")
    validation.check(isinstance(architecture, dict), "architecture must be an object")
    validation.check(isinstance(checkpoint, dict), "checkpoint must be an object")
    validation.check(isinstance(layers, list), "layers must be an array")
    validation.check(isinstance(tensors, list), "tensors must be an array")
    if not all(
        (
            isinstance(source, dict),
            isinstance(architecture, dict),
            isinstance(checkpoint, dict),
            isinstance(layers, list),
            isinstance(tensors, list),
        )
    ):
        return validation, 0, 0

    validation.equal(source.get("commit"), EXPECTED_COMMIT, "source.commit")
    validation.equal(architecture.get("layers"), EXPECTED_LAYERS, "architecture.layers")
    validation.equal(architecture.get("checkpointParameters"), EXPECTED_PARAMETERS, "checkpoint parameters")
    validation.equal(architecture.get("checkpointBytes"), EXPECTED_TENSOR_BYTES, "checkpoint tensor bytes")
    validation.equal(checkpoint.get("tensorCount"), EXPECTED_TENSORS, "checkpoint.tensorCount")
    validation.equal(checkpoint.get("shardCount"), EXPECTED_SAFETENSORS_FILES, "checkpoint.shardCount")
    validation.equal(len(layers), EXPECTED_LAYERS, "layers length")
    validation.equal(len(tensors), EXPECTED_TENSORS, "tensors length")

    tensor_objects = [tensor for tensor in tensors if isinstance(tensor, dict)]
    validation.equal(len(tensor_objects), len(tensors), "tensor object count")
    names = [tensor.get("name") for tensor in tensor_objects]
    validation.check(all(isinstance(name, str) and name for name in names), "all tensor names must be non-empty strings")
    validation.equal(len(set(names)), len(names), "unique tensor name count")
    tensor_by_name = {
        tensor["name"]: tensor
        for tensor in tensor_objects
        if isinstance(tensor.get("name"), str) and tensor.get("name")
    }

    shards = checkpoint.get("shards")
    validation.check(isinstance(shards, list), "checkpoint.shards must be an array")
    shard_objects = [shard for shard in shards if isinstance(shard, dict)] if isinstance(shards, list) else []
    shard_names = [shard.get("name") for shard in shard_objects]
    validation.equal(len(shard_objects), EXPECTED_SAFETENSORS_FILES, "safetensors file count")
    validation.equal(len(set(shard_names)), EXPECTED_SAFETENSORS_FILES, "unique safetensors file names")
    validation.check(
        all(isinstance(name, str) and name.endswith(".safetensors") for name in shard_names),
        "all checkpoint shard names must end in .safetensors",
    )
    validation.equal(set(tensor.get("shard") for tensor in tensor_objects), set(shard_names), "referenced safetensors files")

    parameter_total = 0
    byte_total = 0
    dtype_parameters: Counter[str] = Counter()
    category_parameters: Counter[str] = Counter()
    category_tensors: Counter[str] = Counter()
    layer_tensors: dict[int, list[dict[str, Any]]] = defaultdict(list)
    shard_tensors: Counter[str] = Counter()
    shard_bytes: Counter[str] = Counter()

    for index, tensor in enumerate(tensor_objects):
        label = f"tensors[{index}]"
        shape = tensor.get("shape")
        dtype = tensor.get("dtype")
        parameters = tensor.get("parameters")
        tensor_bytes = tensor.get("bytes")
        validation.check(
            isinstance(shape, list) and shape and all(isinstance(dimension, int) and dimension > 0 for dimension in shape),
            f"{label}.shape must contain positive integer dimensions",
        )
        validation.check(dtype in DTYPE_BYTES, f"{label}.dtype is unsupported: {dtype!r}")
        validation.check(isinstance(parameters, int) and parameters > 0, f"{label}.parameters must be positive")
        validation.check(isinstance(tensor_bytes, int) and tensor_bytes > 0, f"{label}.bytes must be positive")
        if isinstance(shape, list) and all(isinstance(dimension, int) for dimension in shape):
            validation.equal(parameters, product(shape), f"{label}.parameters from shape")
        if dtype in DTYPE_BYTES and isinstance(parameters, int):
            validation.equal(tensor_bytes, parameters * DTYPE_BYTES[dtype], f"{label}.bytes from dtype")

        if isinstance(parameters, int):
            parameter_total += parameters
            if isinstance(dtype, str):
                dtype_parameters[dtype] += parameters
            category = tensor.get("category")
            if isinstance(category, str):
                category_parameters[category] += parameters
                category_tensors[category] += 1
        if isinstance(tensor_bytes, int):
            byte_total += tensor_bytes
        shard = tensor.get("shard")
        if isinstance(shard, str):
            shard_tensors[shard] += 1
            if isinstance(tensor_bytes, int):
                shard_bytes[shard] += tensor_bytes

        name = tensor.get("name")
        layer = tensor.get("layer")
        match = LAYER_NAME_RE.match(name) if isinstance(name, str) else None
        expected_layer = int(match.group(1)) if match else None
        validation.equal(layer, expected_layer, f"{label}.layer")
        if isinstance(layer, int):
            validation.check(0 <= layer < EXPECTED_LAYERS, f"{label}.layer is out of range")
            layer_tensors[layer].append(tensor)

    validation.equal(parameter_total, EXPECTED_PARAMETERS, "sum of tensor parameters")
    validation.equal(byte_total, EXPECTED_TENSOR_BYTES, "sum of tensor bytes")
    validation.equal(checkpoint.get("dtypeParameters"), dict(dtype_parameters), "checkpoint.dtypeParameters")
    validation.equal(checkpoint.get("categoryParameters"), dict(category_parameters), "checkpoint.categoryParameters")
    validation.equal(checkpoint.get("categoryTensors"), dict(category_tensors), "checkpoint.categoryTensors")

    shard_by_name = {
        shard["name"]: shard
        for shard in shard_objects
        if isinstance(shard.get("name"), str) and shard.get("name")
    }
    for name, shard in shard_by_name.items():
        validation.equal(shard.get("tensorCount"), shard_tensors[name], f"shard {name} tensorCount")
        validation.equal(shard.get("weightBytes"), shard_bytes[name], f"shard {name} weightBytes")
        header_bytes = shard.get("headerBytes")
        file_bytes = shard.get("fileBytes")
        if all(isinstance(value, int) for value in (header_bytes, file_bytes, shard.get("weightBytes"))):
            validation.equal(
                file_bytes,
                shard["weightBytes"] + header_bytes + 8,
                f"shard {name} safetensors file size",
            )
    validation.equal(
        sum(shard.get("tensorCount", 0) for shard in shard_objects),
        EXPECTED_TENSORS,
        "sum of shard tensor counts",
    )
    validation.equal(
        sum(shard.get("weightBytes", 0) for shard in shard_objects),
        EXPECTED_TENSOR_BYTES,
        "sum of shard tensor bytes",
    )

    layer_objects = [layer for layer in layers if isinstance(layer, dict)]
    layer_ids = [layer.get("id") for layer in layer_objects]
    validation.equal(layer_ids, list(range(EXPECTED_LAYERS)), "ordered layer IDs")
    for layer_id in range(EXPECTED_LAYERS):
        summary = layer_objects[layer_id] if layer_id < len(layer_objects) else {}
        entries = layer_tensors[layer_id]
        expected_count = 17 if layer_id < 2 else 21
        validation.equal(len(entries), expected_count, f"layer {layer_id} tensor inventory count")
        validation.equal(summary.get("tensorCount"), expected_count, f"layers[{layer_id}].tensorCount")
        validation.equal(
            summary.get("parameters"),
            sum(tensor["parameters"] for tensor in entries),
            f"layers[{layer_id}].parameters",
        )
        validation.equal(
            summary.get("bytes"),
            sum(tensor["bytes"] for tensor in entries),
            f"layers[{layer_id}].bytes",
        )
        expected_category_parameters: Counter[str] = Counter()
        expected_category_tensors: Counter[str] = Counter()
        for tensor in entries:
            category = tensor.get("category")
            if isinstance(category, str):
                expected_category_parameters[category] += tensor["parameters"]
                expected_category_tensors[category] += 1
        validation.equal(
            summary.get("categoryParameters"),
            dict(expected_category_parameters),
            f"layers[{layer_id}].categoryParameters",
        )
        validation.equal(
            summary.get("categoryTensors"),
            dict(expected_category_tensors),
            f"layers[{layer_id}].categoryTensors",
        )

        validate_finite_sample(
            validation,
            summary.get("queryNorm"),
            label=f"layers[{layer_id}].queryNorm",
            expected_tensor=f"model.llm.layers.{layer_id}.attn.q_norm.weight",
            expected_dtype="BF16",
            expected_shape=[128],
            expected_values=128,
        )
        scale_suffix = "mlp.global_scale" if layer_id < 2 else "mlp.gate.global_scale"
        scale_dtype = "BF16" if layer_id < 2 else "F32"
        validate_finite_sample(
            validation,
            summary.get("globalScale"),
            label=f"layers[{layer_id}].globalScale",
            expected_tensor=f"model.llm.layers.{layer_id}.{scale_suffix}",
            expected_dtype=scale_dtype,
            expected_shape=[1],
            expected_values=1,
        )
        if layer_id < 2:
            validation.equal(summary.get("gateBias"), None, f"layers[{layer_id}].gateBias")
        else:
            validate_finite_sample(
                validation,
                summary.get("gateBias"),
                label=f"layers[{layer_id}].gateBias",
                expected_tensor=f"model.llm.layers.{layer_id}.mlp.gate.bias",
                expected_dtype="F32",
                expected_shape=[256],
                expected_values=256,
            )

    sparse_layers = {
        tensor.get("layer")
        for tensor in tensor_objects
        if tensor.get("category") == "routed_experts" and isinstance(tensor.get("layer"), int)
    }
    validation.equal(sparse_layers, EXPECTED_SPARSE_LAYERS, "sparse layer IDs")
    validation.equal(len(sparse_layers), 64, "sparse layer count")

    for layer_id in range(EXPECTED_LAYERS):
        specs = DENSE_TENSOR_SPECS if layer_id < 2 else SPARSE_TENSOR_SPECS
        for suffix, (expected_dtype, expected_shape) in specs.items():
            name = f"model.llm.layers.{layer_id}.{suffix}"
            tensor = tensor_by_name.get(name)
            validation.check(tensor is not None, f"required tensor {name!r} is missing")
            if tensor is not None:
                validation.equal(tensor.get("dtype"), expected_dtype, f"{name}.dtype")
                validation.equal(tensor.get("shape"), expected_shape, f"{name}.shape")

    for suffix in SPARSE_TENSOR_SPECS:
        matching_layers = {
            tensor.get("layer")
            for tensor in tensor_objects
            if tensor.get("name", "").endswith(f".{suffix}")
        }
        validation.equal(matching_layers, EXPECTED_SPARSE_LAYERS, f"{suffix} layer coverage")

    matrix_raw_values = validate_matrix_samples(validation, document.get("matrixSamples"), tensor_by_name)
    return validation, len(sparse_layers), matrix_raw_values


def main() -> int:
    default_path = Path(__file__).resolve().parents[1] / "data" / "inkling-weight-map.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=default_path)
    args = parser.parse_args()

    try:
        document = json.loads(args.path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"FAIL: could not read {args.path}: {exc}", file=sys.stderr)
        return 1

    validation, sparse_layer_count, matrix_raw_values = validate_document(document)
    if validation.errors:
        print(f"FAIL: {args.path} ({len(validation.errors)} validation errors)", file=sys.stderr)
        for error in validation.errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"PASS: {args.path}")
    print(
        f"Validated {EXPECTED_TENSORS:,} tensors, {EXPECTED_LAYERS} layers "
        f"({sparse_layer_count} sparse), and {EXPECTED_SAFETENSORS_FILES} safetensors files."
    )
    print(
        f"Parameters: {EXPECTED_PARAMETERS:,}; tensor bytes: {EXPECTED_TENSOR_BYTES:,}; "
        f"matrix sample raw values: {matrix_raw_values:,}."
    )
    print(f"Checks passed: {validation.checks:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
