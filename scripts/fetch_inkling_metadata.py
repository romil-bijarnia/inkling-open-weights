#!/usr/bin/env python3
"""Build a compact, reproducible map of Inkling's published checkpoint.

The script downloads only the public config, tensor index, safetensors headers,
and a few small named tensors used as visual fingerprints. It never downloads
the 1.9 TB checkpoint payload.
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import json
import math
import re
import struct
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO = "thinkingmachines/Inkling"
HF_ROOT = "https://huggingface.co"
API_URL = f"{HF_ROOT}/api/models/{REPO}?blobs=true"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "inkling-weight-map.json"
USER_AGENT = "InklingOpenWeightMap/1.0"
MAX_WORKERS = 10
LAYER_RE = re.compile(r"model\.llm\.layers\.(\d+)\.")


def request_bytes(url: str, byte_range: tuple[int, int] | None = None, retries: int = 4) -> bytes:
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if byte_range:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"

    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            if byte_range:
                expected = byte_range[1] - byte_range[0] + 1
                if len(payload) != expected:
                    raise RuntimeError(
                        f"Range {byte_range} from {url} returned {len(payload)} bytes, expected {expected}"
                    )
            return payload
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            if attempt == retries - 1:
                raise
            time.sleep(0.7 * (2**attempt))
    raise AssertionError("unreachable")


def request_json(url: str) -> dict[str, Any]:
    return json.loads(request_bytes(url))


def tensor_category(name: str) -> str:
    if name.startswith("model.vision"):
        return "vision"
    if name.startswith("model.audio"):
        return "audio"
    if name.startswith("model.mtp"):
        return "next_token_prediction"
    if ".embed" in name:
        return "embedding"
    if ".lm_head" in name or name.endswith("output.weight"):
        return "output"
    if ".attn." in name or ".attn_" in name:
        return "attention"
    if ".mlp.experts." in name:
        return "routed_experts"
    if ".mlp.shared_experts." in name:
        return "shared_experts"
    if ".mlp.gate." in name:
        return "router"
    if ".mlp." in name:
        return "dense_mlp"
    if "norm" in name:
        return "normalization"
    return "other"


def layer_for(name: str) -> int | None:
    match = LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def numel(shape: list[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def fetch_header(commit: str, shard: str) -> tuple[str, int, dict[str, Any]]:
    url = f"{HF_ROOT}/{REPO}/resolve/{commit}/{shard}"
    prefix = request_bytes(url, (0, 7))
    header_length = struct.unpack("<Q", prefix)[0]
    header = json.loads(request_bytes(url, (8, 8 + header_length - 1)))
    return shard, header_length, header


def decode_values(dtype: str, payload: bytes) -> list[float]:
    if dtype == "F32":
        return list(struct.unpack(f"<{len(payload) // 4}f", payload))
    if dtype == "F16":
        return list(struct.unpack(f"<{len(payload) // 2}e", payload))
    if dtype == "BF16":
        return [
            struct.unpack("<f", b"\x00\x00" + payload[offset : offset + 2])[0]
            for offset in range(0, len(payload), 2)
        ]
    raise ValueError(f"Unsupported sample dtype: {dtype}")


def compact_float(value: float) -> float:
    if not math.isfinite(value):
        return value
    return float(f"{value:.8g}")


def sample_tensor(
    commit: str,
    tensor: dict[str, Any],
    header_lengths: dict[str, int],
    max_values: int | None = None,
) -> dict[str, Any]:
    dtype = tensor["dtype"]
    bytes_per_value = {"BF16": 2, "F16": 2, "F32": 4}[dtype]
    begin, end = tensor["dataOffsets"]
    available_values = (end - begin) // bytes_per_value
    count = min(available_values, max_values) if max_values else available_values
    absolute_start = 8 + header_lengths[tensor["shard"]] + begin
    absolute_end = absolute_start + count * bytes_per_value - 1
    url = f"{HF_ROOT}/{REPO}/resolve/{commit}/{tensor['shard']}"
    payload = request_bytes(url, (absolute_start, absolute_end))
    values = [compact_float(value) for value in decode_values(dtype, payload)]
    return {
        "tensor": tensor["name"],
        "dtype": dtype,
        "shape": tensor["shape"],
        "sampledValues": count,
        "values": values,
        "minimum": compact_float(min(values)),
        "maximum": compact_float(max(values)),
        "mean": compact_float(sum(values) / len(values)),
    }


def sample_matrix(
    commit: str,
    tensor: dict[str, Any],
    header_lengths: dict[str, int],
    sample_rows: int = 32,
    sample_columns: int = 32,
    leading_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Read an exact, sparse row sample from the final two dimensions of a tensor."""
    dtype = tensor["dtype"]
    bytes_per_value = {"BF16": 2, "F16": 2, "F32": 4}[dtype]
    shape = tensor["shape"]
    if len(shape) < 2:
        raise ValueError(f"Matrix sampling requires a rank-2+ tensor: {tensor['name']}")

    leading_shape = shape[:-2]
    leading_indices = leading_indices or [0] * len(leading_shape)
    if len(leading_indices) != len(leading_shape):
        raise ValueError(f"Leading-index rank mismatch for {tensor['name']}")

    leading_flat = 0
    for index, dimension in zip(leading_indices, leading_shape):
        if index < 0 or index >= dimension:
            raise ValueError(f"Leading index {index} is outside dimension {dimension}")
        leading_flat = leading_flat * dimension + index

    row_count, column_count = shape[-2], shape[-1]
    actual_rows = min(sample_rows, row_count)
    actual_columns = min(sample_columns, column_count)
    row_indices = sorted(
        {
            round(index * (row_count - 1) / max(1, actual_rows - 1))
            for index in range(actual_rows)
        }
    )
    column_start = max(0, (column_count - actual_columns) // 2)
    column_indices = list(range(column_start, column_start + actual_columns))
    tensor_begin = tensor["dataOffsets"][0]
    data_start = 8 + header_lengths[tensor["shard"]] + tensor_begin
    matrix_stride = row_count * column_count
    url = f"{HF_ROOT}/{REPO}/resolve/{commit}/{tensor['shard']}"

    def fetch_row(row_index: int) -> tuple[int, list[float]]:
        flat_index = leading_flat * matrix_stride + row_index * column_count + column_start
        absolute_start = data_start + flat_index * bytes_per_value
        absolute_end = absolute_start + actual_columns * bytes_per_value - 1
        values = decode_values(dtype, request_bytes(url, (absolute_start, absolute_end)))
        return row_index, [compact_float(value) for value in values]

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        rows = list(executor.map(fetch_row, row_indices))
    rows.sort(key=lambda item: item[0])
    values = [row for _, row in rows]
    flat_values = [value for row in values for value in row]
    return {
        "tensor": tensor["name"],
        "dtype": dtype,
        "shape": shape,
        "leadingIndices": leading_indices,
        "rowIndices": row_indices,
        "columnIndices": column_indices,
        "values": values,
        "sampledValues": len(flat_values),
        "minimum": compact_float(min(flat_values)),
        "maximum": compact_float(max(flat_values)),
        "mean": compact_float(sum(flat_values) / len(flat_values)),
        "method": "Exact stored values from evenly spaced rows and a centered contiguous column window.",
    }


def main() -> None:
    model_api = request_json(API_URL)
    commit = model_api["sha"]
    config_url = f"{HF_ROOT}/{REPO}/resolve/{commit}/config.json"
    index_url = f"{HF_ROOT}/{REPO}/resolve/{commit}/model.safetensors.index.json"
    config = request_json(config_url)
    index = request_json(index_url)
    shard_names = sorted(set(index["weight_map"].values()))

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        headers = list(executor.map(lambda shard: fetch_header(commit, shard), shard_names))

    header_lengths: dict[str, int] = {}
    header_maps: dict[str, dict[str, Any]] = {}
    for shard, header_length, header in headers:
        header_lengths[shard] = header_length
        header_maps[shard] = header

    tensors: list[dict[str, Any]] = []
    for name, shard in sorted(index["weight_map"].items()):
        raw = header_maps[shard].get(name)
        if raw is None:
            raise RuntimeError(f"Tensor {name} is missing from the header for {shard}")
        shape = raw["shape"]
        begin, end = raw["data_offsets"]
        tensor = {
            "name": name,
            "shard": shard,
            "dtype": raw["dtype"],
            "shape": shape,
            "parameters": numel(shape),
            "bytes": end - begin,
            "dataOffsets": [begin, end],
            "layer": layer_for(name),
            "category": tensor_category(name),
        }
        tensors.append(tensor)

    tensor_by_name = {tensor["name"]: tensor for tensor in tensors}
    sample_jobs: list[tuple[str, dict[str, Any], int | None]] = []
    for layer in range(config["text_config"]["num_hidden_layers"]):
        gate_name = f"model.llm.layers.{layer}.mlp.gate.bias"
        qnorm_name = f"model.llm.layers.{layer}.attn.q_norm.weight"
        scale_name = (
            f"model.llm.layers.{layer}.mlp.global_scale"
            if layer < config["text_config"]["dense_mlp_idx"]
            else f"model.llm.layers.{layer}.mlp.gate.global_scale"
        )
        if gate_name in tensor_by_name:
            sample_jobs.append((f"gateBias:{layer}", tensor_by_name[gate_name], None))
        if qnorm_name in tensor_by_name:
            sample_jobs.append((f"queryNorm:{layer}", tensor_by_name[qnorm_name], None))
        if scale_name in tensor_by_name:
            sample_jobs.append((f"globalScale:{layer}", tensor_by_name[scale_name], None))

    def run_sample(job: tuple[str, dict[str, Any], int | None]) -> tuple[str, dict[str, Any]]:
        key, tensor, limit = job
        return key, sample_tensor(commit, tensor, header_lengths, limit)

    samples: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for key, sample in executor.map(run_sample, sample_jobs):
            samples[key] = sample

    matrix_sample_names = [
        "model.llm.embed.weight",
        "model.audio.encoder.weight",
        "model.visual.layers.linear_3.weight",
        "model.llm.layers.29.attn.wq_du.weight",
        "model.llm.layers.29.mlp.experts.w13_weight",
        "model.llm.layers.29.mlp.shared_experts.shared_w13_weight",
    ]
    matrix_samples: dict[str, dict[str, Any]] = {}
    for name in matrix_sample_names:
        tensor = tensor_by_name.get(name)
        if tensor is None:
            continue
        leading = [0] * max(0, len(tensor["shape"]) - 2)
        matrix_samples[name] = sample_matrix(commit, tensor, header_lengths, leading_indices=leading)

    layer_tensors: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for tensor in tensors:
        if tensor["layer"] is not None:
            layer_tensors[tensor["layer"]].append(tensor)

    local_layers = set(config["text_config"]["local_layer_ids"])
    layers: list[dict[str, Any]] = []
    for layer in range(config["text_config"]["num_hidden_layers"]):
        entries = layer_tensors[layer]
        category_parameters: Counter[str] = Counter()
        category_tensors: Counter[str] = Counter()
        for tensor in entries:
            category_parameters[tensor["category"]] += tensor["parameters"]
            category_tensors[tensor["category"]] += 1
        layers.append(
            {
                "id": layer,
                "attention": "local" if layer in local_layers else "global",
                "tensorCount": len(entries),
                "parameters": sum(tensor["parameters"] for tensor in entries),
                "bytes": sum(tensor["bytes"] for tensor in entries),
                "categoryParameters": dict(category_parameters),
                "categoryTensors": dict(category_tensors),
                "gateBias": samples.get(f"gateBias:{layer}"),
                "queryNorm": samples.get(f"queryNorm:{layer}"),
                "globalScale": samples.get(f"globalScale:{layer}"),
            }
        )

    dtype_parameters: Counter[str] = Counter()
    category_parameters: Counter[str] = Counter()
    category_tensors: Counter[str] = Counter()
    shard_tensors: Counter[str] = Counter()
    shard_bytes: Counter[str] = Counter()
    for tensor in tensors:
        dtype_parameters[tensor["dtype"]] += tensor["parameters"]
        category_parameters[tensor["category"]] += tensor["parameters"]
        category_tensors[tensor["category"]] += 1
        shard_tensors[tensor["shard"]] += 1
        shard_bytes[tensor["shard"]] += tensor["bytes"]

    sibling_sizes = {
        sibling["rfilename"]: sibling.get("size")
        for sibling in model_api.get("siblings", [])
        if sibling.get("size") is not None
    }
    checkpoint_parameters = sum(tensor["parameters"] for tensor in tensors)
    checkpoint_bytes = sum(tensor["bytes"] for tensor in tensors)
    physical_weight_file_bytes = sum(sibling_sizes.get(shard, 0) for shard in shard_names)
    output = {
        "source": {
            "repository": REPO,
            "commit": commit,
            "generatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
            "configUrl": config_url,
            "indexUrl": index_url,
            "license": model_api.get("cardData", {}).get("license", "apache-2.0"),
            "method": (
                "Config, index, and all 109 referenced safetensors headers were read from the pinned public checkpoint. "
                "Router biases, query-normalization vectors, and global-scale values were fetched with exact HTTP byte ranges."
            ),
        },
        "architecture": {
            "officialTotalParameters": 975_000_000_000,
            "officialActiveParameters": 41_000_000_000,
            "checkpointParameters": checkpoint_parameters,
            "checkpointBytes": checkpoint_bytes,
            "contextTokens": config["text_config"]["model_max_length"],
            "hiddenSize": config["text_config"]["hidden_size"],
            "layers": config["text_config"]["num_hidden_layers"],
            "attentionHeads": config["text_config"]["num_attention_heads"],
            "keyValueHeads": config["text_config"]["num_key_value_heads"],
            "routedExperts": config["text_config"]["n_routed_experts"],
            "expertsPerToken": config["text_config"]["num_experts_per_tok"],
            "sharedExperts": config["text_config"]["n_shared_experts"],
            "slidingWindow": config["text_config"]["sliding_window_size"],
            "visionPatchSize": config["vision_config"]["patch_size"],
            "audioMelBins": config["audio_config"]["n_mel_bins"],
        },
        "checkpoint": {
            "tensorCount": len(tensors),
            "shardCount": len(shard_names),
            "indexReportedBytes": index["metadata"]["total_size"],
            "indexToHeaderByteDifference": checkpoint_bytes - index["metadata"]["total_size"],
            "physicalWeightFileBytes": physical_weight_file_bytes,
            "headerBytes": physical_weight_file_bytes - checkpoint_bytes,
            "dtypeParameters": dict(dtype_parameters),
            "categoryParameters": dict(category_parameters),
            "categoryTensors": dict(category_tensors),
            "shards": [
                {
                    "name": shard,
                    "headerBytes": header_lengths[shard],
                    "fileBytes": sibling_sizes.get(shard),
                    "tensorCount": shard_tensors[shard],
                    "weightBytes": shard_bytes[shard],
                }
                for shard in shard_names
            ],
        },
        "layers": layers,
        "matrixSamples": matrix_samples,
        "tensors": tensors,
    }

    api_parameter_total = model_api.get("safetensors", {}).get("total")
    if api_parameter_total is not None and checkpoint_parameters != api_parameter_total:
        raise RuntimeError("Header-derived parameter total does not match Hugging Face's safetensors scan")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Commit: {commit}")
    print(f"Tensors: {len(tensors):,}")
    print(f"Parameters: {output['architecture']['checkpointParameters']:,}")
    print(f"Checkpoint bytes: {output['architecture']['checkpointBytes']:,}")
    print(f"Referenced safetensors files: {len(shard_names):,}")
    print(f"Vector samples: {len(sample_jobs):,}")
    print(f"Matrix samples: {len(matrix_samples):,}")


if __name__ == "__main__":
    main()
