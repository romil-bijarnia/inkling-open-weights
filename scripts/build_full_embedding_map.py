#!/usr/bin/env python3
"""Build the complete 200,058-token Inkling embedding map.

Every tokenizer entry is represented. The builder expands the published BF16
rows exactly to FP32, computes a full-vocabulary PCA layout, and exhaustively
searches the complete vocabulary for each token's top cosine neighbours. Large
outputs are binary and range-addressable so the browser does not load the
2.458 GB embedding tensor at startup.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from build_embedding_map import (
    ABSOLUTE_START,
    COMMIT,
    DIMENSIONS,
    REPO,
    ROW_BYTES,
    SHARD,
    SHARD_URL,
    TOKENIZER_ENTRIES,
    TOKENIZER_URL,
    VOCAB_ROWS,
    load_vocabulary,
    spherical_kmeans,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_PATH = DATA_DIR / "inkling-embedding-vectors-full.bin"
VECTOR_MANIFEST_PATH = DATA_DIR / "inkling-embedding-vectors-full.manifest.json"
VOCAB_PATH = DATA_DIR / "inkling-embedding-vocab.json"
LAYOUT_PATH = DATA_DIR / "inkling-embedding-layout.bin"
CLUSTERS_PATH = DATA_DIR / "inkling-embedding-clusters.bin"
NEIGHBOUR_IDS_PATH = DATA_DIR / "inkling-embedding-neighbor-ids.bin"
NEIGHBOUR_COSINES_PATH = DATA_DIR / "inkling-embedding-neighbor-cosines.bin"
MANIFEST_PATH = DATA_DIR / "inkling-embedding-full-manifest.json"
PCA_STAGE_PATH = DATA_DIR / ".inkling-embedding-pca-stage.json"
NEIGHBOUR_STAGE_PATH = DATA_DIR / ".inkling-embedding-neighbor-stage.json"

TOKEN_COUNT = TOKENIZER_ENTRIES
K_NEIGHBOURS = 32
PCA_COMPONENTS = 12
GEOMETRY_GROUPS = 16
LOAD_ROWS = 1_024
QUERY_ROWS = 1_024
SEED = 42


def compact(value: float, digits: int = 9) -> float:
    if not math.isfinite(value):
        return value
    return float(f"{value:.{digits}g}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any, *, compact_json: bool = False) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if compact_json:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def choose_device(requested: str) -> torch.device:
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_normalized_matrix(device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    expected_bytes = TOKEN_COUNT * ROW_BYTES
    if not VECTOR_PATH.exists() or VECTOR_PATH.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"Missing complete embedding payload at {VECTOR_PATH}; "
            "run scripts/download_full_embeddings.py first"
        )
    source = torch.from_file(
        str(VECTOR_PATH),
        shared=False,
        size=TOKEN_COUNT * DIMENSIONS,
        dtype=torch.bfloat16,
    ).reshape(TOKEN_COUNT, DIMENSIONS)
    matrix = torch.empty((TOKEN_COUNT, DIMENSIONS), dtype=torch.float32, device=device)
    norms = np.empty(TOKEN_COUNT, dtype=np.float32)
    started = time.monotonic()
    print(f"Expanding and normalizing {TOKEN_COUNT:,} exact BF16 rows on {device.type}…", flush=True)
    with torch.inference_mode():
        for start in range(0, TOKEN_COUNT, LOAD_ROWS):
            end = min(TOKEN_COUNT, start + LOAD_ROWS)
            cpu_chunk = source[start:end].float()
            chunk = cpu_chunk.to(device)
            chunk_norms = torch.linalg.vector_norm(chunk, dim=1)
            if not bool(torch.isfinite(chunk_norms).all().item()) or bool((chunk_norms <= 0).any().item()):
                raise RuntimeError(f"Invalid norm in token rows {start}:{end}")
            matrix[start:end].copy_(chunk / chunk_norms[:, None])
            norms[start:end] = chunk_norms.cpu().numpy()
            if end % 16_384 == 0 or end == TOKEN_COUNT:
                elapsed = max(time.monotonic() - started, 0.001)
                print(f"  normalized {end:>7,}/{TOKEN_COUNT:,} rows · {end / elapsed:,.0f} rows/s", flush=True)
    return matrix, norms


def deterministic_orientation(vectors: np.ndarray) -> np.ndarray:
    oriented = vectors.copy()
    for column in range(oriented.shape[1]):
        pivot = int(np.argmax(np.abs(oriented[:, column])))
        if oriented[pivot, column] < 0:
            oriented[:, column] *= -1
    return oriented


def compute_pca_and_groups(
    matrix: torch.Tensor,
    norms: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    expected_layout = TOKEN_COUNT * 4 * np.dtype("<f4").itemsize
    expected_clusters = TOKEN_COUNT
    if (
        PCA_STAGE_PATH.exists()
        and LAYOUT_PATH.exists()
        and LAYOUT_PATH.stat().st_size == expected_layout
        and CLUSTERS_PATH.exists()
        and CLUSTERS_PATH.stat().st_size == expected_clusters
    ):
        print("Reusing verified-size full-vocabulary PCA stage.", flush=True)
        return json.loads(PCA_STAGE_PATH.read_text(encoding="utf-8"))

    print("Computing full-vocabulary centered covariance…", flush=True)
    started = time.monotonic()
    with torch.inference_mode():
        mean = matrix.mean(dim=0)
        centered = matrix - mean
        covariance = (centered.T @ centered) / (TOKEN_COUNT - 1)
        if device.type == "mps":
            torch.mps.synchronize()
        total_variance = float(torch.trace(covariance).item())
        covariance_cpu = covariance.cpu().numpy()
        del covariance
        if device.type == "mps":
            torch.mps.empty_cache()
    print(f"  covariance complete in {time.monotonic() - started:.1f}s; solving 6,144D eigensystem…", flush=True)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_cpu)
    order = np.argsort(eigenvalues)[::-1][:PCA_COMPONENTS]
    selected_values = np.maximum(eigenvalues[order].astype(np.float64), 0)
    selected_vectors = deterministic_orientation(eigenvectors[:, order].astype(np.float32))
    del eigenvalues, eigenvectors, covariance_cpu

    basis = torch.from_numpy(selected_vectors).to(device)
    with torch.inference_mode():
        scores = centered @ basis
        if device.type == "mps":
            torch.mps.synchronize()
        score_values = scores.cpu().numpy().astype(np.float32, copy=False)
    del scores, basis, centered
    if device.type == "mps":
        torch.mps.empty_cache()

    axis_scale = np.maximum(np.percentile(np.abs(score_values[:, :3]), 99, axis=0), 1e-9)
    layout = np.memmap(LAYOUT_PATH, dtype="<f4", mode="w+", shape=(TOKEN_COUNT, 4))
    layout[:, :3] = score_values[:, :3]
    layout[:, 3] = norms
    layout.flush()
    del layout

    labels, centroids = spherical_kmeans(score_values[:, :PCA_COMPONENTS], clusters=GEOMETRY_GROUPS)
    cluster_file = np.memmap(CLUSTERS_PATH, dtype=np.uint8, mode="w+", shape=(TOKEN_COUNT,))
    cluster_file[:] = labels.astype(np.uint8)
    cluster_file.flush()
    del cluster_file

    rng = np.random.default_rng(SEED)
    pair_count = 100_000
    left = rng.integers(0, TOKEN_COUNT, pair_count)
    right = rng.integers(0, TOKEN_COUNT, pair_count)
    collisions = left == right
    right[collisions] = (right[collisions] + 1) % TOKEN_COUNT
    with torch.inference_mode():
        left_tensor = torch.from_numpy(left.astype(np.int64)).to(device)
        right_tensor = torch.from_numpy(right.astype(np.int64)).to(device)
        random_cosines = torch.sum(matrix[left_tensor] * matrix[right_tensor], dim=1).cpu().numpy()
    quantile_levels = np.linspace(0, 1, 1001)
    quantiles = np.quantile(random_cosines, quantile_levels)

    stage = {
        "method": "Full-vocabulary PCA of exact BF16 rows expanded and L2-normalized in FP32",
        "arithmetic": "FP32",
        "population": TOKEN_COUNT,
        "sourceDimensions": DIMENSIONS,
        "componentsComputed": PCA_COMPONENTS,
        "layoutDimensions": 3,
        "explainedVariance": [compact(value / total_variance) for value in selected_values[:3]],
        "cumulativeExplainedVariance": compact(float(selected_values[:3].sum() / total_variance)),
        "axisScaleP99": [compact(float(value)) for value in axis_scale],
        "geometryGroups": GEOMETRY_GROUPS,
        "randomPairCosine": {
            "count": pair_count,
            "minimum": compact(float(random_cosines.min())),
            "p50": compact(float(np.quantile(random_cosines, 0.5))),
            "p95": compact(float(np.quantile(random_cosines, 0.95))),
            "p99": compact(float(np.quantile(random_cosines, 0.99))),
            "maximum": compact(float(random_cosines.max())),
            "quantileCount": len(quantiles),
            "quantiles": [compact(float(value), 7) for value in quantiles],
        },
        "clusterCentroids": [[compact(float(value), 7) for value in row] for row in centroids],
    }
    atomic_json(PCA_STAGE_PATH, stage)
    print(
        f"  PCA complete: {100 * stage['cumulativeExplainedVariance']:.3f}% variance in 3D; "
        f"{GEOMETRY_GROUPS} geometry groups.",
        flush=True,
    )
    return stage


def compute_exact_neighbours(matrix: torch.Tensor, device: torch.device, query_rows: int) -> None:
    expected_ids_bytes = TOKEN_COUNT * K_NEIGHBOURS * np.dtype("<u4").itemsize
    expected_cosines_bytes = TOKEN_COUNT * K_NEIGHBOURS * np.dtype("<f4").itemsize
    completed_rows = 0
    if NEIGHBOUR_STAGE_PATH.exists():
        try:
            stage = json.loads(NEIGHBOUR_STAGE_PATH.read_text(encoding="utf-8"))
            if (
                stage.get("tokenCount") == TOKEN_COUNT
                and stage.get("neighborsPerToken") == K_NEIGHBOURS
                and stage.get("queryRows") == query_rows
                and NEIGHBOUR_IDS_PATH.exists()
                and NEIGHBOUR_IDS_PATH.stat().st_size == expected_ids_bytes
                and NEIGHBOUR_COSINES_PATH.exists()
                and NEIGHBOUR_COSINES_PATH.stat().st_size == expected_cosines_bytes
            ):
                completed_rows = int(stage.get("completedRows", 0))
        except (OSError, ValueError, KeyError):
            completed_rows = 0

    mode = "r+" if completed_rows else "w+"
    ids_output = np.memmap(
        NEIGHBOUR_IDS_PATH,
        dtype="<u4",
        mode=mode,
        shape=(TOKEN_COUNT, K_NEIGHBOURS),
    )
    cosine_output = np.memmap(
        NEIGHBOUR_COSINES_PATH,
        dtype="<f4",
        mode=mode,
        shape=(TOKEN_COUNT, K_NEIGHBOURS),
    )
    if completed_rows >= TOKEN_COUNT:
        print("Reusing completed exhaustive full-vocabulary neighbour graph.", flush=True)
        return

    print(
        f"Computing exhaustive top-{K_NEIGHBOURS} cosine neighbours across all {TOKEN_COUNT:,} entries "
        f"on {device.type}; resuming at row {completed_rows:,}…",
        flush=True,
    )
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(completed_rows, TOKEN_COUNT, query_rows):
            end = min(TOKEN_COUNT, start + query_rows)
            scores = matrix[start:end] @ matrix.T
            local_rows = torch.arange(end - start, device=device)
            global_rows = torch.arange(start, end, device=device)
            scores[local_rows, global_rows] = -torch.inf
            values, indices = torch.topk(scores, k=K_NEIGHBOURS, dim=1, largest=True, sorted=True)
            ids_output[start:end] = indices.cpu().numpy().astype(np.uint32, copy=False)
            cosine_output[start:end] = values.cpu().numpy().astype(np.float32, copy=False)
            del scores, values, indices, local_rows, global_rows
            if device.type == "mps":
                torch.mps.synchronize()
            ids_output.flush()
            cosine_output.flush()
            stage = {
                "tokenCount": TOKEN_COUNT,
                "neighborsPerToken": K_NEIGHBOURS,
                "queryRows": query_rows,
                "completedRows": end,
                "arithmetic": "FP32",
            }
            atomic_json(NEIGHBOUR_STAGE_PATH, stage)
            elapsed = max(time.monotonic() - started, 0.001)
            processed = end - completed_rows
            remaining = TOKEN_COUNT - end
            eta = remaining / max(processed / elapsed, 1e-9)
            print(
                f"  neighbours {end:>7,}/{TOKEN_COUNT:,} · {processed / elapsed:,.0f} queries/s · ETA {eta:,.0f}s",
                flush=True,
            )
    del ids_output, cosine_output


def token_records_and_clusters(vocabulary: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = np.memmap(CLUSTERS_PATH, dtype=np.uint8, mode="r", shape=(TOKEN_COUNT,))
    layout = np.memmap(LAYOUT_PATH, dtype="<f4", mode="r", shape=(TOKEN_COUNT, 4))
    records: list[dict[str, Any]] = []
    for entry in vocabulary:
        token_id = int(entry["id"])
        records.append(
            {
                "id": token_id,
                "display": entry["display"],
                "raw": entry["raw"],
                "type": entry["type"],
                "script": entry["script"],
                "leadingSpace": bool(entry["leadingSpace"]),
                "validUtf8": bool(entry["validUtf8"]),
                "cluster": int(labels[token_id]),
            }
        )

    clusters: list[dict[str, Any]] = []
    for cluster_id in range(GEOMETRY_GROUPS):
        member_ids = np.flatnonzero(labels == cluster_id)
        centroid = np.mean(layout[member_ids, :3], axis=0)
        distances = np.linalg.norm(layout[member_ids, :3] - centroid, axis=1)
        representative_ids = member_ids[np.argsort(distances)[:8]]
        dominant_type = Counter(records[int(index)]["type"] for index in member_ids).most_common(1)[0][0]
        clusters.append(
            {
                "id": cluster_id,
                "label": f"Geometry group {cluster_id + 1}",
                "count": int(len(member_ids)),
                "dominantType": dominant_type,
                "representativeTokenIds": [int(index) for index in representative_ids],
                "representatives": [records[int(index)]["display"] for index in representative_ids],
            }
        )
    del labels, layout
    return records, clusters


def validate_neighbour_files() -> dict[str, Any]:
    ids = np.memmap(
        NEIGHBOUR_IDS_PATH,
        dtype="<u4",
        mode="r",
        shape=(TOKEN_COUNT, K_NEIGHBOURS),
    )
    cosines = np.memmap(
        NEIGHBOUR_COSINES_PATH,
        dtype="<f4",
        mode="r",
        shape=(TOKEN_COUNT, K_NEIGHBOURS),
    )
    probes = np.array([0, 1, 4, TOKEN_COUNT // 4, TOKEN_COUNT // 2, TOKEN_COUNT - 2, TOKEN_COUNT - 1])
    if not np.all(ids[probes] < TOKEN_COUNT):
        raise RuntimeError("Neighbour IDs are outside the tokenizer vocabulary")
    if np.any(ids[probes] == probes[:, None]):
        raise RuntimeError("Neighbour graph contains a self-match")
    if not np.all(np.isfinite(cosines[probes])) or np.any(cosines[probes] < -1.00001) or np.any(cosines[probes] > 1.00001):
        raise RuntimeError("Neighbour cosine values are invalid")
    if np.any(np.diff(cosines[probes], axis=1) > 1e-6):
        raise RuntimeError("Neighbour rows are not sorted")
    minimum = float(np.min(cosines[probes]))
    maximum = float(np.max(cosines[probes]))
    return {"probeRows": probes.tolist(), "probeMinimum": compact(minimum), "probeMaximum": compact(maximum)}


def write_outputs(vocabulary: list[dict[str, Any]], pca: dict[str, Any], device: torch.device) -> None:
    records, clusters = token_records_and_clusters(vocabulary)
    type_counts = Counter(record["type"] for record in records)
    script_counts = Counter(record["script"] for record in records)
    vocab_output = {
        "schemaVersion": 2,
        "count": TOKEN_COUNT,
        "idRule": "Array position equals tokenizer token ID.",
        "typeCounts": dict(sorted(type_counts.items())),
        "scriptCounts": dict(sorted(script_counts.items())),
        "tokens": records,
    }
    atomic_json(VOCAB_PATH, vocab_output, compact_json=True)

    neighbour_validation = validate_neighbour_files()
    vector_manifest = json.loads(VECTOR_MANIFEST_PATH.read_text(encoding="utf-8"))
    files = {
        "vocabulary": {"file": VOCAB_PATH.name, "bytes": VOCAB_PATH.stat().st_size, "sha256": sha256_file(VOCAB_PATH)},
        "layout": {"file": LAYOUT_PATH.name, "bytes": LAYOUT_PATH.stat().st_size, "sha256": sha256_file(LAYOUT_PATH)},
        "clusters": {"file": CLUSTERS_PATH.name, "bytes": CLUSTERS_PATH.stat().st_size, "sha256": sha256_file(CLUSTERS_PATH)},
        "neighborIds": {"file": NEIGHBOUR_IDS_PATH.name, "bytes": NEIGHBOUR_IDS_PATH.stat().st_size, "sha256": sha256_file(NEIGHBOUR_IDS_PATH)},
        "neighborCosines": {"file": NEIGHBOUR_COSINES_PATH.name, "bytes": NEIGHBOUR_COSINES_PATH.stat().st_size, "sha256": sha256_file(NEIGHBOUR_COSINES_PATH)},
        "vectors": {
            "file": VECTOR_PATH.name,
            "bytes": VECTOR_PATH.stat().st_size,
            "sha256": vector_manifest["sha256"],
        },
    }
    manifest = {
        "schemaVersion": 2,
        "completeVocabulary": True,
        "generatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {
            "repository": REPO,
            "commit": COMMIT,
            "license": "apache-2.0",
            "tokenizerUrl": TOKENIZER_URL,
            "shardUrl": SHARD_URL,
        },
        "tokenCount": TOKEN_COUNT,
        "mappedTokens": TOKEN_COUNT,
        "totalTokenizerEntries": TOKEN_COUNT,
        "coverage": 1,
        "dimensions": DIMENSIONS,
        "neighborsPerToken": K_NEIGHBOURS,
        "layoutStrideFloats": 4,
        "layoutFields": ["pcaX", "pcaY", "pcaZ", "rawRowNorm"],
        "tensor": {
            "name": "model.llm.embed.weight",
            "shape": [VOCAB_ROWS, DIMENSIONS],
            "mappedShape": [TOKEN_COUNT, DIMENSIONS],
            "dtype": "BF16",
            "rowBytes": ROW_BYTES,
            "absoluteStartInShard": ABSOLUTE_START,
            "excludedPaddingRows": VOCAB_ROWS - TOKEN_COUNT,
            "vectorFileOffsetRule": "token_id * rowBytes",
        },
        "projection": pca,
        "neighbors": {
            "method": "Exhaustive full-vocabulary cosine search",
            "population": TOKEN_COUNT,
            "normalization": "Exact BF16 rows expanded and L2-normalized in FP32",
            "arithmetic": "FP32",
            "scope": "Every tokenizer entry was compared against every other tokenizer entry.",
            "sorted": True,
            "selfExcluded": True,
            "validation": neighbour_validation,
        },
        "clusters": clusters,
        "files": files,
        "buildDevice": device.type,
    }
    atomic_json(MANIFEST_PATH, manifest)
    print(f"Wrote {VOCAB_PATH} ({VOCAB_PATH.stat().st_size:,} bytes)", flush=True)
    print(f"Wrote {MANIFEST_PATH} ({MANIFEST_PATH.stat().st_size:,} bytes)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--query-rows", type=int, default=QUERY_ROWS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.query_rows < 64 or args.query_rows > 2_048:
        raise SystemExit("--query-rows must be between 64 and 2048")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    vocabulary = load_vocabulary()
    if len(vocabulary) != TOKEN_COUNT or any(int(entry["id"]) != index for index, entry in enumerate(vocabulary)):
        raise RuntimeError("Tokenizer entries are not the complete contiguous ID range 0..200057")
    matrix, norms = load_normalized_matrix(device)
    pca = compute_pca_and_groups(matrix, norms, device)
    compute_exact_neighbours(matrix, device, args.query_rows)
    write_outputs(vocabulary, pca, device)
    print(
        f"COMPLETE: mapped and globally related all {TOKEN_COUNT:,} tokenizer entries; "
        f"{K_NEIGHBOURS} exhaustive neighbours each.",
        flush=True,
    )


if __name__ == "__main__":
    main()
