#!/usr/bin/env python3
"""Build the complete 2D meaning map from Inkling's exact neighbour graph.

The input graph already contains the exhaustive top-32 cosine neighbours for
every tokenizer entry.  This script converts those measured relationships into
UMAP's fuzzy simplicial graph, lays all token IDs out in two dimensions, and
finds weighted Leiden communities on the same graph.

Outputs are deliberately small and browser-friendly:

* inkling-semantic-layout.bin: little-endian Float32 [x, y] per token ID
* inkling-semantic-clusters.bin: little-endian Uint32 cluster ID per token ID
* inkling-semantic-manifest.json: method, labels, representatives and checks
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
import umap
from sklearn.utils import check_random_state
from umap.umap_ import fuzzy_simplicial_set


TOKEN_COUNT = 200_058
NEIGHBOURS = 32
SEED = 42
LEIDEN_RESOLUTION = 1.5
LEIDEN_ITERATIONS = 4
UMAP_EPOCHS = 200
UMAP_MIN_DIST = 0.08


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse hidden fuzzy-graph and UMAP caches when their shapes match.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_size(path: Path, expected: int) -> None:
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"{path.name}: expected {expected:,} bytes, got {actual:,}")


def load_fuzzy_cache(path: Path) -> sp.csr_matrix:
    cached = np.load(path)
    shape = tuple(int(value) for value in cached["shape"])
    if shape != (TOKEN_COUNT, TOKEN_COUNT):
        raise ValueError(f"Unexpected cached graph shape: {shape}")
    return sp.csr_matrix(
        (cached["data"], cached["indices"], cached["indptr"]), shape=shape
    )


def build_fuzzy_graph(
    neighbour_ids: np.ndarray,
    neighbour_distances: np.ndarray,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    graph, sigmas, rhos = fuzzy_simplicial_set(
        np.zeros((TOKEN_COUNT, 1), dtype=np.float32),
        n_neighbors=NEIGHBOURS,
        random_state=check_random_state(SEED),
        metric="euclidean",
        knn_indices=neighbour_ids,
        knn_dists=neighbour_distances,
        angular=False,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        apply_set_operations=True,
        verbose=False,
    )
    return graph.tocsr(), sigmas, rhos


def build_umap_layout(
    neighbour_ids: np.ndarray,
    neighbour_distances: np.ndarray,
) -> np.ndarray:
    mapper = umap.UMAP(
        n_neighbors=NEIGHBOURS,
        n_components=2,
        metric="cosine",
        n_epochs=UMAP_EPOCHS,
        learning_rate=1.0,
        init="spectral",
        min_dist=UMAP_MIN_DIST,
        spread=1.0,
        low_memory=True,
        n_jobs=1,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        random_state=SEED,
        verbose=True,
        precomputed_knn=(neighbour_ids, neighbour_distances, None),
    )
    # Coordinates depend only on the supplied precomputed neighbour graph.  A
    # one-column placeholder avoids re-reading the 2.46 GB vector matrix.
    return mapper.fit_transform(np.zeros((TOKEN_COUNT, 1), dtype=np.float32))


def normalise_layout(raw_layout: np.ndarray) -> tuple[np.ndarray, dict]:
    raw_min = raw_layout.min(axis=0).astype(np.float64)
    raw_max = raw_layout.max(axis=0).astype(np.float64)
    center = (raw_min + raw_max) / 2.0
    uniform_span = float(np.max(raw_max - raw_min))
    if not math.isfinite(uniform_span) or uniform_span <= 0:
        raise ValueError("UMAP returned a degenerate layout")
    scale = 2.0 / uniform_span
    layout = ((raw_layout.astype(np.float64) - center) * scale).astype("<f4")
    transform = {
        "kind": "translation plus uniform affine scale",
        "rawMin": raw_min.tolist(),
        "rawMax": raw_max.tolist(),
        "rawCenter": center.tolist(),
        "uniformScale": scale,
        "preservesAspectRatio": True,
    }
    return layout, transform


def clean_display(token: dict) -> str | None:
    value = str(token.get("display", ""))
    value = value.replace("␠", " ").replace("↵", " ").strip()
    value = re.sub(r"\s+", " ", value)
    if not 3 <= len(value) <= 24:
        return None
    if "�" in value or value.startswith("<|"):
        return None
    letters = sum(character.isalpha() for character in value)
    if letters / max(1, len(value)) < 0.72:
        return None
    return value


def representatives_for_cluster(
    cluster_id: int,
    membership: np.ndarray,
    internal_strength: np.ndarray,
    tokens: list[dict],
    limit: int = 12,
) -> tuple[list[int], list[str]]:
    members = np.flatnonzero(membership == cluster_id)
    centrality = internal_strength[members]
    # Token IDs are tiktoken ranks.  The small bounded rank bonus makes common,
    # legible tokens win close centrality ties without overriding graph evidence.
    rank_bonus = 1.0 + 0.8 / np.log2(members.astype(np.float64) + 4.0)
    order = members[np.argsort(-(centrality * rank_bonus), kind="stable")]
    selected_ids: list[int] = []
    selected_text: list[str] = []
    seen: set[str] = set()
    for token_id in order:
        token = tokens[int(token_id)]
        if token.get("type") not in {"word", "mixed"} or not token.get("validUtf8"):
            continue
        display = clean_display(token)
        if not display:
            continue
        key = display.casefold()
        if key in seen:
            continue
        seen.add(key)
        selected_ids.append(int(token_id))
        selected_text.append(display)
        if len(selected_ids) >= limit:
            break
    if selected_ids:
        return selected_ids, selected_text

    # Non-word communities (punctuation/control/bytes) still receive stable IDs.
    fallback = order[:limit]
    return (
        [int(token_id) for token_id in fallback],
        [str(tokens[int(token_id)].get("display", token_id)) for token_id in fallback],
    )


def make_cluster_label(
    representative_text: list[str],
    dominant_type: str,
    dominant_type_share: float,
    dominant_script: str,
    dominant_script_share: float,
) -> str:
    examples = " · ".join(text[:18] for text in representative_text[:3])
    if dominant_type in {"punctuation", "special", "unused_special", "byte_fragment"} and dominant_type_share >= 0.40:
        prefix = {
            "punctuation": "symbols",
            "special": "control tokens",
            "unused_special": "reserved tokens",
            "byte_fragment": "byte fragments",
        }[dominant_type]
        return f"{prefix}: {examples}" if examples else prefix
    if dominant_script not in {"Latin", "None"} and dominant_script_share >= 0.55:
        return f"{dominant_script}: {examples}" if examples else f"{dominant_script} tokens"
    return examples or f"{dominant_script} {dominant_type} tokens"


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    started = time.perf_counter()
    timings: dict[str, float] = {}

    neighbour_ids_path = data_dir / "inkling-embedding-neighbor-ids.bin"
    neighbour_cosines_path = data_dir / "inkling-embedding-neighbor-cosines.bin"
    vocab_path = data_dir / "inkling-embedding-vocab.json"
    source_manifest_path = data_dir / "inkling-embedding-full-manifest.json"
    require_size(neighbour_ids_path, TOKEN_COUNT * NEIGHBOURS * 4)
    require_size(neighbour_cosines_path, TOKEN_COUNT * NEIGHBOURS * 4)

    neighbour_ids = np.memmap(
        neighbour_ids_path, dtype="<u4", mode="r", shape=(TOKEN_COUNT, NEIGHBOURS)
    )
    neighbour_cosines = np.memmap(
        neighbour_cosines_path,
        dtype="<f4",
        mode="r",
        shape=(TOKEN_COUNT, NEIGHBOURS),
    )
    if int(neighbour_ids.min()) < 0 or int(neighbour_ids.max()) >= TOKEN_COUNT:
        raise ValueError("Neighbour IDs fall outside the tokenizer vocabulary")
    if not np.all(np.diff(neighbour_cosines, axis=1) <= 1e-7):
        raise ValueError("Neighbour cosines are not sorted from most to least similar")
    distances = np.maximum(
        np.float32(0.0), np.float32(1.0) - np.asarray(neighbour_cosines, dtype=np.float32)
    )
    ids_i32 = np.asarray(neighbour_ids, dtype=np.int32)

    fuzzy_cache = data_dir / ".semantic-fuzzy-graph-cache.npz"
    stage = time.perf_counter()
    if args.reuse_cache and fuzzy_cache.exists():
        fuzzy = load_fuzzy_cache(fuzzy_cache)
        cache_used_graph = True
    else:
        fuzzy, sigmas, rhos = build_fuzzy_graph(ids_i32, distances)
        np.savez_compressed(
            fuzzy_cache,
            data=fuzzy.data,
            indices=fuzzy.indices,
            indptr=fuzzy.indptr,
            shape=np.asarray(fuzzy.shape, dtype=np.int64),
            sigmas=sigmas,
            rhos=rhos,
        )
        cache_used_graph = False
    timings["fuzzyGraphSeconds"] = time.perf_counter() - stage

    layout_cache = data_dir / ".semantic-umap-layout-cache.npy"
    stage = time.perf_counter()
    if args.reuse_cache and layout_cache.exists():
        raw_layout = np.load(layout_cache)
        if raw_layout.shape != (TOKEN_COUNT, 2):
            raise ValueError(f"Unexpected cached UMAP shape: {raw_layout.shape}")
        cache_used_layout = True
    else:
        raw_layout = build_umap_layout(ids_i32, distances)
        np.save(layout_cache, raw_layout.astype("<f4"))
        cache_used_layout = False
    if not np.isfinite(raw_layout).all():
        raise ValueError("UMAP produced non-finite coordinates")
    layout, layout_transform = normalise_layout(raw_layout)
    timings["umapSeconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    upper = sp.triu(fuzzy, k=1, format="coo")
    edges = np.empty((upper.nnz, 2), dtype=np.int32)
    edges[:, 0] = upper.row
    edges[:, 1] = upper.col
    graph = ig.Graph(n=TOKEN_COUNT, edges=edges, directed=False)
    graph.es["weight"] = upper.data.astype(float, copy=False)
    partition = la.find_partition(
        graph,
        la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=LEIDEN_RESOLUTION,
        n_iterations=LEIDEN_ITERATIONS,
        seed=SEED,
    )
    original_membership = np.asarray(partition.membership, dtype=np.int64)
    original_sizes = np.bincount(original_membership)
    size_order = np.argsort(-original_sizes, kind="stable")
    remap = np.empty_like(size_order)
    remap[size_order] = np.arange(size_order.size)
    membership = remap[original_membership].astype("<u4")
    cluster_count = int(size_order.size)
    cluster_sizes = np.bincount(membership, minlength=cluster_count)
    timings["leidenSeconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    rows = upper.row.astype(np.int64, copy=False)
    cols = upper.col.astype(np.int64, copy=False)
    weights = upper.data.astype(np.float64, copy=False)
    same = membership[rows] == membership[cols]
    internal_strength = np.zeros(TOKEN_COUNT, dtype=np.float64)
    np.add.at(internal_strength, rows[same], weights[same])
    np.add.at(internal_strength, cols[same], weights[same])

    cluster_edges = np.zeros((cluster_count, cluster_count), dtype=np.float64)
    source_clusters = membership[rows]
    target_clusters = membership[cols]
    crossing = source_clusters != target_clusters
    np.add.at(
        cluster_edges,
        (source_clusters[crossing], target_clusters[crossing]),
        weights[crossing],
    )
    cluster_edges += cluster_edges.T

    with vocab_path.open("r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    tokens = vocab["tokens"]
    if len(tokens) != TOKEN_COUNT or any(token["id"] != index for index, token in enumerate(tokens)):
        raise ValueError("Vocabulary is not a complete token-ID-ordered array")

    clusters: list[dict] = []
    for cluster_id in range(cluster_count):
        members = np.flatnonzero(membership == cluster_id)
        representative_ids, representative_text = representatives_for_cluster(
            cluster_id, membership, internal_strength, tokens
        )
        type_counts = Counter(tokens[int(token_id)]["type"] for token_id in members)
        script_counts = Counter(tokens[int(token_id)]["script"] for token_id in members)
        dominant_type, dominant_type_count = type_counts.most_common(1)[0]
        dominant_script, dominant_script_count = script_counts.most_common(1)[0]
        label = make_cluster_label(
            representative_text,
            dominant_type,
            dominant_type_count / members.size,
            dominant_script,
            dominant_script_count / members.size,
        )
        member_layout = layout[members]
        neighbour_order = np.argsort(-cluster_edges[cluster_id], kind="stable")
        neighbours = [
            {"id": int(other), "weight": float(cluster_edges[cluster_id, other])}
            for other in neighbour_order
            if other != cluster_id and cluster_edges[cluster_id, other] > 0
        ][:8]
        clusters.append(
            {
                "id": cluster_id,
                "count": int(members.size),
                "percentage": float(members.size / TOKEN_COUNT),
                "label": label,
                "representativeTokenIds": representative_ids,
                "representativeTokens": representative_text,
                "dominantType": dominant_type,
                "dominantScript": dominant_script,
                "typeCounts": dict(sorted(type_counts.items())),
                "scriptCounts": dict(sorted(script_counts.items())),
                "centroid": np.median(member_layout, axis=0).astype(float).tolist(),
                "bounds": {
                    "min": member_layout.min(axis=0).astype(float).tolist(),
                    "max": member_layout.max(axis=0).astype(float).tolist(),
                },
                "neighbours": neighbours,
            }
        )
    timings["metadataSeconds"] = time.perf_counter() - stage

    layout_path = data_dir / "inkling-semantic-layout.bin"
    clusters_path = data_dir / "inkling-semantic-clusters.bin"
    layout.tofile(layout_path)
    membership.tofile(clusters_path)
    require_size(layout_path, TOKEN_COUNT * 2 * 4)
    require_size(clusters_path, TOKEN_COUNT * 4)

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    connected_components = graph.connected_components(mode="weak")
    timings["totalSeconds"] = time.perf_counter() - started
    manifest = {
        "schemaVersion": 1,
        "completeVocabulary": True,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": source_manifest["source"]["repository"],
            "commit": source_manifest["source"]["commit"],
            "tensor": source_manifest["tensor"]["name"],
            "dimensions": source_manifest["dimensions"],
            "neighbourIdsFile": neighbour_ids_path.name,
            "neighbourCosinesFile": neighbour_cosines_path.name,
        },
        "tokenCount": TOKEN_COUNT,
        "clusterCount": cluster_count,
        "layout": {
            "file": layout_path.name,
            "dtype": "little-endian Float32",
            "recordOrder": "array record index equals tokenizer token ID",
            "strideFloats": 2,
            "fields": ["x", "y"],
            "byteLength": layout_path.stat().st_size,
            "coordinateSpace": "2D UMAP coordinates, centered and uniformly scaled to a maximum global span of 2",
            "bounds": {
                "min": layout.min(axis=0).astype(float).tolist(),
                "max": layout.max(axis=0).astype(float).tolist(),
            },
            "affineTransformFromRawUmap": layout_transform,
        },
        "clusterAssignments": {
            "file": clusters_path.name,
            "dtype": "little-endian Uint32",
            "recordOrder": "array element index equals tokenizer token ID",
            "byteLength": clusters_path.stat().st_size,
            "idRule": "cluster IDs are sorted from largest community to smallest",
        },
        "method": {
            "graph": {
                "name": "UMAP fuzzy simplicial set",
                "input": "all exact top-32 global cosine neighbours for every token",
                "distance": "1 - original 6,144-dimensional cosine similarity",
                "nNeighbours": NEIGHBOURS,
                "setOpMixRatio": 1.0,
                "localConnectivity": 1.0,
                "directedInputRelationships": TOKEN_COUNT * NEIGHBOURS,
                "fuzzyUndirectedEdges": int(upper.nnz),
            },
            "layout": {
                "algorithm": "UMAP",
                "dimensions": 2,
                "epochs": UMAP_EPOCHS,
                "minDist": UMAP_MIN_DIST,
                "spread": 1.0,
                "negativeSampleRate": 5,
                "initialisation": "spectral",
                "randomSeed": SEED,
                "deterministicSingleThread": True,
            },
            "communityDetection": {
                "algorithm": "Leiden RBConfigurationVertexPartition",
                "weightedGraph": "same UMAP fuzzy graph used for layout",
                "resolution": LEIDEN_RESOLUTION,
                "iterations": LEIDEN_ITERATIONS,
                "randomSeed": SEED,
                "quality": float(partition.quality()),
            },
            "labeling": {
                "kind": "data-derived",
                "rule": "up to three readable high-internal-strength representative tokenizer entries; dominant script/type is prefixed when it explains at least the configured share",
                "noExternalTaxonomy": True,
                "centroid": "component-wise median of member coordinates",
            },
        },
        "graph": {
            "nodes": TOKEN_COUNT,
            "connectedComponents": len(connected_components),
            "largestConnectedComponent": max(connected_components.sizes()),
        },
        "clusters": clusters,
        "verification": {
            "allTokenIdsPreserved": True,
            "layoutRecords": int(layout.shape[0]),
            "clusterAssignmentRecords": int(membership.size),
            "finiteCoordinates": bool(np.isfinite(layout).all()),
            "everyTokenAssignedExactlyOnce": int(cluster_sizes.sum()) == TOKEN_COUNT,
            "emptyClusters": int(np.sum(cluster_sizes == 0)),
            "largestCluster": int(cluster_sizes.max()),
            "smallestCluster": int(cluster_sizes.min()),
        },
        "runtime": {
            **{key: round(value, 6) for key, value in timings.items()},
            "cacheUsed": {
                "fuzzyGraph": cache_used_graph,
                "umapLayout": cache_used_layout,
            },
            "platform": platform.platform(),
            "python": platform.python_version(),
            "libraries": {
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikitLearn": sklearn.__version__,
                "umapLearn": umap.__version__,
                "igraph": ig.__version__,
                "leidenalg": la.__version__,
            },
        },
        "files": {
            layout_path.name: {
                "bytes": layout_path.stat().st_size,
                "sha256": sha256(layout_path),
            },
            clusters_path.name: {
                "bytes": clusters_path.stat().st_size,
                "sha256": sha256(clusters_path),
            },
        },
    }
    manifest_path = data_dir / "inkling-semantic-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {TOKEN_COUNT:,} tokens, {cluster_count:,} communities, "
        f"{upper.nnz:,} fuzzy edges in {timings['totalSeconds']:.1f}s"
    )
    print(f"  {layout_path} ({layout_path.stat().st_size:,} bytes)")
    print(f"  {clusters_path} ({clusters_path.stat().st_size:,} bytes)")
    print(f"  {manifest_path} ({manifest_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
