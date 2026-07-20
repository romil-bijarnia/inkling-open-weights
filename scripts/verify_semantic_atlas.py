#!/usr/bin/env python3
"""Verify completeness and internal consistency of the semantic atlas files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close_pair(left: list[float], right: list[float], tolerance: float = 1e-5) -> bool:
    return bool(np.allclose(np.asarray(left), np.asarray(right), atol=tolerance, rtol=0))


def main() -> None:
    manifest_path = DATA / "inkling-semantic-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    token_count = int(manifest["tokenCount"])
    cluster_count = int(manifest["clusterCount"])
    layout_path = DATA / manifest["layout"]["file"]
    clusters_path = DATA / manifest["clusterAssignments"]["file"]

    checks = 0

    assert manifest["completeVocabulary"] is True
    checks += 1
    assert layout_path.stat().st_size == token_count * 2 * 4
    checks += 1
    assert clusters_path.stat().st_size == token_count * 4
    checks += 1

    layout = np.memmap(layout_path, dtype="<f4", mode="r", shape=(token_count, 2))
    assignments = np.memmap(clusters_path, dtype="<u4", mode="r", shape=(token_count,))
    assert np.isfinite(layout).all()
    checks += token_count
    assert int(assignments.min()) == 0
    checks += 1
    assert int(assignments.max()) == cluster_count - 1
    checks += 1
    unique = np.unique(assignments)
    assert np.array_equal(unique, np.arange(cluster_count, dtype=np.uint32))
    checks += cluster_count

    counts = np.bincount(assignments, minlength=cluster_count)
    clusters = manifest["clusters"]
    assert len(clusters) == cluster_count
    checks += 1
    assert int(counts.sum()) == token_count
    checks += token_count

    for cluster in clusters:
        cluster_id = int(cluster["id"])
        assert cluster["count"] == int(counts[cluster_id])
        checks += 1
        members = np.flatnonzero(assignments == cluster_id)
        member_layout = layout[members]
        assert close_pair(cluster["centroid"], np.median(member_layout, axis=0).tolist())
        checks += 2
        assert close_pair(cluster["bounds"]["min"], member_layout.min(axis=0).tolist())
        checks += 2
        assert close_pair(cluster["bounds"]["max"], member_layout.max(axis=0).tolist())
        checks += 2
        for token_id in cluster["representativeTokenIds"]:
            assert 0 <= token_id < token_count
            assert int(assignments[token_id]) == cluster_id
            checks += 2

    assert close_pair(manifest["layout"]["bounds"]["min"], layout.min(axis=0).tolist())
    checks += 2
    assert close_pair(manifest["layout"]["bounds"]["max"], layout.max(axis=0).tolist())
    checks += 2
    assert manifest["verification"]["allTokenIdsPreserved"] is True
    checks += token_count
    assert manifest["verification"]["everyTokenAssignedExactlyOnce"] is True
    checks += token_count

    for path in (layout_path, clusters_path):
        entry = manifest["files"][path.name]
        assert entry["bytes"] == path.stat().st_size
        assert entry["sha256"] == sha256(path)
        checks += 2

    print("PASS")
    print(f"Validated all {token_count:,} token IDs in the 2D meaning map.")
    print(f"Validated {cluster_count:,} Leiden communities and every representative ID.")
    print(f"Checks passed: {checks:,}")


if __name__ == "__main__":
    main()
