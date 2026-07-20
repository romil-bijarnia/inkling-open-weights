#!/usr/bin/env python3
"""Validate the complete Inkling tokenizer-embedding atlas."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "inkling-embedding-full-manifest.json"
VOCAB_PATH = DATA_DIR / "inkling-embedding-vocab.json"
LAYOUT_PATH = DATA_DIR / "inkling-embedding-layout.bin"
CLUSTERS_PATH = DATA_DIR / "inkling-embedding-clusters.bin"
NEIGHBOUR_IDS_PATH = DATA_DIR / "inkling-embedding-neighbor-ids.bin"
NEIGHBOUR_COSINES_PATH = DATA_DIR / "inkling-embedding-neighbor-cosines.bin"
VECTOR_PATH = DATA_DIR / "inkling-embedding-vectors-full.bin"
SAMPLE_MAP_PATH = DATA_DIR / "inkling-embedding-map.json"
SAMPLE_VECTOR_PATH = DATA_DIR / "inkling-embedding-vectors.bin"

EXPECTED_COMMIT = "86b4d430ab871652a707666b89203a866888c5e5"
TOKEN_COUNT = 200_058
VOCAB_ROWS = 201_024
DIMENSIONS = 6_144
ROW_BYTES = 12_288
K = 32


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, message: str) -> None:
        self.count += 1
        if not condition:
            raise AssertionError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def decode_bf16(words: np.ndarray) -> np.ndarray:
    bits = np.left_shift(words.astype(np.uint32), 16)
    return bits.view(np.float32)


def verify_sample_rows(checks: Checks, vocab_vectors: np.memmap) -> None:
    sample_map = json.loads(SAMPLE_MAP_PATH.read_text(encoding="utf-8"))
    sample_words = np.fromfile(SAMPLE_VECTOR_PATH, dtype="<u2").reshape(-1, DIMENSIONS)
    for sample_index, token in enumerate(sample_map["tokens"]):
        token_id = int(token["id"])
        checks.require(
            np.array_equal(vocab_vectors[token_id], sample_words[sample_index]),
            f"Full BF16 row differs from verified sample token {token_id}",
        )


def recompute_probe_neighbours(
    checks: Checks,
    vectors: np.memmap,
    layout: np.memmap,
    stored_ids: np.memmap,
    stored_cosines: np.memmap,
) -> None:
    probes = np.array([0, 4, 6962, 8773, 13793, 14505, 153449, TOKEN_COUNT - 1], dtype=np.int64)
    query_values = decode_bf16(vectors[probes]).reshape(len(probes), DIMENSIONS)
    query_norms = np.linalg.norm(query_values, axis=1)
    checks.require(
        bool(np.allclose(query_norms, layout[probes, 3], rtol=2e-6, atol=2e-5)),
        "Stored layout norms differ from exact BF16 rows",
    )
    queries = query_values / np.maximum(query_norms[:, None], 1e-12)
    all_scores = np.empty((len(probes), TOKEN_COUNT), dtype=np.float32)
    chunk_rows = 2_048
    for start in range(0, TOKEN_COUNT, chunk_rows):
        end = min(TOKEN_COUNT, start + chunk_rows)
        values = decode_bf16(vectors[start:end]).reshape(end - start, DIMENSIONS)
        norms = np.linalg.norm(values, axis=1)
        normalized = values / np.maximum(norms[:, None], 1e-12)
        all_scores[:, start:end] = queries @ normalized.T
    all_scores[np.arange(len(probes)), probes] = -np.inf
    candidates = np.argpartition(-all_scores, K - 1, axis=1)[:, :K]
    candidate_scores = np.take_along_axis(all_scores, candidates, axis=1)
    order = np.argsort(-candidate_scores, axis=1)
    expected_ids = np.take_along_axis(candidates, order, axis=1)
    expected_scores = np.take_along_axis(candidate_scores, order, axis=1)
    for row, token_id in enumerate(probes):
        checks.require(
            np.array_equal(expected_ids[row], stored_ids[token_id]),
            f"Global top-{K} IDs do not recompute for token {token_id}",
        )
        checks.require(
            bool(np.allclose(expected_scores[row], stored_cosines[token_id], rtol=2e-6, atol=2e-6)),
            f"Global cosine values do not recompute for token {token_id}",
        )


def main() -> None:
    checks = Checks()
    required = [
        MANIFEST_PATH,
        VOCAB_PATH,
        LAYOUT_PATH,
        CLUSTERS_PATH,
        NEIGHBOUR_IDS_PATH,
        NEIGHBOUR_COSINES_PATH,
        VECTOR_PATH,
    ]
    for path in required:
        checks.require(path.exists(), f"Missing {path}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    tokens = vocab["tokens"]
    checks.require(manifest["source"]["commit"] == EXPECTED_COMMIT, "Checkpoint commit drifted")
    checks.require(manifest["completeVocabulary"] is True, "Manifest does not claim complete vocabulary")
    checks.require(manifest["tokenCount"] == TOKEN_COUNT, "Token count mismatch")
    checks.require(manifest["mappedTokens"] == manifest["totalTokenizerEntries"] == TOKEN_COUNT, "Coverage denominator mismatch")
    checks.require(manifest["coverage"] == 1, "Vocabulary coverage is not 100%")
    checks.require(manifest["dimensions"] == DIMENSIONS, "Embedding width mismatch")
    checks.require(manifest["neighborsPerToken"] == K, "Neighbour count mismatch")
    checks.require(manifest["layoutStrideFloats"] == 4, "Layout stride mismatch")
    tensor = manifest["tensor"]
    checks.require(tensor["shape"] == [VOCAB_ROWS, DIMENSIONS], "Full tensor shape mismatch")
    checks.require(tensor["mappedShape"] == [TOKEN_COUNT, DIMENSIONS], "Mapped tensor shape mismatch")
    checks.require(tensor["excludedPaddingRows"] == VOCAB_ROWS - TOKEN_COUNT == 966, "Padding-row ledger mismatch")
    checks.require(tensor["rowBytes"] == ROW_BYTES, "Row byte count mismatch")
    checks.require(len(tokens) == TOKEN_COUNT, "Vocabulary JSON is incomplete")

    for token_id, token in enumerate(tokens):
        checks.require(token["id"] == token_id, f"Vocabulary order broke at token {token_id}")
        checks.require(isinstance(token["display"], str) and isinstance(token["raw"], str), f"Token {token_id} lacks text")
        checks.require(0 <= token["cluster"] < manifest["projection"]["geometryGroups"], f"Token {token_id} cluster is invalid")

    lowered = [token["display"].lower() for token in tokens]
    checks.require(any("aryan" in text for text in lowered), "Full vocabulary search cannot resolve aryan-like entries")
    checks.require(tokens[13793]["display"] == "␠king", "Known token ID 13793 drifted")
    checks.require(tokens[8773]["display"] == "␠woman", "Known token ID 8773 drifted")

    files = manifest["files"]
    for record in files.values():
        path = DATA_DIR / record["file"]
        checks.require(path.stat().st_size == record["bytes"], f"File size mismatch for {path.name}")
        checks.require(sha256_file(path) == record["sha256"], f"Checksum mismatch for {path.name}")

    checks.require(VECTOR_PATH.stat().st_size == TOKEN_COUNT * ROW_BYTES, "Full vector payload length mismatch")
    checks.require(LAYOUT_PATH.stat().st_size == TOKEN_COUNT * 4 * 4, "Layout binary length mismatch")
    checks.require(CLUSTERS_PATH.stat().st_size == TOKEN_COUNT, "Cluster binary length mismatch")
    checks.require(NEIGHBOUR_IDS_PATH.stat().st_size == TOKEN_COUNT * K * 4, "Neighbour-ID binary length mismatch")
    checks.require(NEIGHBOUR_COSINES_PATH.stat().st_size == TOKEN_COUNT * K * 4, "Neighbour-cosine binary length mismatch")

    layout = np.memmap(LAYOUT_PATH, dtype="<f4", mode="r", shape=(TOKEN_COUNT, 4))
    clusters = np.memmap(CLUSTERS_PATH, dtype=np.uint8, mode="r", shape=(TOKEN_COUNT,))
    ids = np.memmap(NEIGHBOUR_IDS_PATH, dtype="<u4", mode="r", shape=(TOKEN_COUNT, K))
    cosines = np.memmap(NEIGHBOUR_COSINES_PATH, dtype="<f4", mode="r", shape=(TOKEN_COUNT, K))
    vectors = np.memmap(VECTOR_PATH, dtype="<u2", mode="r", shape=(TOKEN_COUNT, DIMENSIONS))
    checks.require(bool(np.isfinite(layout).all()), "Layout contains non-finite values")
    checks.require(bool((layout[:, 3] > 0).all()), "Embedding norm is not positive")
    checks.require(bool((clusters < manifest["projection"]["geometryGroups"]).all()), "Cluster IDs are invalid")

    for start in range(0, TOKEN_COUNT, 4_096):
        end = min(TOKEN_COUNT, start + 4_096)
        block_ids = ids[start:end]
        block_cosines = cosines[start:end]
        rows = np.arange(start, end, dtype=np.uint32)[:, None]
        checks.require(bool((block_ids < TOKEN_COUNT).all()), f"Neighbour ID outside vocabulary in rows {start}:{end}")
        checks.require(not bool((block_ids == rows).any()), f"Self-neighbour in rows {start}:{end}")
        checks.require(bool(np.isfinite(block_cosines).all()), f"Non-finite cosine in rows {start}:{end}")
        checks.require(bool((block_cosines >= -1.00001).all() and (block_cosines <= 1.00001).all()), f"Cosine outside [-1,1] in rows {start}:{end}")
        checks.require(bool((np.diff(block_cosines, axis=1) <= 1e-6).all()), f"Unsorted neighbour row in {start}:{end}")
        checks.require(bool((np.diff(np.sort(block_ids, axis=1), axis=1) > 0).all()), f"Duplicate neighbour ID in rows {start}:{end}")

    verify_sample_rows(checks, vectors)
    recompute_probe_neighbours(checks, vectors, layout, ids, cosines)

    variance = manifest["projection"]["explainedVariance"]
    checks.require(len(variance) == 3 and all(0 <= value <= 1 for value in variance), "PCA variance metadata is invalid")
    checks.require(
        math.isclose(sum(variance), manifest["projection"]["cumulativeExplainedVariance"], rel_tol=2e-6, abs_tol=2e-8),
        "PCA cumulative variance mismatch",
    )
    checks.require(sum(cluster["count"] for cluster in manifest["clusters"]) == TOKEN_COUNT, "Geometry-group counts do not sum")

    print(f"PASS: {MANIFEST_PATH}")
    print(f"Validated all {TOKEN_COUNT:,} tokenizer entries × {DIMENSIONS:,} exact BF16 values.")
    print(f"Validated {TOKEN_COUNT * K:,} exhaustive full-vocabulary neighbour relationships.")
    print(f"Checks passed: {checks.count:,}")


if __name__ == "__main__":
    main()
