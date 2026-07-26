#!/usr/bin/env python3
"""Validate the generated Inkling token-embedding data and exact BF16 rows."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = PROJECT_ROOT / "data" / "inkling-embedding-map.json"
VECTOR_PATH = PROJECT_ROOT / "data" / "inkling-embedding-vectors.bin"
EXPECTED_COMMIT = "86b4d430ab871652a707666b89203a866888c5e5"
EXPECTED_TENSOR = "model.llm.embed.weight"
EXPECTED_TOKENS = 2_048
TOKENIZER_ENTRIES = 200_058
VOCAB_ROWS = 201_024
DIMENSIONS = 6_144
ROW_BYTES = 12_288
ABSOLUTE_START = 4_476


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, message: str) -> None:
        self.count += 1
        if not condition:
            raise AssertionError(message)


def decode_bf16(payload: bytes) -> np.ndarray:
    words = np.frombuffer(payload, dtype="<u2")
    bits = np.left_shift(words.astype(np.uint32), 16)
    return bits.view(np.float32)


def main() -> None:
    checks = Checks()
    checks.require(MAP_PATH.exists(), f"Missing {MAP_PATH}")
    checks.require(VECTOR_PATH.exists(), f"Missing {VECTOR_PATH}")

    data = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    binary = VECTOR_PATH.read_bytes()
    source = data["source"]
    tensor = data["tensor"]
    selection = data["selection"]
    projection = data["projection"]
    tokens = data["tokens"]

    checks.require(source["commit"] == EXPECTED_COMMIT, "Checkpoint commit drifted")
    checks.require(tensor["name"] == EXPECTED_TENSOR, "Wrong embedding tensor")
    checks.require(tensor["shape"] == [VOCAB_ROWS, DIMENSIONS], "Embedding tensor shape mismatch")
    checks.require(tensor["dtype"] == "BF16", "Embedding dtype must be BF16")
    checks.require(tensor["rowBytes"] == ROW_BYTES, "Row byte count mismatch")
    checks.require(tensor["absoluteStart"] == ABSOLUTE_START, "Absolute tensor start mismatch")
    checks.require(tensor["tokenizerEntries"] == TOKENIZER_ENTRIES, "Tokenizer-entry count mismatch")
    checks.require(tensor["unmappedRows"] == VOCAB_ROWS - TOKENIZER_ENTRIES == 966, "Unmapped-row count mismatch")
    checks.require(len(tokens) == EXPECTED_TOKENS, "Token embedding count mismatch")
    checks.require(selection["mappedTokens"] == EXPECTED_TOKENS, "Selection count mismatch")
    checks.require(selection["totalTokenizerEntries"] == TOKENIZER_ENTRIES, "Selection denominator mismatch")
    checks.require(len(binary) == EXPECTED_TOKENS * ROW_BYTES, "Embedding binary length mismatch")
    checks.require(tensor["binaryBytes"] == len(binary), "Manifest binary length mismatch")
    checks.require(hashlib.sha256(binary).hexdigest() == tensor["binarySha256"], "Embedding binary checksum mismatch")

    ids = [token["id"] for token in tokens]
    checks.require(ids == sorted(ids), "Token records must be sorted by token ID")
    checks.require(len(ids) == len(set(ids)), "Token IDs must be unique")
    checks.require(all(0 <= token_id < TOKENIZER_ENTRIES for token_id in ids), "Embedding selection contains padding rows")
    checks.require(all(token["index"] == index for index, token in enumerate(tokens)), "Token indices are unstable")
    checks.require(set(selection["classes"]) == set(token["selectionClass"] for token in tokens), "Selection-class ledger mismatch")
    checks.require(sum(selection["classes"].values()) == EXPECTED_TOKENS, "Selection-class counts do not sum")

    matrix = decode_bf16(binary).reshape(EXPECTED_TOKENS, DIMENSIONS)
    checks.require(matrix.shape == (EXPECTED_TOKENS, DIMENSIONS), "Decoded matrix shape mismatch")
    checks.require(bool(np.isfinite(matrix).all()), "Decoded embedding matrix contains non-finite values")
    norms = np.linalg.norm(matrix, axis=1).astype(np.float64)
    normalized = matrix / np.maximum(norms[:, None], 1e-12)
    checks.require(bool(np.allclose(np.linalg.norm(normalized, axis=1), 1.0, atol=2e-5)), "L2 normalization failed")

    for index, token in enumerate(tokens):
        expected_start = ABSOLUTE_START + token["id"] * ROW_BYTES
        checks.require(token["sourceByteRange"] == [expected_start, expected_start + ROW_BYTES - 1], f"Token {token['id']} byte range mismatch")
        checks.require(math.isfinite(token["norm"]) and token["norm"] > 0, f"Token {token['id']} norm is invalid")
        checks.require(abs(float(token["norm"]) - norms[index]) <= max(2e-5, norms[index] * 2e-6), f"Token {token['id']} norm mismatch")
        checks.require(len(token["position"]) == 3 and all(math.isfinite(value) for value in token["position"]), f"Token {token['id']} position is invalid")
        checks.require(len(token["pca"]) == 3 and all(math.isfinite(value) for value in token["pca"]), f"Token {token['id']} PCA coordinates are invalid")
        checks.require(len(token["neighbors"]) == 12, f"Token {token['id']} neighbour count mismatch")
        previous = float("inf")
        for neighbour in token["neighbors"]:
            checks.require(neighbour["id"] != token["id"], f"Token {token['id']} contains itself as a neighbour")
            checks.require(0 <= neighbour["index"] < EXPECTED_TOKENS, f"Token {token['id']} neighbour index is invalid")
            checks.require(tokens[neighbour["index"]]["id"] == neighbour["id"], f"Token {token['id']} neighbour ID/index mismatch")
            checks.require(-1.00001 <= neighbour["cosine"] <= 1.00001, f"Token {token['id']} cosine is outside [-1,1]")
            checks.require(neighbour["cosine"] <= previous + 1e-7, f"Token {token['id']} neighbours are not sorted")
            checks.require(0 <= neighbour.get("percentile", 0) <= 100, f"Token {token['id']} percentile is invalid")
            previous = neighbour["cosine"]

    # Recompute complete nearest-neighbour rows for deterministic probes.
    probes = sorted({0, len(tokens) // 4, len(tokens) // 2, 3 * len(tokens) // 4, len(tokens) - 1})
    for index in probes:
        scores = normalized[index] @ normalized.T
        scores[index] = -np.inf
        expected = np.argsort(-scores)[:12]
        stored = np.array([entry["index"] for entry in tokens[index]["neighbors"]])
        checks.require(np.array_equal(expected, stored), f"Token {tokens[index]['id']} nearest-neighbour IDs do not recompute")
        for rank, neighbour_index in enumerate(expected):
            stored_score = tokens[index]["neighbors"][rank]["cosine"]
            checks.require(abs(float(scores[neighbour_index]) - stored_score) < 2e-6, f"Token {tokens[index]['id']} cosine score mismatch")

    displays = {token["display"] for token in tokens}
    checks.require("␠king" in displays, "Semantic probe ␠king is missing")
    checks.require("␠queen" in displays, "Semantic probe ␠queen is missing")
    checks.require("␠woman" in displays, "Semantic probe ␠woman is missing")
    checks.require("␠Paris" in displays, "Semantic probe ␠Paris is missing")
    checks.require("␠France" in displays, "Semantic probe ␠France is missing")
    checks.require(any(token["type"] == "special" for token in tokens), "Active control tokens are missing")
    checks.require(not any(token["type"] == "unused_special" for token in tokens), "Unused special placeholders were included")

    explained = projection["explainedVariance"]
    checks.require(len(explained) == 3 and all(0 <= value <= 1 for value in explained), "PCA explained variance is invalid")
    checks.require(abs(sum(explained) - projection["cumulativeExplainedVariance"]) < 2e-6, "PCA cumulative variance mismatch")
    checks.require(0 <= projection["neighbourRecallAt10"] <= 1, "Projection neighbour recall is invalid")
    checks.require(-1 <= projection["distanceCorrelation"] <= 1, "Projection distance correlation is invalid")
    checks.require(len(data["clusters"]) == 12, "Geometry-group count mismatch")
    checks.require(sum(cluster["count"] for cluster in data["clusters"]) == EXPECTED_TOKENS, "Geometry-group counts do not sum")

    print(f"PASS: {MAP_PATH}")
    print(f"Validated {EXPECTED_TOKENS:,} exact token rows × {DIMENSIONS:,} BF16 dimensions ({len(binary):,} bytes).")
    print(f"Verified pinned source, byte ranges, checksum, norms, PCA metadata, clusters, and exact cosine neighbours.")
    print(f"Checks passed: {checks.count:,}")


if __name__ == "__main__":
    main()
