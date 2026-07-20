#!/usr/bin/env python3
"""Build a compact, exact token-embedding atlas for Inkling.

The full input embedding tensor is 2.47 GB. This builder selects a deterministic,
readable 2,048-token atlas, range-reads the complete 6,144-value BF16 row for
every selected token, and derives PCA positions and exact cosine neighbours.
"""

from __future__ import annotations

import base64
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import random
import re
import struct
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO = "thinkingmachines/Inkling"
COMMIT = "86b4d430ab871652a707666b89203a866888c5e5"
HF_ROOT = "https://huggingface.co"
TOKENIZER_URL = f"{HF_ROOT}/{REPO}/resolve/{COMMIT}/tiktoken/tokenizer.model"
SHARD = "model-00044-of-00108.safetensors"
SHARD_URL = f"{HF_ROOT}/{REPO}/resolve/{COMMIT}/{SHARD}"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = PROJECT_ROOT / "data" / "inkling-embedding-map.json"
VECTOR_PATH = PROJECT_ROOT / "data" / "inkling-embedding-vectors.bin"
USER_AGENT = "InklingEmbeddingAtlas/1.0"

TENSOR_NAME = "model.llm.embed.weight"
VOCAB_ROWS = 201_024
TOKENIZER_ENTRIES = 200_058
BASE_TOKENS = 199_998
UNMAPPED_ROWS = VOCAB_ROWS - TOKENIZER_ENTRIES
DIMENSIONS = 6_144
BYTES_PER_VALUE = 2
ROW_BYTES = DIMENSIONS * BYTES_PER_VALUE
ABSOLUTE_START = 4_476
TARGET_TOKENS = 2_048
NEIGHBOURS = 12
SEED = 42
MAX_WORKERS = 24

ACTIVE_SPECIAL_IDS = {
    199_999,
    200_000,
    200_001,
    200_002,
    200_003,
    200_004,
    200_005,
    200_006,
    200_008,
    200_010,
    200_020,
    200_022,
    200_023,
    200_024,
    200_028,
    200_043,
    200_049,
    200_057,
}

ANCHOR_WORDS = """
person people human man woman boy girl child children family mother father sister brother
friend love hate happy sad joy fear anger calm hope trust truth lie good bad kind cruel
king queen prince princess leader worker teacher doctor nurse artist writer musician scientist
dog cat horse cow sheep lion tiger bear wolf fox bird fish whale dolphin elephant monkey
tree flower forest river ocean sea mountain valley desert rain snow wind fire earth water air
sun moon star planet space universe world nature life death body mind heart brain eye hand
red blue green yellow orange purple black white colour light dark hot cold big small old young
one two three four five ten hundred thousand first last more less many few same different
today tomorrow yesterday morning evening night time year month week day future past history
home house room city village country nation street road school university hospital market bank
Australia Melbourne Sydney India China Japan France Germany Italy Spain London Paris New York
language word sentence story book music song film movie art game sport food coffee tea bread
science physics chemistry biology mathematics number equation energy matter atom cell gene
computer machine software hardware data code program function class model network algorithm
artificial intelligence learning knowledge reason memory attention token vector embedding
internet web cloud server database Python JavaScript Java HTML JSON Linux Apple Microsoft
work money business company government law politics society culture religion health education
car train plane ship phone camera robot tool table chair door window clock map paper
question answer idea problem solution cause effect change create build make use know think feel
run walk sit stand speak hear see read write learn teach help choose give take open close
fast slow strong weak high low near far left right inside outside before after begin end
freedom peace war power justice equality beauty danger safety success failure possible impossible
""".split()

STRUCTURE_TEXT = """
0 1 2 3 4 5 10 42 100 1000 true false null None return import export const let var def class
function async await SELECT FROM WHERE JSON HTML CSS Python JavaScript C C++ Rust Swift Java
http https www com org net email file path error warning input output array object string integer
+ - * / = == != <= >= -> => { } [ ] ( ) : ; , . ? ! # @ $ % & | _
""".split()


def request_bytes(url: str, byte_range: tuple[int, int] | None = None, retries: int = 5) -> bytes:
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if byte_range:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            if byte_range:
                expected = byte_range[1] - byte_range[0] + 1
                if len(payload) != expected:
                    raise RuntimeError(f"Range {byte_range} returned {len(payload):,} bytes; expected {expected:,}")
            return payload
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, RuntimeError) as exc:
            if attempt == retries - 1:
                raise
            time.sleep(0.65 * (2**attempt))
    raise AssertionError("unreachable")


def byte_encoder() -> dict[int, str]:
    values = list(range(ord("!"), ord("~") + 1))
    values += list(range(ord("¡"), ord("¬") + 1))
    values += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = values[:]
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            codepoints.append(256 + extra)
            extra += 1
    return dict(zip(values, map(chr, codepoints)))


BYTE_ENCODER = byte_encoder()


def visible_text(value: str) -> str:
    return (
        value.replace(" ", "␠")
        .replace("\n", "↵")
        .replace("\r", "␍")
        .replace("\t", "⇥")
        .replace("\x00", "␀")
    )


def script_for(value: str) -> str:
    for character in value:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        for key in ("LATIN", "CYRILLIC", "ARABIC", "DEVANAGARI", "HEBREW", "GREEK", "HANGUL", "THAI"):
            if key in name:
                return key.title()
        if any(label in name for label in ("CJK", "IDEOGRAPH", "HIRAGANA", "KATAKANA")):
            return "CJK/Japanese"
        return "Other script"
    return "None"


def is_word_content(value: str) -> bool:
    if not value or len(value) > 28:
        return False
    return all(character.isalpha() or character in "-'’" for character in value)


def classify_token(token_id: int, decoded: str, valid_utf8: bool) -> str:
    if token_id >= BASE_TOKENS:
        return "special" if token_id in ACTIVE_SPECIAL_IDS else "unused_special"
    if not valid_utf8:
        return "byte_fragment"
    if not decoded or decoded.isspace():
        return "whitespace"
    content = decoded.strip()
    if is_word_content(content):
        return "word"
    if content and all(character.isdigit() or character in ".,:%+-/" for character in content):
        return "numeric"
    if content and all(unicodedata.category(character).startswith(("P", "S")) for character in content):
        return "punctuation"
    return "mixed"


def load_vocabulary() -> list[dict[str, Any]]:
    payload = request_bytes(TOKENIZER_URL)
    entries: list[dict[str, Any] | None] = [None] * TOKENIZER_ENTRIES
    for line in payload.splitlines():
        encoded, raw_id = line.rsplit(b" ", 1)
        token_id = int(raw_id)
        raw_bytes = base64.b64decode(encoded)
        try:
            decoded = raw_bytes.decode("utf-8")
            valid_utf8 = True
        except UnicodeDecodeError:
            decoded = raw_bytes.decode("utf-8", errors="replace")
            valid_utf8 = False
        tokenizer_raw = "".join(BYTE_ENCODER[value] for value in raw_bytes)
        token_type = classify_token(token_id, decoded, valid_utf8)
        entries[token_id] = {
            "id": token_id,
            "bytes": raw_bytes,
            "decoded": decoded,
            "raw": tokenizer_raw,
            "display": visible_text(decoded),
            "type": token_type,
            "script": script_for(decoded),
            "leadingSpace": decoded.startswith(" "),
            "validUtf8": valid_utf8,
        }
    if any(entry is None for entry in entries):
        missing = [index for index, entry in enumerate(entries) if entry is None]
        raise RuntimeError(f"Tokenizer map is incomplete; missing IDs begin {missing[:5]}")
    return [entry for entry in entries if entry is not None]


def evenly_spaced(items: list[int], count: int) -> list[int]:
    if len(items) <= count:
        return items
    return [items[round(index * (len(items) - 1) / max(1, count - 1))] for index in range(count)]


def select_tokens(vocabulary: list[dict[str, Any]]) -> tuple[list[int], dict[int, str]]:
    selected: dict[int, str] = {}
    by_decoded: dict[str, list[int]] = defaultdict(list)
    for entry in vocabulary:
        by_decoded[entry["decoded"]].append(entry["id"])

    def add(token_id: int, selection_class: str) -> None:
        if token_id not in selected and len(selected) < TARGET_TOKENS:
            selected[token_id] = selection_class

    # Curated anchors guarantee that the map contains recognisable relation probes,
    # while retaining distinct leading-space and beginning-of-text variants.
    for word in ANCHOR_WORDS:
        variants = (f" {word}", word, f" {word.capitalize()}", word.capitalize())
        for variant in variants:
            for token_id in by_decoded.get(variant, [])[:1]:
                add(token_id, "semantic_anchor")

    for token_id in sorted(ACTIVE_SPECIAL_IDS):
        add(token_id, "active_control")

    for text in STRUCTURE_TEXT:
        for variant in (f" {text}", text):
            for token_id in by_decoded.get(variant, [])[:1]:
                add(token_id, "numeric_code_symbol")

    structural_candidates = [
        entry["id"]
        for entry in vocabulary
        if entry["type"] in {"numeric", "punctuation"} and entry["id"] < BASE_TOKENS
    ]
    for token_id in evenly_spaced(structural_candidates, 100):
        add(token_id, "numeric_code_symbol")

    script_groups: dict[str, list[int]] = defaultdict(list)
    for entry in vocabulary:
        if entry["type"] == "word" and entry["script"] not in {"Latin", "None"} and len(entry["decoded"].strip()) >= 2:
            script_groups[entry["script"]].append(entry["id"])
    for script in sorted(script_groups):
        for token_id in evenly_spaced(script_groups[script], 28):
            add(token_id, "multilingual")

    common_words = [
        entry["id"]
        for entry in vocabulary
        if entry["type"] == "word"
        and entry["leadingSpace"]
        and entry["decoded"].strip().isascii()
        and entry["decoded"].strip().islower()
        and 3 <= len(entry["decoded"].strip()) <= 18
    ]
    for token_id in common_words[:1_350]:
        add(token_id, "common_word")

    readable = [
        entry["id"]
        for entry in vocabulary[:BASE_TOKENS]
        if entry["validUtf8"]
        and entry["type"] in {"word", "mixed", "numeric"}
        and 1 <= len(entry["decoded"].strip()) <= 28
        and not any(unicodedata.category(character).startswith("C") for character in entry["decoded"])
    ]
    for token_id in evenly_spaced(readable, 900):
        add(token_id, "rank_stratified")

    # Deterministic backfill if earlier categories overlap heavily.
    for token_id in readable:
        add(token_id, "readable_backfill")
        if len(selected) == TARGET_TOKENS:
            break

    if len(selected) != TARGET_TOKENS:
        raise RuntimeError(f"Selected {len(selected):,} tokens; expected {TARGET_TOKENS:,}")
    token_ids = sorted(selected)
    return token_ids, selected


def range_groups(token_ids: list[int], max_gap: int = 20, max_span: int = 160) -> list[tuple[int, int, list[int]]]:
    groups: list[tuple[int, int, list[int]]] = []
    start = token_ids[0]
    previous = start
    members = [start]
    for token_id in token_ids[1:]:
        if token_id - previous <= max_gap and token_id - start < max_span:
            members.append(token_id)
            previous = token_id
            continue
        groups.append((start, previous, members))
        start = previous = token_id
        members = [token_id]
    groups.append((start, previous, members))
    return groups


def fetch_embedding_rows(token_ids: list[int]) -> dict[int, bytes]:
    groups = range_groups(token_ids)
    print(f"Fetching {len(token_ids):,} exact rows in {len(groups):,} verified byte ranges…")

    def fetch_group(group: tuple[int, int, list[int]]) -> tuple[tuple[int, int, list[int]], bytes]:
        start_id, end_id, _ = group
        byte_start = ABSOLUTE_START + start_id * ROW_BYTES
        byte_end = ABSOLUTE_START + (end_id + 1) * ROW_BYTES - 1
        return group, request_bytes(SHARD_URL, (byte_start, byte_end))

    rows: dict[int, bytes] = {}
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(fetch_group, group) for group in groups]
        for future in concurrent.futures.as_completed(futures):
            (start_id, _end_id, members), payload = future.result()
            for token_id in members:
                offset = (token_id - start_id) * ROW_BYTES
                row = payload[offset : offset + ROW_BYTES]
                if len(row) != ROW_BYTES:
                    raise RuntimeError(f"Token {token_id} returned {len(row)} row bytes")
                rows[token_id] = row
            completed += 1
            if completed % 50 == 0 or completed == len(groups):
                print(f"  {completed:,}/{len(groups):,} ranges")
    if set(rows) != set(token_ids):
        raise RuntimeError("Fetched token rows do not match the selected IDs")
    return rows


def bf16_matrix(payload: bytes) -> np.ndarray:
    words = np.frombuffer(payload, dtype="<u2")
    bits = np.left_shift(words.astype(np.uint32), 16)
    return bits.view(np.float32)


def compact(value: float, digits: int = 8) -> float:
    if not math.isfinite(value):
        return value
    return float(f"{value:.{digits}g}")


def randomized_pca(matrix: np.ndarray, components: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = matrix - mean
    omega = rng.standard_normal((matrix.shape[1], components), dtype=np.float32)
    projected = centered @ omega
    for _ in range(2):
        projected = centered @ (centered.T @ projected)
    basis_q, _ = np.linalg.qr(projected, mode="reduced")
    small = basis_q.T @ centered
    _u, singular, vectors_t = np.linalg.svd(small, full_matrices=False)
    scores = centered @ vectors_t.T
    for component in range(vectors_t.shape[0]):
        pivot = int(np.argmax(np.abs(vectors_t[component])))
        if vectors_t[component, pivot] < 0:
            vectors_t[component] *= -1
            scores[:, component] *= -1
    total = float(np.sum(centered.astype(np.float64) ** 2))
    explained = (singular.astype(np.float64) ** 2) / max(total, 1e-12)
    return scores, explained, vectors_t, mean


def spherical_kmeans(features: np.ndarray, clusters: int = 12) -> tuple[np.ndarray, np.ndarray]:
    values = features.astype(np.float64)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    centers = [int(np.argmax(np.linalg.norm(values, axis=1)))]
    closest = np.full(values.shape[0], np.inf)
    for _ in range(1, clusters):
        distance = 1 - values @ values[centers[-1]]
        closest = np.minimum(closest, distance)
        centers.append(int(np.argmax(closest)))
    centroids = values[centers].copy()
    labels = np.zeros(values.shape[0], dtype=np.int32)
    for _ in range(40):
        next_labels = np.argmax(values @ centroids.T, axis=1).astype(np.int32)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for cluster in range(clusters):
            members = values[labels == cluster]
            if len(members) == 0:
                continue
            centroid = members.mean(axis=0)
            centroids[cluster] = centroid / max(np.linalg.norm(centroid), 1e-12)
    return labels, centroids


def selection_cache_matches(token_ids: list[int]) -> bool:
    if not MAP_PATH.exists() or not VECTOR_PATH.exists():
        return False
    try:
        existing = json.loads(MAP_PATH.read_text(encoding="utf-8"))
        existing_ids = [token["id"] for token in existing.get("tokens", [])]
        expected_size = len(token_ids) * ROW_BYTES
        return existing_ids == token_ids and VECTOR_PATH.stat().st_size == expected_size
    except (OSError, ValueError, KeyError):
        return False


def main() -> None:
    vocabulary = load_vocabulary()
    token_ids, selection_classes = select_tokens(vocabulary)
    by_id = {entry["id"]: entry for entry in vocabulary}

    if selection_cache_matches(token_ids):
        print("Reusing the existing exact embedding-row binary; selection IDs match.")
        binary = VECTOR_PATH.read_bytes()
    else:
        fetched = fetch_embedding_rows(token_ids)
        binary = b"".join(fetched[token_id] for token_id in token_ids)
        VECTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        VECTOR_PATH.write_bytes(binary)

    expected_bytes = TARGET_TOKENS * ROW_BYTES
    if len(binary) != expected_bytes:
        raise RuntimeError(f"Embedding binary is {len(binary):,} bytes; expected {expected_bytes:,}")

    matrix = bf16_matrix(binary).reshape(TARGET_TOKENS, DIMENSIONS).copy()
    if not np.isfinite(matrix).all():
        raise RuntimeError("Embedding sample contains non-finite values")
    norms = np.linalg.norm(matrix, axis=1).astype(np.float64)
    normalized = matrix / np.maximum(norms[:, None], 1e-12)

    print("Computing exact 6,144D cosine neighbours…")
    similarity = normalized @ normalized.T
    rng = np.random.default_rng(SEED)
    random_left = rng.integers(0, TARGET_TOKENS, 100_000)
    random_right = rng.integers(0, TARGET_TOKENS, 100_000)
    random_right[random_left == random_right] = (random_right[random_left == random_right] + 1) % TARGET_TOKENS
    baseline = np.einsum("ij,ij->i", normalized[random_left], normalized[random_right])
    baseline_sorted = np.sort(baseline)

    np.fill_diagonal(similarity, -np.inf)
    candidate_indices = np.argpartition(-similarity, NEIGHBOURS - 1, axis=1)[:, :NEIGHBOURS]
    candidate_values = np.take_along_axis(similarity, candidate_indices, axis=1)
    order = np.argsort(-candidate_values, axis=1)
    neighbour_indices = np.take_along_axis(candidate_indices, order, axis=1)
    neighbour_values = np.take_along_axis(candidate_values, order, axis=1)
    neighbour_sets = [set(row.tolist()) for row in neighbour_indices]

    print("Computing deterministic PCA projection and geometry groups…")
    scores, explained, vectors_t, mean = randomized_pca(normalized, components=16)
    raw_coordinates = scores[:, :3]
    axis_scale = np.percentile(np.abs(raw_coordinates), 99, axis=0)
    axis_scale = np.maximum(axis_scale, 1e-8)
    positions = np.clip(raw_coordinates / axis_scale, -1.35, 1.35) * 18.0
    labels, centroids = spherical_kmeans(scores[:, :12], clusters=12)

    squared = np.sum(positions**2, axis=1, keepdims=True)
    projected_distances = squared + squared.T - 2 * (positions @ positions.T)
    np.fill_diagonal(projected_distances, np.inf)
    projected_top = np.argpartition(projected_distances, 9, axis=1)[:, :10]
    recall = float(
        np.mean(
            [
                len(set(projected_top[index].tolist()) & set(neighbour_indices[index, :10].tolist())) / 10
                for index in range(TARGET_TOKENS)
            ]
        )
    )
    pair_count = 50_000
    pair_left = rng.integers(0, TARGET_TOKENS, pair_count)
    pair_right = rng.integers(0, TARGET_TOKENS, pair_count)
    pair_right[pair_left == pair_right] = (pair_right[pair_left == pair_right] + 1) % TARGET_TOKENS
    high_distance = 1 - np.einsum("ij,ij->i", normalized[pair_left], normalized[pair_right])
    low_distance = np.linalg.norm(positions[pair_left] - positions[pair_right], axis=1)
    distance_correlation = float(np.corrcoef(high_distance, low_distance)[0, 1])

    token_records: list[dict[str, Any]] = []
    for index, token_id in enumerate(token_ids):
        entry = by_id[token_id]
        byte_start = ABSOLUTE_START + token_id * ROW_BYTES
        neighbours: list[dict[str, Any]] = []
        for rank in range(NEIGHBOURS):
            neighbour_index = int(neighbour_indices[index, rank])
            cosine = float(neighbour_values[index, rank])
            percentile = 100 * float(np.searchsorted(baseline_sorted, cosine, side="right")) / len(baseline_sorted)
            neighbours.append(
                {
                    "index": neighbour_index,
                    "id": token_ids[neighbour_index],
                    "cosine": compact(cosine),
                    "percentile": compact(percentile, 6),
                    "mutual": index in neighbour_sets[neighbour_index],
                }
            )
        token_records.append(
            {
                "index": index,
                "id": token_id,
                "raw": entry["raw"],
                "display": entry["display"],
                "type": entry["type"],
                "script": entry["script"],
                "leadingSpace": entry["leadingSpace"],
                "selectionClass": selection_classes[token_id],
                "norm": compact(float(norms[index])),
                "position": [compact(float(value)) for value in positions[index]],
                "pca": [compact(float(value)) for value in raw_coordinates[index]],
                "cluster": int(labels[index]),
                "sourceByteRange": [byte_start, byte_start + ROW_BYTES - 1],
                "neighbors": neighbours,
            }
        )

    clusters: list[dict[str, Any]] = []
    for cluster in range(centroids.shape[0]):
        member_indices = np.flatnonzero(labels == cluster)
        member_features = scores[member_indices, :12].astype(np.float64)
        member_features /= np.maximum(np.linalg.norm(member_features, axis=1, keepdims=True), 1e-12)
        closeness = member_features @ centroids[cluster]
        representatives = member_indices[np.argsort(-closeness)[:5]]
        dominant_type = Counter(token_records[index]["type"] for index in member_indices).most_common(1)[0][0]
        clusters.append(
            {
                "id": cluster,
                "label": f"Geometry group {cluster + 1}",
                "count": int(len(member_indices)),
                "dominantType": dominant_type,
                "representativeTokenIds": [token_ids[int(index)] for index in representatives],
                "representatives": [token_records[int(index)]["display"] for index in representatives],
            }
        )

    checksum = hashlib.sha256(binary).hexdigest()
    selection_counts = Counter(selection_classes.values())
    output = {
        "source": {
            "repository": REPO,
            "commit": COMMIT,
            "generatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
            "tokenizerUrl": TOKENIZER_URL,
            "shardUrl": SHARD_URL,
            "license": "apache-2.0",
            "method": (
                "Complete BF16 rows were byte-range read from the pinned input embedding tensor. "
                "Cosine neighbours were calculated from L2-normalized 6,144D raw learned rows; "
                "3D coordinates are a deterministic PCA projection."
            ),
        },
        "tensor": {
            "name": TENSOR_NAME,
            "shape": [VOCAB_ROWS, DIMENSIONS],
            "dtype": "BF16",
            "shard": SHARD,
            "absoluteStart": ABSOLUTE_START,
            "rowBytes": ROW_BYTES,
            "payloadBytes": VOCAB_ROWS * ROW_BYTES,
            "tokenizerEntries": TOKENIZER_ENTRIES,
            "baseTokens": BASE_TOKENS,
            "unmappedRows": UNMAPPED_ROWS,
            "binaryFile": "inkling-embedding-vectors.bin",
            "binaryBytes": len(binary),
            "binarySha256": checksum,
        },
        "selection": {
            "mappedTokens": TARGET_TOKENS,
            "totalTokenizerEntries": TOKENIZER_ENTRIES,
            "coverage": compact(TARGET_TOKENS / TOKENIZER_ENTRIES),
            "seed": SEED,
            "classes": dict(sorted(selection_counts.items())),
            "method": (
                "Representative atlas: common complete tokens, vocabulary-rank strata, multilingual tokens, "
                "semantic relation probes, numeric/code/symbol tokens, and 18 active control tokens."
            ),
            "neighbourScope": "Nearest neighbours are exact within the displayed 2,048-token atlas, not the full vocabulary.",
        },
        "projection": {
            "method": "Deterministic randomized PCA of L2-normalized raw embedding rows",
            "seed": SEED,
            "sourceDimensions": DIMENSIONS,
            "dimensions": 3,
            "explainedVariance": [compact(float(value)) for value in explained[:3]],
            "cumulativeExplainedVariance": compact(float(explained[:3].sum())),
            "axisScale": [compact(float(value)) for value in axis_scale],
            "neighbourRecallAt10": compact(recall),
            "distanceCorrelation": compact(distance_correlation),
            "randomPairCosine": {
                "count": len(baseline),
                "minimum": compact(float(baseline.min())),
                "p50": compact(float(np.percentile(baseline, 50))),
                "p95": compact(float(np.percentile(baseline, 95))),
                "p99": compact(float(np.percentile(baseline, 99))),
                "maximum": compact(float(baseline.max())),
            },
        },
        "clusters": clusters,
        "tokens": token_records,
    }

    MAP_PATH.write_text(json.dumps(output, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {MAP_PATH} ({MAP_PATH.stat().st_size:,} bytes)")
    print(f"Wrote {VECTOR_PATH} ({VECTOR_PATH.stat().st_size:,} bytes, SHA-256 {checksum[:16]}…)")
    print(f"Mapped {TARGET_TOKENS:,} of {TOKENIZER_ENTRIES:,} tokenizer entries")
    print(f"PCA 3D variance: {100 * explained[:3].sum():.3f}% · neighbour recall@10: {100 * recall:.2f}%")


if __name__ == "__main__":
    main()
