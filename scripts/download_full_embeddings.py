#!/usr/bin/env python3
"""Download every real Inkling tokenizer embedding row with resumable ranges.

The source safetensors shard is roughly 13 GB, but the tokenizer-backed prefix
of ``model.llm.embed.weight`` is a contiguous 2.46 GB payload. This script
downloads only those 200,058 rows, keeps row order equal to token ID, verifies
the existing 2,048-row sample against the completed file, and records a SHA-256.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO = "thinkingmachines/Inkling"
COMMIT = "86b4d430ab871652a707666b89203a866888c5e5"
SHARD = "model-00044-of-00108.safetensors"
SHARD_URL = f"https://huggingface.co/{REPO}/resolve/{COMMIT}/{SHARD}"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "inkling-embedding-vectors-full.bin"
PARTIAL_PATH = OUTPUT_PATH.with_suffix(".bin.partial")
STATE_DIR = PROJECT_ROOT / "data" / ".embedding-full-download"
SAMPLE_MAP_PATH = PROJECT_ROOT / "data" / "inkling-embedding-map.json"
SAMPLE_VECTOR_PATH = PROJECT_ROOT / "data" / "inkling-embedding-vectors.bin"
MANIFEST_PATH = PROJECT_ROOT / "data" / "inkling-embedding-vectors-full.manifest.json"

TOKENIZER_ENTRIES = 200_058
DIMENSIONS = 6_144
ROW_BYTES = DIMENSIONS * 2
ABSOLUTE_START = 4_476
ROWS_PER_CHUNK = 2_048
WORKERS = 8
EXPECTED_BYTES = TOKENIZER_ENTRIES * ROW_BYTES
USER_AGENT = "InklingFullEmbeddingMap/1.0"


def fetch_range(start: int, end: int, retries: int = 6) -> bytes:
    expected = end - start + 1
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "identity",
        "Range": f"bytes={start}-{end}",
    }
    for attempt in range(retries):
        try:
            request = urllib.request.Request(SHARD_URL, headers=headers)
            with urllib.request.urlopen(request, timeout=180) as response:
                content_range = response.headers.get("Content-Range", "")
                content_length = int(response.headers.get("Content-Length", expected))
                if response.status != 206 and not content_range:
                    raise RuntimeError(
                        f"Server ignored Range {start}-{end} and offered {content_length:,} bytes"
                    )
                payload = response.read(expected + 1)
            if len(payload) != expected:
                raise RuntimeError(
                    f"Range {start}-{end} returned {len(payload):,} bytes; expected {expected:,}"
                )
            return payload
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, RuntimeError) as error:
            if attempt == retries - 1:
                raise
            time.sleep(0.75 * (2**attempt))
    raise AssertionError("unreachable")


def chunk_records() -> list[dict[str, int]]:
    records = []
    for chunk_id, start_row in enumerate(range(0, TOKENIZER_ENTRIES, ROWS_PER_CHUNK)):
        end_row = min(TOKENIZER_ENTRIES, start_row + ROWS_PER_CHUNK)
        records.append(
            {
                "id": chunk_id,
                "startRow": start_row,
                "endRow": end_row,
                "fileOffset": start_row * ROW_BYTES,
                "sourceStart": ABSOLUTE_START + start_row * ROW_BYTES,
                "sourceEnd": ABSOLUTE_START + end_row * ROW_BYTES - 1,
            }
        )
    return records


def marker_path(chunk_id: int) -> Path:
    return STATE_DIR / f"{chunk_id:04d}.json"


def completed_chunk(record: dict[str, int]) -> bool:
    path = marker_path(record["id"])
    if not path.exists():
        return False
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
        return (
            marker["startRow"] == record["startRow"]
            and marker["endRow"] == record["endRow"]
            and marker["bytes"] == (record["endRow"] - record["startRow"]) * ROW_BYTES
        )
    except (OSError, ValueError, KeyError):
        return False


def verify_against_sample(path: Path) -> int:
    if not SAMPLE_MAP_PATH.exists() or not SAMPLE_VECTOR_PATH.exists():
        return 0
    sample_map = json.loads(SAMPLE_MAP_PATH.read_text(encoding="utf-8"))
    sample_binary = SAMPLE_VECTOR_PATH.read_bytes()
    checked = 0
    with path.open("rb") as full_file:
        for sample_index, token in enumerate(sample_map.get("tokens", [])):
            token_id = int(token["id"])
            full_file.seek(token_id * ROW_BYTES)
            actual = full_file.read(ROW_BYTES)
            expected = sample_binary[sample_index * ROW_BYTES : (sample_index + 1) * ROW_BYTES]
            if actual != expected:
                raise RuntimeError(f"Full row for token {token_id} differs from the verified sample")
            checked += 1
    return checked


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(checksum: str, sample_rows_checked: int) -> None:
    manifest = {
        "source": {
            "repository": REPO,
            "commit": COMMIT,
            "shard": SHARD,
            "url": SHARD_URL,
            "tensor": "model.llm.embed.weight",
            "absoluteStart": ABSOLUTE_START,
        },
        "rows": TOKENIZER_ENTRIES,
        "dimensions": DIMENSIONS,
        "dtype": "BF16",
        "rowBytes": ROW_BYTES,
        "bytes": EXPECTED_BYTES,
        "sha256": checksum,
        "sampleRowsChecked": sample_rows_checked,
        "excludedPaddingRows": 201_024 - TOKENIZER_ENTRIES,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size == EXPECTED_BYTES:
        print(f"Using complete file: {OUTPUT_PATH} ({EXPECTED_BYTES:,} bytes)")
        checked = verify_against_sample(OUTPUT_PATH)
        checksum = sha256_file(OUTPUT_PATH)
        write_manifest(checksum, checked)
        print(f"Verified {checked:,} known rows; SHA-256 {checksum}")
        return

    records = chunk_records()
    if not PARTIAL_PATH.exists() or PARTIAL_PATH.stat().st_size != EXPECTED_BYTES:
        if STATE_DIR.exists():
            shutil.rmtree(STATE_DIR)
        with PARTIAL_PATH.open("wb") as target:
            target.truncate(EXPECTED_BYTES)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    pending = [record for record in records if not completed_chunk(record)]
    print(
        f"Downloading {TOKENIZER_ENTRIES:,} rows ({EXPECTED_BYTES / 1e9:.3f} GB) "
        f"in {len(records):,} resumable ranges; {len(records) - len(pending):,} already complete."
    )

    descriptor = os.open(PARTIAL_PATH, os.O_RDWR)
    completed = len(records) - len(pending)
    started = time.monotonic()

    def download(record: dict[str, int]) -> tuple[dict[str, int], str]:
        payload = fetch_range(record["sourceStart"], record["sourceEnd"])
        os.pwrite(descriptor, payload, record["fileOffset"])
        return record, hashlib.sha256(payload).hexdigest()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = [executor.submit(download, record) for record in pending]
            for future in concurrent.futures.as_completed(futures):
                record, checksum = future.result()
                byte_count = (record["endRow"] - record["startRow"]) * ROW_BYTES
                marker = {
                    "startRow": record["startRow"],
                    "endRow": record["endRow"],
                    "bytes": byte_count,
                    "sha256": checksum,
                }
                marker_path(record["id"]).write_text(json.dumps(marker), encoding="utf-8")
                completed += 1
                elapsed = max(time.monotonic() - started, 0.001)
                downloaded = sum(
                    (item["endRow"] - item["startRow"]) * ROW_BYTES
                    for item in records
                    if completed_chunk(item)
                )
                speed = downloaded / elapsed / 1e6
                print(f"  {completed:>3}/{len(records)} ranges · {downloaded / 1e9:5.2f} GB · {speed:5.1f} MB/s")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    if not all(completed_chunk(record) for record in records):
        raise RuntimeError("Download ended with incomplete range markers")
    os.replace(PARTIAL_PATH, OUTPUT_PATH)
    checked = verify_against_sample(OUTPUT_PATH)
    checksum = sha256_file(OUTPUT_PATH)
    write_manifest(checksum, checked)
    shutil.rmtree(STATE_DIR)
    print(f"Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size:,} bytes)")
    print(f"Verified {checked:,} known rows; SHA-256 {checksum}")


if __name__ == "__main__":
    main()
