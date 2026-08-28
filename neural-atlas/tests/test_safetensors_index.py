from __future__ import annotations

import struct

import numpy as np
import pytest
from safetensors.numpy import save_file

from neural_atlas.safetensors_index import SafetensorsIndex, TileTooLargeError


def build_checkpoint(tmp_path):
    matrix = np.arange(-10, 10, dtype=np.float32).reshape(4, 5)
    volume = (np.arange(24, dtype=np.float32) / 8).reshape(2, 3, 4)
    vector = np.array([7, -3, 11], dtype=np.int32)
    checkpoint = tmp_path / "model.safetensors"
    save_file({"matrix.weight": matrix, "volume.weight": volume, "vector": vector}, checkpoint)
    return checkpoint, matrix, volume, vector


def test_manifest_and_search_are_header_derived(tmp_path):
    checkpoint, matrix, volume, vector = build_checkpoint(tmp_path)
    index = SafetensorsIndex.from_path(checkpoint)

    manifest = index.manifest()
    assert manifest["tensorCount"] == 3
    assert manifest["scalarCount"] == matrix.size + volume.size + vector.size
    assert manifest["fileCount"] == 1
    assert manifest["exact"] is True

    result = index.search("weight")
    assert result["total"] == 2
    assert [entry["name"] for entry in result["items"]] == ["matrix.weight", "volume.weight"]


def test_higher_rank_matrix_coordinates_preserve_original_indices(tmp_path):
    checkpoint, _, volume, _ = build_checkpoint(tmp_path)
    index = SafetensorsIndex.from_path(checkpoint)
    description = index.describe("volume.weight")

    assert description["matrixRows"] == 6
    assert description["matrixColumns"] == 4

    scalar = index.read_matrix_scalar("volume.weight", 4, 2)
    assert scalar["indices"] == [1, 1, 2]
    assert scalar["value"] == pytest.approx(float(volume[1, 1, 2]))


def test_scalar_reports_literal_bytes_bits_and_absolute_offset(tmp_path):
    checkpoint, matrix, _, _ = build_checkpoint(tmp_path)
    index = SafetensorsIndex.from_path(checkpoint)
    record = index.get("matrix.weight")

    scalar = index.read_scalar("matrix.weight", [2, 3])
    expected_value = float(matrix[2, 3])
    expected_raw = struct.pack("<f", expected_value)
    expected_flat = 2 * matrix.shape[1] + 3

    assert scalar["value"] == pytest.approx(expected_value)
    assert scalar["flatIndex"] == expected_flat
    assert scalar["rawHex"] == expected_raw.hex()
    assert scalar["rawBits"] == format(int.from_bytes(expected_raw, "little"), "032b")
    assert scalar["absoluteByteOffset"] == record.data_start + expected_flat * 4
    assert scalar["byteRange"] == [scalar["absoluteByteOffset"], scalar["absoluteByteOffset"] + 3]
    assert scalar["exact"] is True


def test_tile_statistics_scan_the_exact_population(tmp_path):
    checkpoint, matrix, _, _ = build_checkpoint(tmp_path)
    index = SafetensorsIndex.from_path(checkpoint)

    tile = index.tile(
        "matrix.weight",
        row_start=0,
        column_start=0,
        row_count=4,
        column_count=5,
        target_rows=2,
        target_columns=1,
    )

    first = tile["cells"][0][0]
    second = tile["cells"][1][0]
    assert first["count"] == 10
    assert second["count"] == 10
    assert first["mean"] == pytest.approx(float(matrix[:2].mean()))
    assert first["minimum"] == pytest.approx(float(matrix[:2].min()))
    assert first["maximum"] == pytest.approx(float(matrix[:2].max()))
    assert first["standardDeviation"] == pytest.approx(float(matrix[:2].std()))
    assert second["absoluteMean"] == pytest.approx(float(np.abs(matrix[2:]).mean()))
    assert tile["sourceValues"] == matrix.size
    assert tile["exact"] is True


def test_oversized_tile_is_rejected_instead_of_sampled(tmp_path):
    checkpoint, _, _, _ = build_checkpoint(tmp_path)
    index = SafetensorsIndex.from_path(checkpoint)

    with pytest.raises(TileTooLargeError):
        index.tile("matrix.weight", max_source_values=19)
