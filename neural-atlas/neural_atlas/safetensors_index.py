from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


class SafetensorsIndexError(RuntimeError):
    """Base error for checkpoint indexing and exact-value reads."""


class UnsupportedDtypeError(SafetensorsIndexError):
    """Raised when a tensor can be indexed but not numerically decoded."""


class TileTooLargeError(SafetensorsIndexError):
    """Raised rather than silently sampling an oversized tensor region."""


_ITEM_SIZES: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}

_NUMPY_DTYPES: dict[str, str] = {
    "BOOL": "|u1",
    "U8": "|u1",
    "I8": "|i1",
    "I16": "<i2",
    "U16": "<u2",
    "F16": "<f2",
    "I32": "<i4",
    "U32": "<u4",
    "F32": "<f4",
    "I64": "<i8",
    "U64": "<u8",
    "F64": "<f8",
}


def _product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _json_number(value: Any) -> int | float | bool:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    number = float(value)
    if not math.isfinite(number):
        raise SafetensorsIndexError("Checkpoint value is not finite")
    return number


def _decode_values(dtype: str, payload: bytes) -> np.ndarray:
    if dtype == "BF16":
        words = np.frombuffer(payload, dtype="<u2")
        bits = np.left_shift(words.astype(np.uint32), 16)
        return bits.view(np.float32)
    numpy_dtype = _NUMPY_DTYPES.get(dtype)
    if numpy_dtype is None:
        raise UnsupportedDtypeError(f"Numerical decoding is not implemented for safetensors dtype {dtype}")
    values = np.frombuffer(payload, dtype=numpy_dtype)
    if dtype == "BOOL":
        return values.astype(np.bool_)
    return values


@dataclass(frozen=True, slots=True)
class TensorRecord:
    name: str
    file: Path
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int
    header_length: int

    @property
    def itemsize(self) -> int:
        try:
            return _ITEM_SIZES[self.dtype]
        except KeyError as exc:
            raise UnsupportedDtypeError(f"Unknown safetensors dtype {self.dtype}") from exc

    @property
    def numel(self) -> int:
        return _product(self.shape)

    @property
    def nbytes(self) -> int:
        return self.data_end - self.data_start

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def matrix_rows(self) -> int:
        if self.rank <= 1:
            return 1
        return _product(self.shape[:-1])

    @property
    def matrix_columns(self) -> int:
        if self.rank == 0:
            return 1
        return self.shape[-1]

    def matrix_to_indices(self, row: int, column: int) -> tuple[int, ...]:
        if row < 0 or row >= self.matrix_rows:
            raise IndexError(f"Matrix row {row} is outside 0..{self.matrix_rows - 1}")
        if column < 0 or column >= self.matrix_columns:
            raise IndexError(f"Matrix column {column} is outside 0..{self.matrix_columns - 1}")
        if self.rank == 0:
            return ()
        if self.rank == 1:
            return (column,)
        leading = np.unravel_index(row, self.shape[:-1], order="C")
        return tuple(int(value) for value in leading) + (column,)

    def indices_to_flat(self, indices: Sequence[int]) -> int:
        if len(indices) != self.rank:
            raise IndexError(f"Tensor {self.name} has rank {self.rank}, not {len(indices)}")
        if self.rank == 0:
            return 0
        flat = 0
        for index, dimension in zip(indices, self.shape):
            if index < 0 or index >= dimension:
                raise IndexError(f"Index {index} is outside tensor dimension {dimension}")
            flat = flat * dimension + int(index)
        return flat


class SafetensorsIndex:
    """Index safetensors headers and expose exact range-addressable values.

    Higher-rank tensors are presented as a matrix by flattening all leading
    dimensions into rows and preserving the final dimension as columns. Every
    matrix coordinate maps deterministically back to the original tensor index.
    """

    def __init__(self, root: Path, records: Iterable[TensorRecord]) -> None:
        self.root = Path(root).resolve()
        ordered = sorted(records, key=lambda record: record.name)
        self._records = {record.name: record for record in ordered}
        if len(self._records) != len(ordered):
            raise SafetensorsIndexError("Duplicate tensor names exist across checkpoint shards")

    @classmethod
    def from_path(cls, path: str | Path) -> "SafetensorsIndex":
        root = Path(path).expanduser().resolve()
        if root.is_file():
            files = [root]
            index_root = root.parent
        else:
            files = sorted(root.rglob("*.safetensors"))
            index_root = root
        if not files:
            raise FileNotFoundError(f"No .safetensors files found under {root}")
        records: list[TensorRecord] = []
        for file in files:
            records.extend(cls._read_header(file))
        return cls(index_root, records)

    @staticmethod
    def _read_header(file: Path) -> list[TensorRecord]:
        with file.open("rb") as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise SafetensorsIndexError(f"{file} does not contain an 8-byte safetensors header prefix")
            header_length = struct.unpack("<Q", prefix)[0]
            if header_length <= 0 or header_length > file.stat().st_size - 8:
                raise SafetensorsIndexError(f"Invalid safetensors header length in {file}")
            raw_header = source.read(header_length)
        try:
            header = json.loads(raw_header)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SafetensorsIndexError(f"Invalid safetensors JSON header in {file}") from exc
        payload_start = 8 + header_length
        records: list[TensorRecord] = []
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(entry, dict):
                raise SafetensorsIndexError(f"Tensor header {name!r} in {file} is not an object")
            dtype = str(entry["dtype"])
            shape = tuple(int(value) for value in entry["shape"])
            begin, end = (int(value) for value in entry["data_offsets"])
            if any(dimension < 0 for dimension in shape) or begin < 0 or end < begin:
                raise SafetensorsIndexError(f"Invalid shape or offsets for tensor {name!r}")
            record = TensorRecord(
                name=name,
                file=file.resolve(),
                dtype=dtype,
                shape=shape,
                data_start=payload_start + begin,
                data_end=payload_start + end,
                header_length=header_length,
            )
            expected = record.numel * record.itemsize
            if expected != record.nbytes:
                raise SafetensorsIndexError(
                    f"Tensor {name!r} contains {record.nbytes} bytes, expected {expected} from {shape} {dtype}"
                )
            if record.data_end > file.stat().st_size:
                raise SafetensorsIndexError(f"Tensor {name!r} extends past the end of {file}")
            records.append(record)
        return records

    @property
    def tensor_count(self) -> int:
        return len(self._records)

    @property
    def scalar_count(self) -> int:
        return sum(record.numel for record in self._records.values())

    @property
    def tensor_bytes(self) -> int:
        return sum(record.nbytes for record in self._records.values())

    def names(self) -> list[str]:
        return list(self._records)

    def get(self, name: str) -> TensorRecord:
        try:
            return self._records[name]
        except KeyError as exc:
            raise KeyError(f"Unknown checkpoint tensor {name!r}") from exc

    def _relative_file(self, file: Path) -> str:
        try:
            return str(file.relative_to(self.root))
        except ValueError:
            return str(file)

    def describe(self, name: str) -> dict[str, Any]:
        record = self.get(name)
        return {
            "name": record.name,
            "shape": list(record.shape),
            "rank": record.rank,
            "dtype": record.dtype,
            "itemBytes": record.itemsize,
            "scalarCount": record.numel,
            "tensorBytes": record.nbytes,
            "sourceFile": self._relative_file(record.file),
            "absoluteDataStart": record.data_start,
            "absoluteDataEndExclusive": record.data_end,
            "matrixRows": record.matrix_rows,
            "matrixColumns": record.matrix_columns,
            "matrixRule": "All leading dimensions are flattened into rows; the final tensor dimension is the matrix column.",
        }

    def manifest(self) -> dict[str, Any]:
        files = sorted({self._relative_file(record.file) for record in self._records.values()})
        dtypes: dict[str, int] = {}
        for record in self._records.values():
            dtypes[record.dtype] = dtypes.get(record.dtype, 0) + record.numel
        return {
            "root": str(self.root),
            "tensorCount": self.tensor_count,
            "scalarCount": self.scalar_count,
            "tensorBytes": self.tensor_bytes,
            "files": files,
            "fileCount": len(files),
            "dtypeScalars": dtypes,
            "exact": True,
        }

    def search(self, query: str = "", offset: int = 0, limit: int = 100) -> dict[str, Any]:
        normalized = query.strip().lower()
        matches = [
            record
            for record in self._records.values()
            if not normalized or normalized in record.name.lower()
        ]
        offset = max(0, int(offset))
        limit = min(500, max(1, int(limit)))
        page = matches[offset : offset + limit]
        return {
            "query": query,
            "offset": offset,
            "limit": limit,
            "total": len(matches),
            "items": [self.describe(record.name) for record in page],
        }

    def _read_flat(self, record: TensorRecord, flat_start: int, count: int) -> np.ndarray:
        if flat_start < 0 or count < 0 or flat_start + count > record.numel:
            raise IndexError(
                f"Flat range {flat_start}:{flat_start + count} is outside tensor {record.name} with {record.numel} values"
            )
        byte_count = count * record.itemsize
        with record.file.open("rb") as source:
            source.seek(record.data_start + flat_start * record.itemsize)
            payload = source.read(byte_count)
        if len(payload) != byte_count:
            raise SafetensorsIndexError(
                f"Read {len(payload)} bytes for {record.name}; expected {byte_count}"
            )
        return _decode_values(record.dtype, payload)

    def read_scalar(self, name: str, indices: Sequence[int]) -> dict[str, Any]:
        record = self.get(name)
        flat_index = record.indices_to_flat(indices)
        byte_offset = record.data_start + flat_index * record.itemsize
        with record.file.open("rb") as source:
            source.seek(byte_offset)
            raw = source.read(record.itemsize)
        if len(raw) != record.itemsize:
            raise SafetensorsIndexError(f"Could not read scalar {indices} from {record.name}")
        value = _json_number(_decode_values(record.dtype, raw)[0])
        matrix_row = flat_index // record.matrix_columns
        matrix_column = flat_index % record.matrix_columns
        return {
            "tensor": record.name,
            "indices": [int(value) for value in indices],
            "matrixRow": matrix_row,
            "matrixColumn": matrix_column,
            "flatIndex": flat_index,
            "value": value,
            "dtype": record.dtype,
            "itemBytes": record.itemsize,
            "rawHex": raw.hex(),
            "rawBits": format(int.from_bytes(raw, byteorder="little", signed=False), f"0{record.itemsize * 8}b"),
            "sourceFile": self._relative_file(record.file),
            "absoluteByteOffset": byte_offset,
            "byteRange": [byte_offset, byte_offset + record.itemsize - 1],
            "exact": True,
        }

    def read_matrix_scalar(self, name: str, row: int, column: int) -> dict[str, Any]:
        record = self.get(name)
        return self.read_scalar(name, record.matrix_to_indices(int(row), int(column)))

    def tile(
        self,
        name: str,
        *,
        row_start: int = 0,
        column_start: int = 0,
        row_count: int | None = None,
        column_count: int | None = None,
        target_rows: int = 64,
        target_columns: int = 64,
        max_source_values: int = 10_000_000,
    ) -> dict[str, Any]:
        record = self.get(name)
        matrix_rows = record.matrix_rows
        matrix_columns = record.matrix_columns
        row_start = int(row_start)
        column_start = int(column_start)
        row_count = matrix_rows - row_start if row_count is None else int(row_count)
        column_count = matrix_columns - column_start if column_count is None else int(column_count)
        if row_start < 0 or column_start < 0 or row_count <= 0 or column_count <= 0:
            raise IndexError("Tile origin and dimensions must describe a positive matrix region")
        if row_start + row_count > matrix_rows or column_start + column_count > matrix_columns:
            raise IndexError(
                f"Tile region exceeds {matrix_rows} × {matrix_columns} matrix bounds"
            )
        source_values = row_count * column_count
        if source_values > max_source_values:
            raise TileTooLargeError(
                f"Exact tile would scan {source_values:,} values; limit is {max_source_values:,}. Zoom into a smaller region."
            )
        output_rows = min(row_count, max(1, min(256, int(target_rows))))
        output_columns = min(column_count, max(1, min(256, int(target_columns))))
        row_edges = np.linspace(0, row_count, output_rows + 1, dtype=np.int64)
        column_edges = np.linspace(0, column_count, output_columns + 1, dtype=np.int64)

        region = np.empty((row_count, column_count), dtype=np.float64)
        for local_row in range(row_count):
            matrix_row = row_start + local_row
            flat_start = matrix_row * matrix_columns + column_start
            region[local_row, :] = self._read_flat(record, flat_start, column_count).astype(np.float64)

        cells: list[list[dict[str, Any]]] = []
        for output_row in range(output_rows):
            source_row_start = int(row_edges[output_row])
            source_row_end = int(row_edges[output_row + 1])
            cell_row: list[dict[str, Any]] = []
            for output_column in range(output_columns):
                source_column_start = int(column_edges[output_column])
                source_column_end = int(column_edges[output_column + 1])
                block = region[
                    source_row_start:source_row_end,
                    source_column_start:source_column_end,
                ]
                count = int(block.size)
                cell_row.append(
                    {
                        "rowStart": row_start + source_row_start,
                        "rowCount": source_row_end - source_row_start,
                        "columnStart": column_start + source_column_start,
                        "columnCount": source_column_end - source_column_start,
                        "count": count,
                        "minimum": float(np.min(block)),
                        "maximum": float(np.max(block)),
                        "mean": float(np.mean(block)),
                        "standardDeviation": float(np.std(block)),
                        "absoluteMean": float(np.mean(np.abs(block))),
                        "isScalar": count == 1,
                    }
                )
            cells.append(cell_row)

        return {
            "tensor": self.describe(name),
            "region": {
                "rowStart": row_start,
                "rowCount": row_count,
                "columnStart": column_start,
                "columnCount": column_count,
            },
            "outputRows": output_rows,
            "outputColumns": output_columns,
            "sourceValues": source_values,
            "aggregation": "Exact population minimum, maximum, mean, population standard deviation, and absolute mean.",
            "exact": True,
            "cells": cells,
        }
