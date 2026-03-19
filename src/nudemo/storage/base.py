from __future__ import annotations

import io
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from nudemo.domain.models import UnifiedSample


@dataclass(slots=True)
class StorageWriteResult:
    backend: str
    samples_written: int
    elapsed_sec: float
    bytes_written: int

    @property
    def throughput(self) -> float:
        return self.samples_written / self.elapsed_sec if self.elapsed_sec else 0.0


class StorageBackend(Protocol):
    name: str

    def write_samples(self, samples: Iterable[UnifiedSample]) -> StorageWriteResult:
        ...

    def sequential_iter(self) -> Iterator[dict[str, bytes | int | str]]:
        ...

    def fetch(self, sample_idx: int) -> dict[str, bytes | int | str]:
        ...

    def curation_query(self) -> list[int]:
        ...

    def disk_footprint(self) -> int:
        ...


def image_to_jpeg_bytes(image: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def array_to_npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for file_name in files:
            total += (Path(root) / file_name).stat().st_size
    return total

