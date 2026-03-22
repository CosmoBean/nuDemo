from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from nudemo.domain.models import CAMERAS, RADARS
from nudemo.storage.base import (
    StorageWriteResult,
    array_to_npy_bytes,
    directory_size,
    image_to_jpeg_bytes,
)


@dataclass(slots=True)
class ParquetBackend:
    dataset_path: Path
    name: str = "Parquet"

    def _part_files(self) -> list[Path]:
        if not self.dataset_path.exists():
            return []
        return sorted(self.dataset_path.glob("*.parquet"))

    def _dataset(self):
        import pyarrow.dataset as ds

        part_files = self._part_files()
        if not part_files:
            raise FileNotFoundError(f"no parquet parts found in {self.dataset_path}")
        return ds.dataset([str(path) for path in part_files], format="parquet")

    def _clear_dataset(self) -> None:
        if self.dataset_path.exists():
            shutil.rmtree(self.dataset_path)
        self.dataset_path.mkdir(parents=True, exist_ok=True)

    def write_samples(self, samples):
        import pyarrow as pa
        import pyarrow.parquet as pq

        self._clear_dataset()
        t0 = time.perf_counter()
        bytes_written = 0
        samples_written = 0
        rows = []
        part_idx = 0
        chunk_size = 64

        def flush_batch(batch_rows: list[dict[str, object]]) -> None:
            nonlocal part_idx
            if not batch_rows:
                return
            table = pa.Table.from_pylist(batch_rows)
            pq.write_table(table, self.dataset_path / f"part-{part_idx:05d}.parquet")
            part_idx += 1

        for sample_idx, sample in enumerate(samples):
            row = sample.to_dict(sample_idx)
            row["has_pedestrian"] = any(
                "pedestrian" in category for category in row["annotation_categories"]
            )
            for camera in CAMERAS:
                payload = image_to_jpeg_bytes(sample.cameras[camera])
                row[f"{camera}_bytes"] = payload
                bytes_written += len(payload)
            for sensor in ("LIDAR_TOP", *RADARS):
                data = sample.lidar_top if sensor == "LIDAR_TOP" else sample.radars[sensor]
                payload = array_to_npy_bytes(data)
                row[f"{sensor}_bytes"] = payload
                bytes_written += len(payload)
            rows.append(row)
            samples_written += 1
            if len(rows) >= chunk_size:
                flush_batch(rows)
                rows = []

        flush_batch(rows)
        elapsed = time.perf_counter() - t0
        return StorageWriteResult(
            backend=self.name,
            samples_written=samples_written,
            elapsed_sec=elapsed,
            bytes_written=bytes_written,
        )

    def sequential_iter(self):
        if not self._part_files():
            return
        columns = ["sample_idx", "CAM_FRONT_bytes", "LIDAR_TOP_bytes"]
        for batch in self._dataset().to_batches(columns=columns):
            for row_idx in range(batch.num_rows):
                yield {
                    "idx": batch.column("sample_idx")[row_idx].as_py(),
                    "cam": batch.column("CAM_FRONT_bytes")[row_idx].as_py(),
                    "lidar": batch.column("LIDAR_TOP_bytes")[row_idx].as_py(),
                }

    def fetch(self, sample_idx: int):
        import pyarrow.dataset as ds

        if not self._part_files():
            raise FileNotFoundError(f"no parquet parts found in {self.dataset_path}")
        table = self._dataset().to_table(
            columns=["CAM_FRONT_bytes", "LIDAR_TOP_bytes"],
            filter=ds.field("sample_idx") == sample_idx,
        )
        if table.num_rows == 0:
            raise IndexError(f"sample_idx {sample_idx} not found")
        return {
            "cam": table.column("CAM_FRONT_bytes")[0].as_py(),
            "lidar": table.column("LIDAR_TOP_bytes")[0].as_py(),
        }

    def curation_query(self):
        import pyarrow.dataset as ds

        if not self._part_files():
            return []
        table = self._dataset().to_table(
            columns=["sample_idx"],
            filter=(
                (ds.field("location") == "boston-seaport")
                & (ds.field("num_annotations") > 5)
                & ds.field("has_pedestrian")
            ),
        )
        table = table.sort_by("sample_idx")
        return [table.column("sample_idx")[row_idx].as_py() for row_idx in range(table.num_rows)]

    def disk_footprint(self) -> int:
        return directory_size(self.dataset_path)
