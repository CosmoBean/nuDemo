from __future__ import annotations

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
class LanceBackend:
    dataset_path: Path
    name: str = "Lance"

    def _dataset(self):
        import lance

        return lance.dataset(str(self.dataset_path))

    def write_samples(self, samples):
        import lance
        import pyarrow as pa

        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        bytes_written = 0
        rows = []
        samples_written = 0
        mode = "overwrite"
        for sample_idx, sample in enumerate(samples):
            row = sample.to_dict(sample_idx)
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
            if len(rows) >= 16:
                table = pa.Table.from_pylist(rows)
                lance.write_dataset(table, str(self.dataset_path), mode=mode)
                mode = "append"
                rows = []

        if rows:
            table = pa.Table.from_pylist(rows)
            lance.write_dataset(table, str(self.dataset_path), mode=mode)
        elapsed = time.perf_counter() - t0
        return StorageWriteResult(
            backend=self.name,
            samples_written=samples_written,
            elapsed_sec=elapsed,
            bytes_written=bytes_written,
        )

    def sequential_iter(self):
        dataset = self._dataset()
        columns = ["sample_idx", "CAM_FRONT_bytes", "LIDAR_TOP_bytes"]
        for batch in dataset.to_batches(columns=columns):
            for row_idx in range(batch.num_rows):
                yield {
                    "idx": batch.column("sample_idx")[row_idx].as_py(),
                    "cam": batch.column("CAM_FRONT_bytes")[row_idx].as_py(),
                    "lidar": batch.column("LIDAR_TOP_bytes")[row_idx].as_py(),
                }

    def fetch(self, sample_idx: int):
        row = self._dataset().take(
            [sample_idx],
            columns=["CAM_FRONT_bytes", "LIDAR_TOP_bytes"],
        ).to_pydict()
        return {"cam": row["CAM_FRONT_bytes"][0], "lidar": row["LIDAR_TOP_bytes"][0]}

    def curation_query(self):
        dataset = self._dataset()
        scanner = dataset.scanner(
            columns=["sample_idx", "location", "num_annotations", "annotation_categories"],
            filter="location = 'boston-seaport' AND num_annotations > 5",
        )
        table = scanner.to_table()
        results = []
        for row_idx in range(table.num_rows):
            categories = table.column("annotation_categories")[row_idx].as_py() or []
            if any("pedestrian" in category for category in categories):
                results.append(table.column("sample_idx")[row_idx].as_py())
        return results

    def disk_footprint(self) -> int:
        return directory_size(self.dataset_path)
