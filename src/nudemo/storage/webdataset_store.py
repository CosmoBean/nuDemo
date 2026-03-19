from __future__ import annotations

import json
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
class WebDatasetBackend:
    shard_pattern: str
    maxcount: int
    name: str = "WebDataset"

    def _root_dir(self) -> Path:
        parts = self.shard_pattern.split("%", 1)[0]
        return Path(parts).parent

    def write_samples(self, samples):
        import webdataset as wds

        root = self._root_dir()
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        bytes_written = 0
        count = 0
        with wds.ShardWriter(self.shard_pattern, maxcount=self.maxcount) as sink:
            for sample_idx, sample in enumerate(samples):
                record = {"__key__": f"sample_{sample_idx:04d}"}
                for camera in CAMERAS:
                    payload = image_to_jpeg_bytes(sample.cameras[camera])
                    record[f"{camera}.jpg"] = payload
                    bytes_written += len(payload)
                for sensor in ("LIDAR_TOP", *RADARS):
                    data = sample.lidar_top if sensor == "LIDAR_TOP" else sample.radars[sensor]
                    payload = array_to_npy_bytes(data)
                    record[f"{sensor}.npy"] = payload
                    bytes_written += len(payload)
                metadata = sample.to_dict(sample_idx)
                record["metadata.json"] = json.dumps(metadata, sort_keys=True).encode("utf-8")
                sink.write(record)
                count += 1
        elapsed = time.perf_counter() - t0
        return StorageWriteResult(
            backend=self.name,
            samples_written=count,
            elapsed_sec=elapsed,
            bytes_written=bytes_written,
        )

    def sequential_iter(self):
        import webdataset as wds

        shard_paths = [str(path) for path in sorted(self._root_dir().glob("*.tar"))]
        if not shard_paths:
            return
        for sample in wds.WebDataset(shard_paths, shardshuffle=False):
            cam = sample.get("cam_front.jpg", sample.get("CAM_FRONT.jpg"))
            lidar = sample.get("lidar_top.npy", sample.get("LIDAR_TOP.npy"))
            yield {
                "key": sample["__key__"],
                "cam": cam,
                "lidar": lidar,
            }

    def fetch(self, sample_idx: int):
        raise NotImplementedError("WebDataset is sequential only")

    def curation_query(self):
        raise NotImplementedError("WebDataset does not support metadata queries")

    def disk_footprint(self) -> int:
        return directory_size(self._root_dir())
