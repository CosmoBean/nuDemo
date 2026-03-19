from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nudemo.benchmarks.backends import (
    LanceBackend,
    MinioPostgresBackend,
    RedisBackend,
    WebDatasetBackend,
)
from nudemo.benchmarks.synthetic import SyntheticNuScenesDataset


class SyntheticFlowTests(unittest.TestCase):
    def test_default_dataset_matches_spec_shape(self) -> None:
        dataset = SyntheticNuScenesDataset().build()

        self.assertEqual(len(dataset), 404)
        self.assertEqual(len({sample.scene_name for sample in dataset}), 10)
        self.assertTrue(any(sample.location == "boston-seaport" for sample in dataset))
        self.assertGreater(dataset[0].payload_bytes(), 0)
        self.assertGreater(dataset[0].manifest_bytes(), 0)

    def test_backends_support_expected_flows(self) -> None:
        dataset = SyntheticNuScenesDataset(sample_count=24, scene_count=4).build()

        minio = MinioPostgresBackend()
        lance = LanceBackend()
        redis = RedisBackend()
        webdataset = WebDatasetBackend()

        for backend in (minio, lance, redis, webdataset):
            backend.load(dataset)

        curation_filter = SyntheticNuScenesDataset(sample_count=24, scene_count=4).curation_filter()
        minio_matches = minio.query_indices(curation_filter)
        lance_matches = lance.query_indices(curation_filter)

        self.assertEqual(minio_matches, lance_matches)
        self.assertTrue(minio_matches)
        self.assertGreater(minio.fetch_sample(minio_matches[0]).payload_bytes(), 0)
        self.assertGreater(redis.fetch_sample(0, payload=False).manifest_bytes(), 0)

        with self.assertRaises(NotImplementedError):
            webdataset.fetch_sample(0)


if __name__ == "__main__":
    unittest.main()
