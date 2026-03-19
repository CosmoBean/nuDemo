from pathlib import Path

from nudemo.benchmarks.runner import (
    BenchmarkRecord,
    BenchmarkRunner,
    build_disk_record,
    build_write_record,
    export_records,
)
from nudemo.storage.base import StorageWriteResult


class FakeBackend:
    name = "fake"

    def sequential_iter(self):
        yield from [{"idx": 0}, {"idx": 1}, {"idx": 2}]

    def fetch(self, sample_idx: int):
        return {"idx": sample_idx}

    def curation_query(self):
        return [0, 2]

    def disk_footprint(self):
        return 1024


def test_runner_emits_expected_patterns():
    runner = BenchmarkRunner(backends={"fake": FakeBackend()})
    records = runner.run_storage_suite(random_indices=[0, 2])
    patterns = {record.pattern for record in records}

    assert "sequential_scan" in patterns
    assert "random_access" in patterns
    assert "curation_query" in patterns
    assert "e2e_curation" in patterns
    assert "disk_footprint" in patterns


def test_export_records_writes_json_and_csv(tmp_path: Path):
    records = [
        BenchmarkRecord(backend="fake", pattern="sequential_scan", metrics={"throughput_mean": 1.0}),
        build_write_record(
            StorageWriteResult(
                backend="fake",
                samples_written=3,
                elapsed_sec=2.0,
                bytes_written=30,
            )
        ),
        build_disk_record("fake", 1024),
    ]
    json_path, csv_path = export_records(records, tmp_path)

    assert json_path.exists()
    assert csv_path.exists()
    assert "throughput_mean" in json_path.read_text(encoding="utf-8")
