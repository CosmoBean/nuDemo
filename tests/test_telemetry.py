from __future__ import annotations

from datetime import UTC, datetime

from nudemo.observability.metrics import build_service_measurements
from nudemo.telemetry.dashboard import build_telemetry_dashboard_html
from nudemo.telemetry.docker import (
    parse_byte_size,
    parse_compose_services,
    parse_stats_lines,
)


def test_parse_byte_size_handles_binary_and_decimal_units() -> None:
    assert parse_byte_size("496MiB") == 520093696
    assert parse_byte_size("92.2MB") == 92200000
    assert parse_byte_size("0B") == 0


def test_parse_compose_services_and_stats_lines() -> None:
    services = parse_compose_services(
        '\n'.join(
            [
                '{"Service":"kafka","Name":"config-kafka-1"}',
                '{"Service":"redis","Name":"config-redis-1"}',
            ]
        )
    )
    assert [service.service for service in services] == ["kafka", "redis"]

    snapshots = parse_stats_lines(
        '\n'.join(
            [
                (
                    '{"BlockIO":"0B / 100MB","CPUPerc":"0.79%","Container":"config-kafka-1",'
                    '"MemPerc":"1.78%","MemUsage":"496MiB / 27.26GiB",'
                    '"Name":"config-kafka-1","NetIO":"92.2MB / 134MB","PIDs":"87"}'
                ),
                (
                    '{"BlockIO":"0B / 139kB","CPUPerc":"0.57%","Container":"config-redis-1",'
                    '"MemPerc":"0.01%","MemUsage":"4.125MiB / 27.26GiB",'
                    '"Name":"config-redis-1","NetIO":"1.98MB / 1.09MB","PIDs":"6"}'
                ),
            ]
        ),
        snapshot_label="run_start",
        service_lookup={"config-kafka-1": "kafka", "config-redis-1": "redis"},
        observed_at=datetime(2026, 3, 19, tzinfo=UTC),
    )
    assert len(snapshots) == 2
    assert snapshots[0].service == "kafka"
    assert snapshots[0].mem_usage_bytes == 520093696
    assert snapshots[1].net_output_bytes == 1090000


def test_build_telemetry_dashboard_html_contains_observability_sections() -> None:
    run = {
        "run_id": "abc123",
        "suite_name": "nuDemo live benchmark suite",
        "provider": "real",
        "status": "ok",
        "elapsed_sec": 42.5,
        "dataset": {"samples": 32},
        "summary": {"result_count": 8, "error_count": 0},
        "report_path": "artifacts/reports/benchmark_report.json",
        "json_path": "artifacts/reports/benchmark_results.json",
        "csv_path": "artifacts/reports/benchmark_results.csv",
        "dashboard_path": "artifacts/reports/benchmark_dashboard.html",
        "telemetry_dashboard_path": "artifacts/reports/telemetry_dashboard.html",
    }
    spans = [
        {
            "stage": "ingestion",
            "backend": "Kafka",
            "pattern": "kafka_full_payload_consume",
            "status": "ok",
            "started_at": "2026-03-19T21:00:00+00:00",
            "elapsed_sec": 4.7,
            "sample_count": 32,
            "metrics": {"throughput_msg_sec": 6.8},
            "error": None,
        },
        {
            "stage": "storage",
            "backend": "Redis",
            "pattern": "write_throughput",
            "status": "ok",
            "started_at": "2026-03-19T21:00:05+00:00",
            "elapsed_sec": 1.6,
            "sample_count": 32,
            "metrics": {"throughput_samples_per_sec": 19.8},
            "error": None,
        },
    ]
    snapshots = [
        {
            "snapshot_label": "run_start",
            "service": "kafka",
            "observed_at": "2026-03-19T21:00:00+00:00",
            "cpu_percent": 0.79,
            "mem_percent": 1.78,
            "mem_usage_bytes": 520093696,
            "net_input_bytes": 92200000,
            "net_output_bytes": 134000000,
            "pids": 87,
        }
    ]

    html = build_telemetry_dashboard_html(run, spans, snapshots)

    assert "Telemetry Dashboard" in html
    assert "Top Bottlenecks" in html
    assert "Service Peaks" in html
    assert "Span Timeline" in html
    assert "Kafka" in html


def test_build_service_measurements_aggregates_latest_and_peak_values() -> None:
    measurements = build_service_measurements(
        "run-1",
        [
            {
                "service": "kafka",
                "snapshot_label": "periodic",
                "observed_at": "2026-03-19T21:00:00+00:00",
                "cpu_percent": 0.5,
                "mem_percent": 1.0,
                "mem_usage_bytes": 100,
                "mem_limit_bytes": 200,
                "net_input_bytes": 10,
                "net_output_bytes": 20,
                "block_input_bytes": 1,
                "block_output_bytes": 2,
                "pids": 5,
            },
            {
                "service": "kafka",
                "snapshot_label": "periodic",
                "observed_at": "2026-03-19T21:00:10+00:00",
                "cpu_percent": 0.9,
                "mem_percent": 1.5,
                "mem_usage_bytes": 150,
                "mem_limit_bytes": 200,
                "net_input_bytes": 25,
                "net_output_bytes": 30,
                "block_input_bytes": 3,
                "block_output_bytes": 4,
                "pids": 6,
            },
        ],
    )

    packed = {attrs["metric_name"]: value for value, attrs in measurements}
    assert packed["latest_cpu_percent"] == 0.9
    assert packed["latest_mem_usage_bytes"] == 150.0
    assert packed["peak_cpu_percent"] == 0.9
    assert packed["peak_net_input_bytes"] == 25.0
    assert packed["snapshot_count"] == 2.0
