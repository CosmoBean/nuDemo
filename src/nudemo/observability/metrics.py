from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request, Response

from nudemo.config import PostgresSettings
from nudemo.telemetry.store import fetch_run_bundle

_START_LOCK = threading.Lock()
_EXPORTER_STARTED = False
_METER = None


@dataclass(slots=True)
class CachedTelemetryBundle:
    run: dict[str, Any]
    spans: list[dict[str, Any]]
    snapshots: list[dict[str, Any]]
    loaded_at: float


class TelemetrySnapshotCache:
    def __init__(self, settings: PostgresSettings, ttl_seconds: float = 5.0) -> None:
        self._settings = settings
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._bundle: CachedTelemetryBundle | None = None

    def latest(self) -> CachedTelemetryBundle | None:
        now = time.monotonic()
        with self._lock:
            if self._bundle and (now - self._bundle.loaded_at) < self._ttl_seconds:
                return self._bundle
            try:
                run, spans, snapshots = fetch_run_bundle(self._settings)
            except Exception:
                return self._bundle
            self._bundle = CachedTelemetryBundle(
                run=run,
                spans=spans,
                snapshots=snapshots,
                loaded_at=now,
            )
            return self._bundle


def build_run_measurements(run: dict[str, Any]) -> list[tuple[float, dict[str, str]]]:
    run_id = str(run.get("run_id", "unknown"))
    base = {
        "run_id": run_id,
        "provider": str(run.get("provider", "unknown")),
        "status": str(run.get("status", "unknown")),
        "simulate": str(bool(run.get("simulate", False))).lower(),
    }
    dataset = run.get("dataset") or {}
    summary = run.get("summary") or {}
    values = {
        "elapsed_sec": run.get("elapsed_sec"),
        "sample_limit": run.get("sample_limit"),
        "dataset_samples": dataset.get("samples"),
        "dataset_scenes": dataset.get("scenes"),
        "summary_result_count": summary.get("result_count"),
        "summary_ok_count": summary.get("ok_count"),
        "summary_error_count": summary.get("error_count"),
    }
    return _pack_numeric_values(base, values)


def build_span_measurements(
    run_id: str,
    spans: list[dict[str, Any]],
) -> list[tuple[float, dict[str, str]]]:
    measurements: list[tuple[float, dict[str, str]]] = []
    for span in spans:
        attrs = {
            "run_id": run_id,
            "stage": str(span.get("stage", "unknown")),
            "backend": str(span.get("backend", "unknown")),
            "pattern": str(span.get("pattern", "unknown")),
            "status": str(span.get("status", "unknown")),
        }
        values = {
            "elapsed_sec": span.get("elapsed_sec"),
            "sample_count": span.get("sample_count"),
        }
        span_metrics = span.get("metrics") or {}
        for key, value in span_metrics.items():
            values[str(key)] = value
        measurements.extend(_pack_numeric_values(attrs, values))
    return measurements


def build_service_measurements(
    run_id: str,
    snapshots: list[dict[str, Any]],
) -> list[tuple[float, dict[str, str]]]:
    measurements: list[tuple[float, dict[str, str]]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for snapshot in snapshots:
        grouped.setdefault(str(snapshot.get("service", "unknown")), []).append(snapshot)
    for service, service_snapshots in grouped.items():
        latest_snapshot = max(
            service_snapshots,
            key=lambda snapshot: str(snapshot.get("observed_at", "")),
        )
        latest_values = {
            "latest_cpu_percent": latest_snapshot.get("cpu_percent"),
            "latest_mem_percent": latest_snapshot.get("mem_percent"),
            "latest_mem_usage_bytes": latest_snapshot.get("mem_usage_bytes"),
            "latest_mem_limit_bytes": latest_snapshot.get("mem_limit_bytes"),
            "latest_net_input_bytes": latest_snapshot.get("net_input_bytes"),
            "latest_net_output_bytes": latest_snapshot.get("net_output_bytes"),
            "latest_block_input_bytes": latest_snapshot.get("block_input_bytes"),
            "latest_block_output_bytes": latest_snapshot.get("block_output_bytes"),
            "latest_pids": latest_snapshot.get("pids"),
            "snapshot_count": len(service_snapshots),
        }
        peak_values = {
            "peak_cpu_percent": max(
                float(row.get("cpu_percent") or 0.0) for row in service_snapshots
            ),
            "peak_mem_percent": max(
                float(row.get("mem_percent") or 0.0) for row in service_snapshots
            ),
            "peak_mem_usage_bytes": max(
                int(row.get("mem_usage_bytes") or 0) for row in service_snapshots
            ),
            "peak_net_input_bytes": max(
                int(row.get("net_input_bytes") or 0) for row in service_snapshots
            ),
            "peak_net_output_bytes": max(
                int(row.get("net_output_bytes") or 0) for row in service_snapshots
            ),
            "peak_block_input_bytes": max(
                int(row.get("block_input_bytes") or 0) for row in service_snapshots
            ),
            "peak_block_output_bytes": max(
                int(row.get("block_output_bytes") or 0) for row in service_snapshots
            ),
            "peak_pids": max(int(row.get("pids") or 0) for row in service_snapshots),
        }
        measurements.extend(
            _pack_numeric_values(
                {"run_id": run_id, "service": service},
                {**latest_values, **peak_values},
            )
        )
    return measurements


def ensure_metrics_exporter(settings: PostgresSettings) -> None:
    if not _metrics_enabled():
        return

    global _EXPORTER_STARTED, _METER
    with _START_LOCK:
        if _EXPORTER_STARTED:
            return

        from opentelemetry import metrics
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
        from opentelemetry.metrics import Observation
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from prometheus_client import start_http_server

        host = os.getenv("NUDEMO_METRICS_HOST", "0.0.0.0")
        port = int(os.getenv("NUDEMO_METRICS_PORT", "9464"))

        reader = PrometheusMetricReader()
        provider = MeterProvider(
            resource=Resource.create({SERVICE_NAME: "nudemo-browser"}),
            metric_readers=[reader],
        )
        metrics.set_meter_provider(provider)
        start_http_server(port=port, addr=host)

        cache = TelemetrySnapshotCache(settings)
        meter = metrics.get_meter("nudemo.observability", "0.1.0")

        def run_callback(_options):
            bundle = cache.latest()
            if bundle is None:
                return []
            return [
                Observation(value, attributes=attributes)
                for value, attributes in build_run_measurements(bundle.run)
            ]

        def span_callback(_options):
            bundle = cache.latest()
            if bundle is None:
                return []
            run_id = str(bundle.run.get("run_id", "unknown"))
            return [
                Observation(value, attributes=attributes)
                for value, attributes in build_span_measurements(run_id, bundle.spans)
            ]

        def service_callback(_options):
            bundle = cache.latest()
            if bundle is None:
                return []
            run_id = str(bundle.run.get("run_id", "unknown"))
            return [
                Observation(value, attributes=attributes)
                for value, attributes in build_service_measurements(run_id, bundle.snapshots)
            ]

        meter.create_observable_gauge(
            "nudemo_latest_run_metric_value",
            callbacks=[run_callback],
            description="Latest benchmark run values exported from persisted telemetry.",
        )
        meter.create_observable_gauge(
            "nudemo_latest_span_metric_value",
            callbacks=[span_callback],
            description="Latest benchmark span values exported from persisted telemetry.",
        )
        meter.create_observable_gauge(
            "nudemo_latest_service_metric_value",
            callbacks=[service_callback],
            description="Latest Docker service snapshot values exported from persisted telemetry.",
        )
        _METER = meter
        _EXPORTER_STARTED = True


def install_http_metrics(app: FastAPI) -> None:
    if not _metrics_enabled():
        return

    from opentelemetry import metrics

    meter = _METER or metrics.get_meter("nudemo.observability", "0.1.0")
    counter = meter.create_counter(
        "nudemo_http_request_total",
        description="HTTP requests served by the nuDemo browser app.",
    )
    latency = meter.create_histogram(
        "nudemo_http_request_duration_ms",
        unit="ms",
        description="Request duration for the nuDemo browser app.",
    )

    @app.middleware("http")
    async def collect_http_metrics(request: Request, call_next):
        start = time.perf_counter()
        response: Response | None = None
        try:
            response = await call_next(request)
            return response
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            route = request.scope.get("route")
            path_template = getattr(route, "path", request.url.path)
            status_code = str(response.status_code if response is not None else 500)
            attributes = {
                "method": request.method,
                "path": str(path_template),
                "status_code": status_code,
            }
            counter.add(1, attributes)
            latency.record(elapsed_ms, attributes)


def _pack_numeric_values(
    base_attributes: dict[str, str],
    values: dict[str, Any],
) -> list[tuple[float, dict[str, str]]]:
    measurements: list[tuple[float, dict[str, str]]] = []
    for key, value in values.items():
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int | float):
            measurements.append(
                (
                    float(value),
                    {**base_attributes, "metric_name": str(key)},
                )
            )
    return measurements


def _metrics_enabled() -> bool:
    raw = os.getenv("NUDEMO_METRICS_ENABLED", "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}
