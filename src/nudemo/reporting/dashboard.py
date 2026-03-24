from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from statistics import mean

from nudemo.benchmarks.export import load_report
from nudemo.benchmarks.models import BenchmarkReport, BenchmarkResult

__all__ = [
    "DashboardApp",
    "build_comparison_note",
    "build_dashboard_html",
    "build_recommendation_summary",
    "build_storage_format_rows",
    "load_results",
]


def load_results(results_path: str | Path) -> list[dict[str, object]]:
    path = Path(results_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "results" in payload:
        report = BenchmarkReport.from_dict(payload)
        return [_flatten_result(result) for result in report.results]
    raise ValueError(f"Unsupported results payload in {path}")


def build_recommendation_summary(report: BenchmarkReport) -> dict[str, str]:
    training = report.best_result("dataloader", "throughput_samples_per_sec", high_is_better=True)
    if training is None:
        training = report.best_result("sequential_scan", "throughput_mean", high_is_better=True)
    random_access = report.best_result("random_access", "latency_p50_ms", high_is_better=False)
    curation = report.best_result("curation_query", "query_time_ms", high_is_better=False)
    if curation is None:
        curation = report.best_result("curation_query", "query_time_ms_mean", high_is_better=False)
    ingestion = _best_stage_candidate(
        report,
        stage="ingestion",
        preferred_metrics=("throughput_msg_sec", "throughput_mb_sec"),
        high_is_better=True,
    )
    storage = _best_stage_candidate(
        report,
        stage="storage",
        preferred_metrics=("throughput_samples_per_sec", "throughput"),
        high_is_better=True,
    )
    return {
        "ingestion": ingestion.backend if ingestion else "N/A",
        "storage": storage.backend if storage else "N/A",
        "training": training.backend if training else "N/A",
        "random_access": random_access.backend if random_access else "N/A",
        "curation": curation.backend if curation else "N/A",
    }


def build_dashboard_html(report: BenchmarkReport) -> str:
    recommendations = build_recommendation_summary(report)
    storage_rows = build_storage_format_rows(report)
    comparison_note = build_comparison_note(storage_rows)
    summary_cards = _build_summary_cards(report)
    comparison_cards = _build_comparison_cards(storage_rows)
    comparison_table = _build_storage_summary_table(storage_rows)
    stage_table = _build_stage_table(report)
    dataloader_table = _build_dataloader_table(report)
    result_rows = "".join(_build_result_row(result) for result in report.results)
    recommendation_rows = "".join(
        f"<tr><th>{stage}</th><td>{backend}</td></tr>"
        for stage, backend in recommendations.items()
    )
    sample_rows = "".join(
        _build_sample_row(result) for result in report.results[:8]
    )
    return f"""
    <html>
      <head>
        <style>
          :root {{
            --bg: #0b0b10;
            --panel: #161622;
            --line: #544bb0;
            --ink: #e7e8f3;
            --muted: #b5b0cf;
            --accent-alt: #6b59dd;
            --accent-soft: #201f31;
            --shadow: 6px 6px 0 #211b52;
          }}
          body {{
            font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
            margin: 32px;
            color: var(--ink);
            background:
              radial-gradient(circle at top right, rgba(84, 54, 252, 0.12), transparent 22%),
              linear-gradient(180deg, #0f0f16 0%, var(--bg) 100%);
          }}
          main {{ max-width: 1440px; margin: 0 auto; }}
          h1, h2 {{ margin-bottom: 12px; }}
          h1 {{ letter-spacing: -0.04em; line-height: 1; }}
          .cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin: 16px 0 24px;
          }}
          .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 12px;
            margin: 16px 0 24px;
          }}
          .card {{
            border: 3px solid var(--line);
            border-radius: 20px;
            padding: 16px;
            background: var(--panel);
            box-shadow: 4px 4px 0 #211b52;
          }}
          .metric-chart {{
            border: 3px solid var(--line);
            border-radius: 20px;
            padding: 16px;
            background: var(--panel);
            box-shadow: 4px 4px 0 #211b52;
          }}
          .metric-chart h3 {{ margin: 0 0 6px; }}
          .metric-chart p {{ margin: 0 0 12px; }}
          .chart-rows {{
            display: grid;
            gap: 10px;
          }}
          .chart-row {{
            display: grid;
            gap: 6px;
          }}
          .chart-row-head {{
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: baseline;
          }}
          .chart-track {{
            width: 100%;
            height: 14px;
            border: 2px solid var(--line);
            border-radius: 999px;
            background: var(--accent-soft);
            overflow: hidden;
          }}
          .chart-bar {{
            height: 100%;
            min-width: 8px;
            background: linear-gradient(90deg, var(--accent-alt), #8e82e8);
          }}
          .backend-meta {{
            display: block;
            font-size: 0.82rem;
            color: var(--muted);
            margin-top: 3px;
          }}
          .table-wrap {{
            overflow-x: auto;
            margin: 12px 0 24px;
            border: 3px solid var(--line);
            border-radius: 20px;
            background: var(--panel);
            box-shadow: var(--shadow);
          }}
          table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            margin: 0;
          }}
          th, td {{
            border: 2px solid var(--line);
            padding: 8px 10px;
            text-align: left;
            vertical-align: top;
            overflow-wrap: anywhere;
            word-break: break-word;
          }}
          th {{
            background: var(--accent-alt);
            color: var(--ink);
          }}
          .muted {{ color: var(--muted); }}
          code {{
            overflow-wrap: anywhere;
            word-break: break-word;
            background: var(--accent-soft);
            padding: 2px 6px;
            border-radius: 8px;
          }}
          .compact {{ font-size: 0.92rem; line-height: 1.45; }}
        </style>
      </head>
      <body>
        <main>
          <h1>{report.suite_name}</h1>
          <p class="muted compact">
            Created at {_report_created_at(report)}.
            Dataset summary: {_format_dataset_summary(report.dataset)}.
          </p>
          <div class="cards">{summary_cards}</div>
          <h2>Backend Comparison</h2>
          <p class="muted compact">{escape(comparison_note)}</p>
          <div class="chart-grid">{comparison_cards}</div>
          <div class="table-wrap"><table>{comparison_table}</table></div>
          <h2>Recommendation Summary</h2>
          <div class="table-wrap"><table>{recommendation_rows}</table></div>
          <h2>Stage Summary</h2>
          <div class="table-wrap"><table>{stage_table}</table></div>
          <h2>Throughput vs. num_workers</h2>
          <div class="table-wrap"><table>{dataloader_table}</table></div>
          <h2>Access Pattern Matrix</h2>
          <div class="table-wrap"><table>{sample_rows}</table></div>
          <h2>Results</h2>
          <div class="table-wrap">
            <table>
              <tr>
                <th>Stage</th>
                <th>Backend</th>
                <th>Pattern</th>
                <th>Status</th>
                <th>Metrics</th>
                <th>Samples</th>
                <th>Elapsed (s)</th>
              </tr>
              {result_rows}
            </table>
          </div>
        </main>
      </body>
    </html>
    """


def _best_stage_candidate(
    report: BenchmarkReport,
    *,
    stage: str,
    preferred_metrics: tuple[str, ...],
    high_is_better: bool,
):
    for metric in preferred_metrics:
        matches = [
            result
            for result in report.results
            if result.stage == stage and result.status == "ok" and metric in result.metrics
        ]
        if matches:
            chooser = max if high_is_better else min
            return chooser(matches, key=lambda item: item.metrics[metric])
    return None


def _build_summary_cards(report: BenchmarkReport) -> str:
    stages = sorted({result.stage for result in report.results})
    backends = sorted({result.backend for result in report.results})
    cards = [
        ("Results", str(len(report.results))),
        ("Stages", str(len(stages))),
        ("Backends", str(len(backends))),
        ("Samples", str(report.dataset.get("samples", "n/a"))),
    ]
    return "".join(
        (
            "<div class='card'>"
            f"<div class='muted'>{label}</div>"
            f"<div><strong>{value}</strong></div>"
            "</div>"
        )
        for label, value in cards
    )


def build_storage_format_rows(report: BenchmarkReport) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for backend in _storage_backends(report):
        write_result = _match(report, backend, "write_throughput")
        batch_results = _matches(report, backend, "batch_ingest")
        scan_result = _match(report, backend, "sequential_scan")
        random_result = _match(report, backend, "random_access")
        disk_result = _match(report, backend, "disk_footprint")
        curation_result = _match(report, backend, "curation_query")
        source = _source_metadata(
            write_result,
            *batch_results,
            scan_result,
            random_result,
            disk_result,
            curation_result,
        )
        rows.append(
            {
                "backend": backend,
                "write_samples_per_sec": _storage_write_metric(write_result, batch_results),
                "sequential_samples_per_sec": _metric(
                    scan_result, "throughput_mean", "throughput_samples_per_sec"
                ),
                "random_access_p50_ms": _metric(random_result, "latency_p50_ms"),
                "random_access_p95_ms": _metric(random_result, "latency_p95_ms"),
                "disk_mb": _metric(disk_result, "disk_mb"),
                "curation_query_ms": _metric(
                    curation_result, "query_time_ms_mean", "query_time_ms"
                ),
                "status": _status(
                    write_result,
                    *batch_results,
                    scan_result,
                    random_result,
                    disk_result,
                ),
                "samples": source.get("source_samples", report.dataset.get("samples")),
                "scenes": source.get("source_scenes", report.dataset.get("scenes")),
                "provider": source.get("source_provider", report.dataset.get("provider")),
                "run_id": source.get("source_run_id", report.dataset.get("run_id")),
                "created_at": source.get("source_created_at", report.created_at),
            }
        )
    return rows


def build_comparison_note(storage_rows: list[dict[str, object]]) -> str:
    if not storage_rows:
        return "No storage backend metrics were found."
    scopes = {(row.get("samples"), row.get("scenes")) for row in storage_rows}
    if len(scopes) <= 1:
        return "All backend rows come from the same sample and scene scope."
    return (
        "Rows come from the latest completed run for each backend. Sample and scene counts differ "
        "across backends, so compare relative behavior unless you rerun them on a common slice."
    )


def _build_stage_table(report: BenchmarkReport) -> str:
    rows = []
    for stage in sorted({result.stage for result in report.results}):
        stage_results = [result for result in report.results if result.stage == stage]
        ok_count = sum(1 for result in stage_results if result.status == "ok")
        elapsed_values = [result.elapsed_sec for result in stage_results if result.elapsed_sec]
        avg_elapsed = mean(elapsed_values) if elapsed_values else 0.0
        rows.append(
            "<tr>"
            f"<th>{stage}</th>"
            f"<td>{len(stage_results)}</td>"
            f"<td>{ok_count}</td>"
            f"<td>{avg_elapsed:.4f}</td>"
            "</tr>"
        )
    return (
        "<tr><th>Stage</th><th>Rows</th><th>OK Rows</th><th>Avg Elapsed (s)</th></tr>"
        + "".join(rows)
    )


def _build_dataloader_table(report: BenchmarkReport) -> str:
    rows = []
    dataloader_results = [result for result in report.results if result.pattern == "dataloader"]
    if not dataloader_results:
        return "<tr><td>No DataLoader metrics captured in this report.</td></tr>"
    for result in dataloader_results:
        rows.append(
            "<tr>"
            f"<td>{result.backend}</td>"
            f"<td>{result.metadata.get('num_workers', 'n/a')}</td>"
            f"<td>{result.metadata.get('batch_size', 'n/a')}</td>"
            f"<td>{result.metrics.get('throughput_samples_per_sec', 0.0):.4f}</td>"
            "</tr>"
        )
    return (
        "<tr><th>Backend</th><th>num_workers</th><th>batch_size</th><th>Throughput</th></tr>"
        + "".join(rows)
    )


def _build_comparison_cards(storage_rows: list[dict[str, object]]) -> str:
    metrics = [
        ("Write Throughput", "write_samples_per_sec", "higher is better", True, "samples/s"),
        (
            "Sequential Read",
            "sequential_samples_per_sec",
            "higher is better",
            True,
            "samples/s",
        ),
        ("Random Access", "random_access_p50_ms", "lower is better", False, "ms p50"),
        ("Random Access p95", "random_access_p95_ms", "lower is better", False, "ms p95"),
        ("Curation Query", "curation_query_ms", "lower is better", False, "ms"),
        ("Disk Footprint", "disk_mb", "lower is better", False, "MB"),
    ]
    return "".join(
        _build_metric_chart(storage_rows, title, key, subtitle, higher_is_better, unit)
        for title, key, subtitle, higher_is_better, unit in metrics
    )


def _build_metric_chart(
    storage_rows: list[dict[str, object]],
    title: str,
    key: str,
    subtitle: str,
    higher_is_better: bool,
    unit: str,
) -> str:
    values = [
        float(row[key])
        for row in storage_rows
        if isinstance(row.get(key), int | float) and float(row[key]) > 0
    ]
    if not values:
        return (
            "<section class='metric-chart'>"
            f"<h3>{escape(title)}</h3>"
            f"<p class='muted compact'>{escape(subtitle)}</p>"
            "<div class='muted compact'>No data captured for this metric.</div>"
            "</section>"
        )

    baseline = max(values) if higher_is_better else min(values)
    rows: list[str] = []
    for row in storage_rows:
        value = row.get(key)
        if not isinstance(value, int | float) or float(value) <= 0:
            continue
        numeric = float(value)
        width = (
            (numeric / baseline) * 100.0
            if higher_is_better
            else (baseline / numeric) * 100.0
        )
        rows.append(
            "<div class='chart-row'>"
            "<div class='chart-row-head'>"
            f"<div><strong>{escape(str(row['backend']))}</strong>"
            f"<span class='backend-meta'>{escape(_scope_label(row))}</span></div>"
            f"<span>{escape(_format_metric_value(numeric, unit))}</span>"
            "</div>"
            "<div class='chart-track'>"
            f"<div class='chart-bar' style='width:{max(8.0, min(width, 100.0)):.2f}%'></div>"
            "</div>"
            "</div>"
        )
    return (
        "<section class='metric-chart'>"
        f"<h3>{escape(title)}</h3>"
        f"<p class='muted compact'>{escape(subtitle)}</p>"
        f"<div class='chart-rows'>{''.join(rows)}</div>"
        "</section>"
    )


def _build_storage_summary_table(storage_rows: list[dict[str, object]]) -> str:
    rows = [
        (
            "<tr><th>Backend</th><th>Dataset Scope</th><th>Write samples/s</th>"
            "<th>Sequential samples/s</th><th>Random access p50 ms</th>"
            "<th>Random access p95 ms</th>"
            "<th>Disk MB</th><th>Curation ms</th><th>Status</th></tr>"
        )
    ]
    for row in storage_rows:
        rows.append(
            "<tr>"
            f"<td>{escape(str(row['backend']))}</td>"
            f"<td>{escape(_scope_label(row))}</td>"
            f"<td>{escape(_format_table_metric(row.get('write_samples_per_sec')))}</td>"
            f"<td>{escape(_format_table_metric(row.get('sequential_samples_per_sec')))}</td>"
            f"<td>{escape(_format_table_metric(row.get('random_access_p50_ms')))}</td>"
            f"<td>{escape(_format_table_metric(row.get('random_access_p95_ms')))}</td>"
            f"<td>{escape(_format_table_metric(row.get('disk_mb')))}</td>"
            f"<td>{escape(_format_table_metric(row.get('curation_query_ms')))}</td>"
            f"<td>{escape(str(row.get('status', 'unknown')))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _format_metrics(metrics: dict[str, float]) -> str:
    if not metrics:
        return "n/a"
    return ", ".join(f"{key}={float(value):.4f}" for key, value in metrics.items())


def _build_result_row(result: BenchmarkResult) -> str:
    elapsed = result.elapsed_sec if result.elapsed_sec is not None else 0.0
    return (
        "<tr>"
        f"<td>{result.stage}</td>"
        f"<td>{result.backend}</td>"
        f"<td>{result.pattern}</td>"
        f"<td>{result.status}</td>"
        f"<td class='compact'>{_format_metrics(result.metrics)}</td>"
        f"<td>{result.sample_count}</td>"
        f"<td>{elapsed:.4f}</td>"
        "</tr>"
    )


def _build_sample_row(result: BenchmarkResult) -> str:
    return (
        "<tr>"
        f"<td>{result.stage}</td>"
        f"<td>{result.backend}</td>"
        f"<td>{result.pattern}</td>"
        f"<td class='compact'>{_format_metrics(result.metrics)}</td>"
        "</tr>"
    )


def _report_created_at(report: BenchmarkReport) -> str:
    return getattr(report, "created_at", getattr(report, "generated_at", "n/a"))


def _format_dataset_summary(dataset: dict[str, object]) -> str:
    if not dataset:
        return "n/a"
    parts = [f"{key}={value}" for key, value in dataset.items()]
    return ", ".join(parts)


def _flatten_result(result: BenchmarkResult) -> dict[str, object]:
    return {
        "stage": result.stage,
        "backend": result.backend,
        "pattern": result.pattern,
        "sample_count": result.sample_count,
        "elapsed_sec": result.elapsed_sec,
        "status": result.status,
        "error": result.error or "",
        **result.metrics,
        **{f"meta_{key}": value for key, value in result.metadata.items()},
    }


def _storage_backends(report: BenchmarkReport) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for result in report.results:
        if result.backend in {"Kafka", "real"}:
            continue
        if result.backend in seen:
            continue
        ordered.append(result.backend)
        seen.add(result.backend)
    return ordered


def _match(report: BenchmarkReport, backend: str, pattern: str) -> BenchmarkResult | None:
    for result in report.results:
        if result.backend == backend and result.pattern == pattern:
            return result
    return None


def _matches(report: BenchmarkReport, backend: str, pattern: str) -> list[BenchmarkResult]:
    return [
        result
        for result in report.results
        if result.backend == backend and result.pattern == pattern
    ]


def _metric(result: BenchmarkResult | None, *keys: str) -> float | None:
    if result is None:
        return None
    for key in keys:
        value = result.metrics.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def _status(*results: BenchmarkResult | None) -> str:
    statuses = [result.status for result in results if result is not None]
    if not statuses:
        return "missing"
    if any(status != "ok" for status in statuses):
        return "degraded"
    return "ok"


def _source_metadata(*results: BenchmarkResult | None) -> dict[str, object]:
    for result in results:
        if result is not None and result.metadata:
            return result.metadata
    return {}


def _storage_write_metric(
    write_result: BenchmarkResult | None,
    batch_results: list[BenchmarkResult],
) -> float | None:
    metric = _metric(write_result, "throughput_samples_per_sec", "throughput")
    if metric is not None:
        return metric
    if not batch_results:
        return None
    total_samples = sum(result.sample_count for result in batch_results)
    total_elapsed = sum(result.elapsed_sec for result in batch_results if result.elapsed_sec)
    if total_samples > 0 and total_elapsed > 0:
        return total_samples / total_elapsed
    throughputs = [
        float(value)
        for result in batch_results
        if (value := result.metrics.get("throughput_samples_per_sec")) is not None
    ]
    if not throughputs:
        return None
    return sum(throughputs) / len(throughputs)


def _scope_label(row: dict[str, object]) -> str:
    samples = row.get("samples")
    scenes = row.get("scenes")
    pieces = []
    if samples not in (None, ""):
        pieces.append(f"{samples} samples")
    if scenes not in (None, ""):
        pieces.append(f"{scenes} scenes")
    return " · ".join(pieces) if pieces else "scope unavailable"


def _format_metric_value(value: float, unit: str) -> str:
    return f"{value:,.2f} {unit}"


def _format_table_metric(value: object) -> str:
    if isinstance(value, int | float):
        return f"{float(value):,.2f}"
    return "n/a"


def main(results_path: str | Path) -> int:
    input_path = Path(results_path)
    report = load_report(input_path)
    html = build_dashboard_html(report)
    output_path = input_path.with_name("benchmark_dashboard.html")
    output_path.write_text(html, encoding="utf-8")
    print(output_path)
    return 0


@dataclass(slots=True)
class DashboardApp:
    report: BenchmarkReport

    @classmethod
    def from_json(cls, path: str | Path) -> DashboardApp:
        return cls(load_report(path))

    def render(self) -> str:
        return build_dashboard_html(self.report)
