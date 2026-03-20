from __future__ import annotations

from html import escape


def build_telemetry_dashboard_html(
    run: dict[str, object],
    spans: list[dict[str, object]],
    snapshots: list[dict[str, object]],
) -> str:
    summary = dict(run.get("summary") or {})
    dataset = dict(run.get("dataset") or {})
    summary_cards = _build_summary_cards(run, summary, dataset)
    bottleneck_rows = _build_bottleneck_rows(spans)
    span_rows = _build_span_rows(spans)
    peak_rows = _build_service_peaks(snapshots)
    snapshot_rows = _build_snapshot_rows(snapshots)
    artifact_rows = _build_artifact_rows(run)
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
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
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
            font-family: monospace;
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
          <h1>Telemetry Dashboard</h1>
          <p class="muted compact">
            Run <code>{escape(str(run.get("run_id", "n/a")))}</code> for
            <strong>{escape(str(run.get("suite_name", "n/a")))}</strong>.
            Status: <strong>{escape(str(run.get("status", "n/a")))}</strong>.
          </p>
          <div class="cards">{summary_cards}</div>
          <h2>Artifacts</h2>
          <div class="table-wrap"><table>{artifact_rows}</table></div>
          <h2>Top Bottlenecks</h2>
          <div class="table-wrap"><table>{bottleneck_rows}</table></div>
          <h2>Service Peaks</h2>
          <div class="table-wrap"><table>{peak_rows}</table></div>
          <h2>Span Timeline</h2>
          <div class="table-wrap"><table>{span_rows}</table></div>
          <h2>Service Snapshots</h2>
          <div class="table-wrap"><table>{snapshot_rows}</table></div>
        </main>
      </body>
    </html>
    """


def _build_summary_cards(
    run: dict[str, object],
    summary: dict[str, object],
    dataset: dict[str, object],
) -> str:
    cards = [
        ("Provider", str(run.get("provider", "n/a"))),
        ("Samples", str(dataset.get("samples", "n/a"))),
        ("Elapsed (s)", _format_float(run.get("elapsed_sec"))),
        ("Results", str(summary.get("result_count", "n/a"))),
        ("Errors", str(summary.get("error_count", "n/a"))),
    ]
    return "".join(
        (
            "<div class='card'>"
            f"<div class='muted'>{escape(label)}</div>"
            f"<div><strong>{escape(value)}</strong></div>"
            "</div>"
        )
        for label, value in cards
    )


def _build_artifact_rows(run: dict[str, object]) -> str:
    rows = [
        ("Benchmark Report", run.get("report_path")),
        ("Flat JSON", run.get("json_path")),
        ("CSV", run.get("csv_path")),
        ("Benchmark Dashboard", run.get("dashboard_path")),
        ("Telemetry Dashboard", run.get("telemetry_dashboard_path")),
    ]
    rendered = [
        f"<tr><th>{escape(label)}</th><td>{escape(str(value or 'n/a'))}</td></tr>"
        for label, value in rows
    ]
    return "".join(rendered)


def _build_bottleneck_rows(spans: list[dict[str, object]]) -> str:
    top_spans = sorted(
        spans,
        key=lambda span: float(span.get("elapsed_sec") or 0.0),
        reverse=True,
    )[:10]
    rows = [
        (
            "<tr>"
            "<th>Stage</th><th>Backend</th><th>Pattern</th><th>Status</th>"
            "<th>Elapsed (s)</th><th>Samples</th><th>Error</th>"
            "</tr>"
        )
    ]
    for span in top_spans:
        rows.append(
            "<tr>"
            f"<td>{escape(str(span.get('stage', '')))}</td>"
            f"<td>{escape(str(span.get('backend', '')))}</td>"
            f"<td>{escape(str(span.get('pattern', '')))}</td>"
            f"<td>{escape(str(span.get('status', '')))}</td>"
            f"<td>{_format_float(span.get('elapsed_sec'))}</td>"
            f"<td>{escape(str(span.get('sample_count', 0)))}</td>"
            f"<td>{escape(str(span.get('error') or ''))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _build_service_peaks(snapshots: list[dict[str, object]]) -> str:
    grouped: dict[str, list[dict[str, object]]] = {}
    for snapshot in snapshots:
        grouped.setdefault(str(snapshot.get("service", "unknown")), []).append(snapshot)

    rows = [
        (
            "<tr><th>Service</th><th>Peak CPU %</th><th>Peak Memory</th>"
            "<th>Peak Net In</th><th>Peak Net Out</th><th>Snapshots</th></tr>"
        )
    ]
    for service in sorted(grouped):
        service_rows = grouped[service]
        peak_cpu = max(float(row.get("cpu_percent") or 0.0) for row in service_rows)
        peak_mem = max(int(row.get("mem_usage_bytes") or 0) for row in service_rows)
        peak_net_in = max(int(row.get("net_input_bytes") or 0) for row in service_rows)
        peak_net_out = max(int(row.get("net_output_bytes") or 0) for row in service_rows)
        rows.append(
            "<tr>"
            f"<td>{escape(service)}</td>"
            f"<td>{peak_cpu:.2f}</td>"
            f"<td>{_format_bytes(peak_mem)}</td>"
            f"<td>{_format_bytes(peak_net_in)}</td>"
            f"<td>{_format_bytes(peak_net_out)}</td>"
            f"<td>{len(service_rows)}</td>"
            "</tr>"
        )
    return "".join(rows)


def _build_span_rows(spans: list[dict[str, object]]) -> str:
    rows = [
        (
            "<tr><th>Started</th><th>Stage</th><th>Backend</th><th>Pattern</th>"
            "<th>Status</th><th>Elapsed (s)</th><th>Sample Count</th><th>Key Metrics</th></tr>"
        )
    ]
    for span in spans:
        metrics = dict(span.get("metrics") or {})
        rows.append(
            "<tr>"
            f"<td>{escape(str(span.get('started_at', '')))}</td>"
            f"<td>{escape(str(span.get('stage', '')))}</td>"
            f"<td>{escape(str(span.get('backend', '')))}</td>"
            f"<td>{escape(str(span.get('pattern', '')))}</td>"
            f"<td>{escape(str(span.get('status', '')))}</td>"
            f"<td>{_format_float(span.get('elapsed_sec'))}</td>"
            f"<td>{escape(str(span.get('sample_count', 0)))}</td>"
            f"<td class='compact'>{escape(_format_metrics(metrics))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _build_snapshot_rows(snapshots: list[dict[str, object]]) -> str:
    rows = [
        (
            "<tr><th>Observed</th><th>Label</th><th>Service</th><th>CPU %</th>"
            "<th>Memory</th><th>Mem %</th><th>Net In</th><th>Net Out</th><th>PIDs</th></tr>"
        )
    ]
    for snapshot in snapshots:
        rows.append(
            "<tr>"
            f"<td>{escape(str(snapshot.get('observed_at', '')))}</td>"
            f"<td>{escape(str(snapshot.get('snapshot_label', '')))}</td>"
            f"<td>{escape(str(snapshot.get('service', '')))}</td>"
            f"<td>{_format_float(snapshot.get('cpu_percent'))}</td>"
            f"<td>{_format_bytes(int(snapshot.get('mem_usage_bytes') or 0))}</td>"
            f"<td>{_format_float(snapshot.get('mem_percent'))}</td>"
            f"<td>{_format_bytes(int(snapshot.get('net_input_bytes') or 0))}</td>"
            f"<td>{_format_bytes(int(snapshot.get('net_output_bytes') or 0))}</td>"
            f"<td>{escape(str(snapshot.get('pids') or ''))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _format_metrics(metrics: dict[str, object]) -> str:
    if not metrics:
        return "n/a"
    preview = []
    for key, value in metrics.items():
        if isinstance(value, int | float):
            preview.append(f"{key}={value:.4f}")
        else:
            preview.append(f"{key}={value}")
    return ", ".join(preview[:4])


def _format_bytes(value: int) -> str:
    if value <= 0:
        return "0B"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024
    return f"{value}B"


def _format_float(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"
