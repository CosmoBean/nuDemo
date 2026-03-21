#!/usr/bin/env python3
"""
Merge overnight study runs with the known full-trainval minio-postgres numbers
and emit a single comparative HTML + JSON.

Usage:
    # Single run dir containing all backends:
    python scripts/build_comparative_summary.py [run_dir]

    # Split runs (redis ran separately from file backends):
    python scripts/build_comparative_summary.py --file-run <dir> --redis-run <dir>

If no arguments are given, the most-recent overnight_runs subdirectory that
contains redis/lance/parquet/webdataset results is used.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Completed full-trainval minio-postgres numbers (850 scenes, 34,149 samples).
MINIO_POSTGRES_ANCHOR = {
    "backend_key": "minio-postgres",
    "backend_name": "MinIO+PostgreSQL",
    "run_id": "full-trainval-anchor",
    "status": "ok",
    "samples": 34149,
    "scenes": 850,
    "batch_count": 1067,
    "ingest_elapsed_sec": 4197.6,
    "batch_p50_sec": 3.93,
    "batch_p95_sec": 5.12,
    "ingest_throughput_mean": 8.15,
    "random_access_p50_ms": 12.64,
    "sequential_throughput": 263.64,
    "curation_query_ms": 227.87,
    "disk_gb": 68.59,
    "peak_service_cpu": 96.83,
    "peak_service_name": "minio",
    "note": "Full v1.0-trainval measured run (not re-run here to preserve UI data)",
    "benchmark_dashboard": None,
    "telemetry_dashboard": None,
    "report_dir": None,
}


def _find_latest_run(base: Path) -> Path:
    candidates = sorted(
        [
            d
            for d in base.iterdir()
            if d.is_dir()
            and (d / "summary.json").exists()
            and not d.name.startswith("non_minio")
            and not d.name.startswith("test_")
        ],
        key=lambda d: d.name,
        reverse=True,
    )
    for candidate in candidates:
        payload = json.loads((candidate / "summary.json").read_text())
        keys = {b["backend_key"] for b in payload.get("backends", [])}
        if keys & {"redis", "lance", "parquet", "webdataset"}:
            return candidate
    raise FileNotFoundError("No suitable overnight run found")


def _bar(value: float, max_value: float, width: int = 18) -> str:
    if max_value <= 0:
        return "░" * width
    filled = round(min(value / max_value, 1.0) * width)
    return "█" * filled + "░" * (width - filled)


def build_html(payload: dict) -> str:
    backends = payload["backends"]

    def _val(b: dict, key: str) -> float | None:
        return b.get(key)

    def _fmt(v, suffix="", decimals=1):
        if v is None:
            return "<em>n/a</em>"
        return f"{v:,.{decimals}f}{suffix}"

    max_ingest = max((b["ingest_throughput_mean"] or 0) for b in backends) or 1
    max_seq = max((b["sequential_throughput"] or 0) for b in backends) or 1
    max_curation = max((b["curation_query_ms"] or 0) for b in backends) or 1
    max_disk = max((b["disk_gb"] or 0) for b in backends) or 1

    rows = ""
    for b in backends:
        note = f"<br><small style='color:#aaa'>{b['note']}</small>" if b.get("note") else ""
        bench_link = (
            f"<a href='{Path(b['benchmark_dashboard']).name}'>dashboard</a>"
            if b.get("benchmark_dashboard")
            else "anchor"
        )
        ingest = b["ingest_throughput_mean"] or 0
        seq = b["sequential_throughput"] or 0
        cur = b["curation_query_ms"]
        disk = b["disk_gb"] or 0

        rows += f"""
        <tr>
          <td><strong>{b['backend_name']}</strong>{note}</td>
          <td>{b['samples']:,} / {b['scenes']:,}</td>
          <td>{_fmt(b['ingest_throughput_mean'])} s/s
              <div class='bar' style='--fill:{round(ingest/max_ingest*100)}%'></div></td>
          <td>{_fmt(b['batch_p50_sec'], 's', 3)} / {_fmt(b['batch_p95_sec'], 's', 3)}</td>
          <td>{_fmt(b['sequential_throughput'])} s/s
              <div class='bar' style='--fill:{round(seq/max_seq*100)}%'></div></td>
          <td>{_fmt(b['random_access_p50_ms'], ' ms')}</td>
          <td>{_fmt(b['curation_query_ms'], ' ms')}
              {'<div class="bar" style="--fill:' + str(round((cur or 0)/max_curation*100)) + '%"></div>' if cur else ''}</td>
          <td>{_fmt(b['disk_gb'], ' GB')}
              <div class='bar' style='--fill:{round(disk/max_disk*100)}%'></div></td>
          <td>{b.get('peak_service_name') or 'n/a'} {_fmt(b.get('peak_service_cpu'), '%')}</td>
          <td>{bench_link}</td>
        </tr>"""

    meta_note = (
        "<p style='color:#b5b0cf;font-size:0.85em'>"
        "MinIO+PostgreSQL row is the completed full v1.0-trainval anchor run (34,149 samples). "
        "All other backends were measured on the same trainval dataset capped at 200 scenes "
        f"({backends[1]['samples']:,} samples) in this study run."
        "</p>"
        if len(backends) > 1
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>nuDemo — Comparative Backend Summary</title>
  <style>
    :root {{
      --bg: #0b0b10; --panel: #161622; --line: #544bb0;
      --ink: #e7e8f3; --muted: #b5b0cf; --accent: #6b59dd;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: "Space Grotesk","IBM Plex Sans",sans-serif;
      background: radial-gradient(circle at top right,rgba(84,54,252,.12),transparent 22%),
                  linear-gradient(180deg,#0f0f16 0%,var(--bg) 100%);
      color: var(--ink); padding: 32px;
    }}
    main {{ max-width: 1500px; margin: 0 auto; }}
    h1 {{ font-size: 1.8rem; margin-bottom: 8px; }}
    h2 {{ font-size: 1.1rem; color: var(--muted); margin-bottom: 20px; font-weight: 400; }}
    .meta {{ color: var(--muted); font-size: 0.85em; margin-bottom: 24px; }}
    .table-wrap {{
      overflow-x: auto; border: 3px solid var(--line);
      border-radius: 20px; background: var(--panel);
      box-shadow: 6px 6px 0 #211b52;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border: 1px solid var(--line); padding: 10px 12px; vertical-align: top; }}
    th {{ background: var(--accent); font-size: 0.8rem; text-transform: uppercase;
          letter-spacing: .05em; white-space: nowrap; }}
    tr:nth-child(even) {{ background: #12121c; }}
    .bar {{
      height: 4px; background: var(--line); border-radius: 2px; margin-top: 4px;
      position: relative; overflow: hidden;
    }}
    .bar::after {{
      content: ''; position: absolute; left: 0; top: 0; bottom: 0;
      width: var(--fill, 0%); background: var(--accent); border-radius: 2px;
    }}
    a {{ color: var(--ink); }}
    em {{ color: var(--muted); font-style: normal; }}
    small {{ font-size: 0.78em; }}
  </style>
</head>
<body>
  <main>
    <h1>nuDemo — Comparative Backend Summary</h1>
    <h2>nuScenes v1.0-trainval · real data · all five storage backends</h2>
    {meta_note}
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Backend</th>
            <th>Samples / Scenes</th>
            <th>Ingest throughput</th>
            <th>Batch p50 / p95</th>
            <th>Sequential scan</th>
            <th>Random access p50</th>
            <th>Curation query</th>
            <th>Disk footprint</th>
            <th>Peak service CPU</th>
            <th>Report</th>
          </tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </div>
    <p class="meta" style="margin-top:16px">
      Ingest = samples written / elapsed wall-clock.
      Sequential = ordered full-dataset scan.
      Random access = p50 of individual sample fetches.
      Curation = metadata-only filter query (no blob fetch).
      WebDataset does not support random access or curation queries.
    </p>
  </main>
</body>
</html>"""


def _backend_from_run_dir(run_dir: Path, backend_key: str) -> dict | None:
    """Reconstruct a BackendStudySummary dict from a per-backend benchmark_results.json."""
    backend_dir = run_dir / backend_key
    results_path = backend_dir / "benchmark_results.json"
    report_path = backend_dir / "benchmark_report.json"
    if not results_path.exists() or not report_path.exists():
        return None

    results = json.loads(results_path.read_text())
    report = json.loads(report_path.read_text())
    dataset = report.get("dataset", {})

    ingest = [r for r in results if r["pattern"] == "batch_ingest"]
    seq = next((r for r in results if r["pattern"] == "sequential_scan"), None)
    rand = next((r for r in results if r["pattern"] == "random_access"), None)
    curation = next((r for r in results if r["pattern"] == "curation_query"), None)
    disk = next((r for r in results if r["pattern"] == "disk_footprint"), None)

    throughputs = [r["throughput_samples_per_sec"] for r in ingest]
    batch_times = [r["elapsed_sec"] for r in ingest]

    return {
        "backend_key": backend_key,
        "backend_name": report.get("suite_name", backend_key).split(" - ")[-1],
        "run_id": dataset.get("run_id", "unknown"),
        "status": "ok",
        "samples": dataset.get("samples", 0),
        "scenes": dataset.get("scenes", 0),
        "batch_count": len(ingest),
        "ingest_elapsed_sec": round(sum(batch_times), 4),
        "batch_p50_sec": round(statistics.median(batch_times), 4) if batch_times else 0.0,
        "batch_p95_sec": round(_percentile(sorted(batch_times), 95), 4) if batch_times else 0.0,
        "ingest_throughput_mean": round(statistics.mean(throughputs), 4) if throughputs else 0.0,
        "random_access_p50_ms": seq and rand and round(rand["latency_p50_ms"], 4),
        "sequential_throughput": round(seq["throughput_mean"], 4) if seq else None,
        "curation_query_ms": round(curation["query_time_ms_mean"], 4) if curation else None,
        "disk_gb": round(disk["disk_bytes"] / (1024**3), 4) if disk else 0.0,
        "peak_service_cpu": None,
        "peak_service_name": None,
        "benchmark_dashboard": str(backend_dir / "benchmark_dashboard.html"),
        "telemetry_dashboard": str(backend_dir / "telemetry_dashboard.html"),
        "report_dir": str(backend_dir),
    }


def _percentile(sorted_vals: list[float], pct: int) -> float:
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * pct / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] * (1 - (idx - lo)) + sorted_vals[hi] * (idx - lo)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", help="Single run dir with all backends")
    parser.add_argument("--file-run", help="Run dir containing lance/parquet/webdataset")
    parser.add_argument("--redis-run", help="Run dir containing redis (when run separately)")
    args = parser.parse_args(argv)

    base = REPO_ROOT / "artifacts/reports/overnight_runs"
    file_backends = ["lance", "parquet", "webdataset"]

    if args.file_run:
        # Split-run mode: redis from one dir, file backends from another
        file_run = Path(args.file_run)
        redis_run = Path(args.redis_run) if args.redis_run else None
        payload = json.loads((file_run / "summary.json").read_text())
        non_minio = [b for b in payload["backends"] if b["backend_key"] != "minio-postgres"]

        # If file backends are missing from summary.json, reconstruct from per-backend dirs
        present_keys = {b["backend_key"] for b in non_minio}
        for key in file_backends:
            if key not in present_keys:
                reconstructed = _backend_from_run_dir(file_run, key)
                if reconstructed:
                    non_minio.append(reconstructed)

        # Add redis from its own run dir
        if redis_run:
            redis_entry = _backend_from_run_dir(redis_run, "redis")
            if redis_entry:
                redis_entry["note"] = f"Measured on {redis_entry['samples']:,} samples / {redis_entry['scenes']} scenes (separate run)"
                non_minio = [redis_entry] + [b for b in non_minio if b["backend_key"] != "redis"]

        out_dir = file_run
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        payload = json.loads((run_dir / "summary.json").read_text())
        non_minio = [b for b in payload["backends"] if b["backend_key"] != "minio-postgres"]
        out_dir = run_dir
    else:
        run_dir = _find_latest_run(base)
        print(f"Using run dir: {run_dir}")
        payload = json.loads((run_dir / "summary.json").read_text())
        non_minio = [b for b in payload["backends"] if b["backend_key"] != "minio-postgres"]
        out_dir = run_dir

    merged_backends = [MINIO_POSTGRES_ANCHOR, *non_minio]
    merged_payload = {**payload, "backends": merged_backends}

    out_path = out_dir / "comparative_summary.json"
    out_path.write_text(json.dumps(merged_payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    html_path = out_dir / "comparative_summary.html"
    html_path.write_text(build_html(merged_payload), encoding="utf-8")
    print(f"Wrote {html_path}")

    top = REPO_ROOT / "artifacts/reports/comparative_summary.html"
    top.write_text(build_html(merged_payload), encoding="utf-8")
    print(f"Wrote {top}")


if __name__ == "__main__":
    main(sys.argv[1:])
