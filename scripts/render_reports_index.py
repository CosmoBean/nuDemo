#!/usr/bin/env python3
from __future__ import annotations

import sys
from datetime import datetime
from html import escape
from pathlib import Path


def _links(reports_root: Path) -> list[tuple[str, str]]:
    items = [
        ("Benchmark Dashboard", "benchmark_dashboard.html"),
        ("Telemetry Dashboard", "telemetry_dashboard.html"),
        ("Benchmark Report JSON", "benchmark_report.json"),
        ("Benchmark Results JSON", "benchmark_results.json"),
        ("Benchmark Results CSV", "benchmark_results.csv"),
    ]
    links = [(label, name) for label, name in items if (reports_root / name).exists()]
    extra_dashboards = sorted(reports_root.glob("telemetry_dashboard_*.html"))
    for path in extra_dashboards:
        links.append((path.stem.replace("_", " "), path.name))
    return links


def build_index_html(reports_root: Path) -> str:
    links = _links(reports_root)
    rendered_links = "\n".join(
        (
            "          <li>"
            f"<a href=\"{escape(name)}\">{escape(label)}</a>"
            f"<span>{escape(name)}</span>"
            "</li>"
        )
        for label, name in links
    )
    if not rendered_links:
        rendered_links = (
            "          <li><span>No report artifacts exist yet. Run a benchmark first.</span></li>"
        )
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>nuDemo Reports</title>
    <style>
      :root {{
        --bg: #050507;
        --panel: #181819;
        --line: #6246fb;
        --ink: #f3f7f7;
        --muted: #aabae5;
        --accent: #5436fc;
        --accent-soft: #1a1a20;
        --shadow: 8px 8px 0 #2a2170;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top right, rgba(84,54,252,0.2), transparent 28%),
          linear-gradient(180deg, #09090f 0%, var(--bg) 100%);
      }}
      main {{
        max-width: 960px;
        margin: 0 auto;
        padding: 40px 20px 64px;
      }}
      .hero, .panel {{
        background: var(--panel);
        border: 3px solid var(--line);
        border-radius: 20px;
        box-shadow: var(--shadow);
      }}
      .hero {{
        padding: 28px;
        margin-bottom: 20px;
      }}
      .hero h1 {{
        margin: 0 0 10px;
        font-size: 2rem;
      }}
      .hero p {{
        margin: 0;
        color: var(--muted);
        line-height: 1.6;
      }}
      .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 14px;
        margin-top: 18px;
      }}
      .card {{
        padding: 18px;
        border-radius: 16px;
        background: #1a1a20;
        border: 3px solid var(--line);
        box-shadow: 5px 5px 0 #2a2170;
      }}
      .card strong {{
        display: block;
        font-size: 1.1rem;
        margin-bottom: 6px;
      }}
      .panel {{
        padding: 8px 0;
      }}
      ul {{
        list-style: none;
        margin: 0;
        padding: 0;
      }}
      li {{
        display: flex;
        justify-content: space-between;
        gap: 16px;
        padding: 14px 20px;
        border-top: 2px solid var(--line);
      }}
      li:first-child {{
        border-top: 0;
      }}
      a {{
        color: var(--accent);
        font-weight: 600;
        text-decoration: none;
        overflow-wrap: anywhere;
      }}
      a:hover {{
        text-decoration: underline;
      }}
      span {{
        color: var(--muted);
        min-width: 0;
        overflow-wrap: anywhere;
      }}
      code {{
        font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
        background: var(--accent-soft);
        padding: 2px 6px;
        border-radius: 8px;
      }}
      @media (max-width: 720px) {{
        li {{
          flex-direction: column;
          align-items: flex-start;
        }}
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="hero">
        <h1>nuDemo Report Browser</h1>
        <p>
          Browser entry point for the latest benchmark and telemetry artifacts under
          <code>{escape(str(reports_root))}</code>.
        </p>
        <div class="grid">
          <div class="card">
            <strong>{len(links)}</strong>
            available artifacts
          </div>
          <div class="card">
            <strong>/benchmark_dashboard.html</strong>
            benchmark comparison view
          </div>
          <div class="card">
            <strong>/telemetry_dashboard.html</strong>
            pipeline bottleneck view
          </div>
          <div class="card">
            <strong>{escape(generated_at)}</strong>
            landing page generated
          </div>
        </div>
      </section>
      <section class="panel">
        <ul>
{rendered_links}
        </ul>
      </section>
    </main>
  </body>
</html>
"""


def main() -> int:
    reports_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/reports")
    reports_root.mkdir(parents=True, exist_ok=True)
    output_path = reports_root / "index.html"
    output_path.write_text(build_index_html(reports_root), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
