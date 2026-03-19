from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from nudemo.benchmarks.export import load_report
from nudemo.benchmarks.models import BenchmarkReport


def load_results(results_path: str | Path) -> list[dict[str, object]]:
    path = Path(results_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "results" in payload:
        report = BenchmarkReport.from_dict(payload)
        return [
            {
                "backend": result.backend,
                "pattern": result.pattern,
                **result.metrics,
                **{f"meta_{key}": value for key, value in result.metadata.items()},
            }
            for result in report.results
        ]
    raise ValueError(f"Unsupported results payload in {path}")


def build_app(results_path: str | Path):
    import plotly.express as px
    from dash import Dash, Input, Output, dcc, html

    rows = load_results(results_path)
    app = Dash(__name__)
    backends = sorted({row["backend"] for row in rows})
    patterns = sorted({row["pattern"] for row in rows})

    app.layout = html.Div(
        [
            html.H1("nuDemo Benchmark Dashboard"),
            html.P("Storage and access-pattern comparison for the nuScenes pipeline scaffold."),
            dcc.Dropdown(
                id="pattern-dropdown",
                options=[{"label": pattern, "value": pattern} for pattern in patterns],
                value=patterns[0] if patterns else None,
                clearable=False,
            ),
            dcc.Graph(id="metric-chart"),
            dcc.Graph(
                id="pattern-heatmap",
                figure=_build_heatmap(rows, backends, patterns),
            ),
        ]
    )

    @app.callback(Output("metric-chart", "figure"), Input("pattern-dropdown", "value"))
    def update_chart(selected_pattern: str):
        filtered = [row for row in rows if row["pattern"] == selected_pattern]
        if not filtered:
            return px.bar(title=f"No rows found for {selected_pattern}")
        metric_keys = [
            key
            for key in filtered[0].keys()
            if key not in {"backend", "pattern"}
            and isinstance(filtered[0][key], int | float)
        ]
        metric = metric_keys[0] if metric_keys else None
        if not metric:
            return px.bar(title=f"No numeric metrics for {selected_pattern}")
        return px.bar(
            filtered,
            x="backend",
            y=metric,
            color="backend",
            title=f"{selected_pattern}: {metric}",
        )

    return app


def _build_heatmap(rows: list[dict[str, object]], backends: list[str], patterns: list[str]):
    import plotly.graph_objects as go

    values = []
    for backend in backends:
        row_values = []
        for pattern in patterns:
            match = next(
                (
                    item
                    for item in rows
                    if item["backend"] == backend and item["pattern"] == pattern
                ),
                None,
            )
            numeric_values = [
                value
                for key, value in (match or {}).items()
                if key not in {"backend", "pattern"} and isinstance(value, int | float)
            ]
            row_values.append(float(numeric_values[0]) if numeric_values else 0.0)
        values.append(row_values)
    return go.Figure(
        data=go.Heatmap(z=values, x=patterns, y=backends, colorscale="Viridis"),
        layout={"title": "Access Pattern Heatmap"},
    )


def main(results_path: str | Path) -> int:
    app = build_app(results_path)
    app.run(debug=False)
    return 0


def build_recommendation_summary(report: BenchmarkReport) -> dict[str, str]:
    training = report.best_result("dataloader", "throughput_samples_per_sec", high_is_better=True)
    random_access = report.best_result("random_access", "latency_p50_ms", high_is_better=False)
    curation = report.best_result("curation_query", "query_time_ms", high_is_better=False)
    return {
        "training": training.backend if training else "N/A",
        "random_access": random_access.backend if random_access else "N/A",
        "curation": curation.backend if curation else "N/A",
    }


def build_dashboard_html(report: BenchmarkReport) -> str:
    recommendations = build_recommendation_summary(report)
    rows = "".join(
        f"<tr><td>{result.backend}</td><td>{result.pattern}</td><td>{result.metrics}</td></tr>"
        for result in report.results
    )
    return f"""
    <html>
      <body>
        <h1>{report.suite_name}</h1>
        <h2>Recommendation Summary</h2>
        <p>training: {recommendations['training']}</p>
        <p>random_access: {recommendations['random_access']}</p>
        <p>curation: {recommendations['curation']}</p>
        <h2>Throughput vs. num_workers</h2>
        <h2>Access Pattern Matrix</h2>
        <h2>Results</h2>
        <table>{rows}</table>
      </body>
    </html>
    """


@dataclass(slots=True)
class DashboardApp:
    report: BenchmarkReport

    @classmethod
    def from_json(cls, path: str | Path) -> DashboardApp:
        return cls(load_report(path))

    def render(self) -> str:
        return build_dashboard_html(self.report)
