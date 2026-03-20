import importlib.util
from pathlib import Path

from PIL import Image

from nudemo.config import AppConfig
from nudemo.rendering import render_sample_contact_sheet


def _synthetic_config(monkeypatch, tmp_path: Path) -> AppConfig:
    monkeypatch.setenv("NUDEMO_SYNTHETIC_ENABLED", "true")
    monkeypatch.setenv("NUDEMO_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("NUDEMO_FORMATS_ROOT", str(tmp_path / "artifacts" / "formats"))
    monkeypatch.setenv("NUDEMO_REPORTS_ROOT", str(tmp_path / "artifacts" / "reports"))
    return AppConfig.load()


def test_render_sample_contact_sheet_creates_png(monkeypatch, tmp_path: Path) -> None:
    config = _synthetic_config(monkeypatch, tmp_path)

    artifact = render_sample_contact_sheet(config, sample_idx=0, provider_name="synthetic")

    assert artifact.artifact_type == "sample_contact_sheet"
    assert artifact.output_path.exists()
    assert artifact.output_path.suffix == ".png"
    image = Image.open(artifact.output_path)
    assert image.size[0] > 1000
    assert image.size[1] > 500
    assert artifact.metadata["provider"] == "synthetic"


def test_reports_index_includes_render_artifacts(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "render_reports_index",
        Path(__file__).resolve().parents[1] / "scripts" / "render_reports_index.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    reports_root = tmp_path / "reports"
    renders_root = reports_root / "renders"
    renders_root.mkdir(parents=True)
    (reports_root / "benchmark_dashboard.html").write_text("<html></html>", encoding="utf-8")
    (renders_root / "scene_demo.gif").write_bytes(b"GIF89a")

    html = module.build_index_html(reports_root)

    assert "Render: scene demo" in html
    assert "renders/scene_demo.gif" in html
