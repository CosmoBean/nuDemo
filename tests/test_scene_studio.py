from __future__ import annotations

from nudemo.explorer.app import build_scene_studio_html


def test_scene_studio_html_contains_threejs_and_lidar_api() -> None:
    html = build_scene_studio_html()

    assert "Scene Studio" in html
    assert "three.module.js" in html
    assert "/api/samples/${sampleIdx}/lidar/points" in html
    assert "/api/scenes/${encodeURIComponent(sceneToken)}" in html
