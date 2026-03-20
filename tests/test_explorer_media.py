from __future__ import annotations

import io

import numpy as np
from PIL import Image

from nudemo.explorer.media import lidar_payload_to_svg, process_camera_payload


def test_process_camera_payload_supports_multiple_modes() -> None:
    image = Image.new("RGB", (24, 12), color=(120, 90, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    payload = buffer.getvalue()

    for mode in ("edges", "grayscale", "contrast"):
        processed = process_camera_payload(payload, mode=mode)
        assert processed[:2] == b"\xff\xd8"


def test_process_camera_payload_rejects_unknown_mode() -> None:
    image = Image.new("RGB", (10, 10), color=(10, 20, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")

    try:
        process_camera_payload(buffer.getvalue(), mode="unknown")
    except ValueError as exc:
        assert "unsupported processing mode" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for unknown mode")


def test_lidar_payload_to_svg_renders_points_and_legend() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.1, 1.0, 0.0],
            [2.0, 1.0, 0.8, 0.7, 0.0],
            [-1.5, -2.2, -0.3, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    buffer = io.BytesIO()
    np.save(buffer, points)

    svg = lidar_payload_to_svg(buffer.getvalue(), width=320, height=180, max_points=16)

    assert svg.startswith("<svg")
    assert "points rendered" in svg
    assert "<circle" in svg
