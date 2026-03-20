from __future__ import annotations

import io
from html import escape

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def process_camera_payload(payload: bytes, *, mode: str = "edges") -> bytes:
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    if mode == "grayscale":
        processed = ImageOps.grayscale(image).convert("RGB")
    elif mode == "contrast":
        processed = ImageEnhance.Contrast(image).enhance(1.7)
    elif mode == "edges":
        processed = ImageOps.grayscale(image).filter(ImageFilter.FIND_EDGES).convert("RGB")
    else:
        raise ValueError(f"unsupported processing mode: {mode}")

    buffer = io.BytesIO()
    processed.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def lidar_payload_to_svg(
    payload: bytes,
    *,
    width: int = 720,
    height: int = 420,
    max_points: int = 2500,
) -> str:
    points = np.load(io.BytesIO(payload), allow_pickle=False)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("expected lidar array shaped like [N, >=3]")
    if not len(points):
        return _empty_svg(width=width, height=height, label="No LiDAR points available")

    if len(points) > max_points:
        stride = max(1, len(points) // max_points)
        points = points[::stride]

    x = points[:, 0].astype(np.float32)
    y = points[:, 1].astype(np.float32)
    z = points[:, 2].astype(np.float32)

    max_abs_x = float(np.max(np.abs(x))) or 1.0
    max_abs_y = float(np.max(np.abs(y))) or 1.0
    pad = 24.0
    inner_width = max(width - 2 * pad, 1.0)
    inner_height = max(height - 2 * pad, 1.0)

    normalized_x = ((x + max_abs_x) / (2 * max_abs_x)) * inner_width + pad
    normalized_y = height - ((((y + max_abs_y) / (2 * max_abs_y)) * inner_height) + pad)

    z_min = float(np.min(z))
    z_span = float(np.max(z) - z_min) or 1.0

    circles: list[str] = []
    for idx in range(len(points)):
        hue = 260 - (((float(z[idx]) - z_min) / z_span) * 130)
        opacity = 0.26 + min(0.7, abs(float(z[idx])) / 4.0)
        circles.append(
            "<circle "
            f'cx="{normalized_x[idx]:.1f}" '
            f'cy="{normalized_y[idx]:.1f}" '
            'r="1.4" '
            f'fill="hsla({hue:.1f}, 94%, 72%, {opacity:.3f})" />'
        )

    legend = f"{len(points):,} points rendered"
    svg_viewbox = f"0 0 {width} {height}"
    plot_rect = (
        f'<rect x="{pad}" y="{pad}" width="{inner_width:.1f}" height="{inner_height:.1f}" '
        'rx="18" fill="#0b0b10" stroke="#544bb0" stroke-width="2"/>'
    )
    center_line_x = (
        f'<line x1="{width / 2:.1f}" y1="{pad}" x2="{width / 2:.1f}" y2="{height - pad}" '
        'stroke="#3f3a6f" stroke-width="1.2" stroke-dasharray="5 5"/>'
    )
    center_line_y = (
        f'<line x1="{pad}" y1="{height / 2:.1f}" x2="{width - pad}" y2="{height / 2:.1f}" '
        'stroke="#3f3a6f" stroke-width="1.2" stroke-dasharray="5 5"/>'
    )
    origin_dot = (
        f'<circle cx="{width / 2:.1f}" cy="{height / 2:.1f}" r="5.5" '
        'fill="#f5f6ff" stroke="#0b0b10" stroke-width="2"/>'
    )
    legend_text = (
        f'<text x="{pad + 8:.1f}" y="{height - 12:.1f}" fill="#b5b0cf" font-size="14" '
        f'font-family="IBM Plex Mono, monospace">{escape(legend)}</text>'
    )
    return f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="{svg_viewbox}">
  <defs>
    <linearGradient id="beam" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#1b1a2a"/>
      <stop offset="100%" stop-color="#111119"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="{width}" height="{height}" rx="22" fill="url(#beam)"/>
  {plot_rect}
  {center_line_x}
  {center_line_y}
  {origin_dot}
  {''.join(circles)}
  {legend_text}
</svg>
""".strip()


def _empty_svg(*, width: int, height: int, label: str) -> str:
    safe_label = escape(label)
    svg_viewbox = f"0 0 {width} {height}"
    label_text = (
        '<text x="50%" y="50%" text-anchor="middle" fill="#b5b0cf" font-size="18" '
        f'font-family="IBM Plex Sans, sans-serif">{safe_label}</text>'
    )
    return f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="{svg_viewbox}">
  <rect x="0" y="0" width="{width}" height="{height}" rx="22" fill="#111119"/>
  {label_text}
</svg>
""".strip()
