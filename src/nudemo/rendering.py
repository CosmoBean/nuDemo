from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from nudemo.config import AppConfig
from nudemo.domain.models import CAMERAS
from nudemo.extraction.providers import resolve_provider


@dataclass(slots=True)
class RenderArtifact:
    artifact_type: str
    output_path: Path
    metadata: dict[str, object]


def _reports_render_root(config: AppConfig) -> Path:
    root = config.runtime.reports_root / "renders"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _camera_tile(array: np.ndarray, *, label: str, tile_size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(array.astype(np.uint8))
    image = ImageOps.fit(image, tile_size, method=Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    font = _load_font(20)
    text_box_height = 38
    draw.rectangle(
        [(0, tile_size[1] - text_box_height), (tile_size[0], tile_size[1])],
        fill=(22, 22, 34),
    )
    draw.text((12, tile_size[1] - text_box_height + 8), label, fill=(231, 232, 243), font=font)
    return image


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    *,
    origin: tuple[int, int],
    width: int,
    lines: list[str],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    line_height: int,
) -> None:
    x, y = origin
    for line in lines:
        words = line.split()
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            bbox = draw.textbbox((x, y), candidate, font=font)
            if bbox[2] - bbox[0] <= width or not current:
                current = candidate
                continue
            draw.text((x, y), current, fill=fill, font=font)
            y += line_height
            current = word
        if current:
            draw.text((x, y), current, fill=fill, font=font)
            y += line_height


def render_sample_contact_sheet(
    config: AppConfig,
    *,
    sample_idx: int,
    provider_name: str = "real",
    output_path: Path | None = None,
) -> RenderArtifact:
    if sample_idx < 0:
        raise ValueError("sample_idx must be non-negative")

    provider = resolve_provider(config, provider_name)
    selected_sample = None
    for current_idx, sample in enumerate(provider.iter_samples(limit=sample_idx + 1)):
        if current_idx == sample_idx:
            selected_sample = sample
            break

    if selected_sample is None:
        raise IndexError(f"sample_idx {sample_idx} is out of range for provider={provider_name}")

    output_path = output_path or (
        _reports_render_root(config) / f"sample_{provider_name}_{sample_idx:05d}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    padding = 24
    header_height = 132
    tile_size = (420, 236)
    canvas_width = padding * 4 + tile_size[0] * 3
    canvas_height = header_height + padding * 3 + tile_size[1] * 2

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(11, 11, 16))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(34)
    meta_font = _load_font(18)
    strong_font = _load_font(20)

    draw.rounded_rectangle(
        [(padding, padding), (canvas_width - padding, header_height)],
        radius=22,
        fill=(22, 22, 34),
        outline=(84, 75, 176),
        width=3,
    )
    draw.text(
        (padding + 18, padding + 16),
        f"Sample {sample_idx:05d}",
        fill=(231, 232, 243),
        font=title_font,
    )
    header_lines = [
        f"Scene: {selected_sample.scene_name}  |  Location: {selected_sample.location}",
        (
            f"Token: {selected_sample.token[:18]}...  |  "
            f"Timestamp: {selected_sample.timestamp}  |  "
            f"Annotations: {len(selected_sample.annotations)}  |  LiDAR points: "
            f"{int(selected_sample.lidar_top.shape[0])}"
        ),
    ]
    _draw_wrapped_text(
        draw,
        origin=(padding + 18, padding + 64),
        width=canvas_width - (padding + 18) * 2,
        lines=header_lines,
        font=meta_font,
        fill=(181, 176, 207),
        line_height=24,
    )

    for index, camera in enumerate(CAMERAS):
        tile = _camera_tile(selected_sample.cameras[camera], label=camera, tile_size=tile_size)
        row, col = divmod(index, 3)
        left = padding + col * (tile_size[0] + padding)
        top = header_height + padding + row * (tile_size[1] + padding)
        canvas.paste(tile, (left, top))
        draw.rounded_rectangle(
            [(left, top), (left + tile_size[0], top + tile_size[1])],
            radius=18,
            outline=(84, 75, 176),
            width=3,
        )

    draw.text(
        (canvas_width - 300, canvas_height - 28),
        f"provider={provider_name}",
        fill=(107, 89, 221),
        font=strong_font,
    )
    canvas.save(output_path, format="PNG")
    return RenderArtifact(
        artifact_type="sample_contact_sheet",
        output_path=output_path,
        metadata={
            "provider": provider_name,
            "sample_idx": sample_idx,
            "scene_name": selected_sample.scene_name,
            "location": selected_sample.location,
            "annotations": len(selected_sample.annotations),
            "lidar_points": int(selected_sample.lidar_top.shape[0]),
        },
    )


def render_scene_gif(
    config: AppConfig,
    *,
    scene_name: str | None = None,
    camera: str = "CAM_FRONT",
    max_frames: int = 24,
    step: int = 1,
    fps: int = 2,
    output_path: Path | None = None,
) -> RenderArtifact:
    from nuscenes.nuscenes import NuScenes

    if camera not in CAMERAS:
        raise ValueError(f"camera must be one of {CAMERAS!r}")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if step <= 0:
        raise ValueError("step must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")

    nusc = NuScenes(
        version=config.pipeline.dataset_version,
        dataroot=str(config.runtime.dataset_root),
        verbose=False,
    )
    scene_record = None
    if scene_name is None:
        scene_record = nusc.scene[0] if nusc.scene else None
    else:
        for candidate in nusc.scene:
            if candidate["name"] == scene_name or candidate["token"] == scene_name:
                scene_record = candidate
                break
    if scene_record is None:
        raise ValueError(f"scene {scene_name!r} was not found")

    output_path = output_path or (
        _reports_render_root(config)
        / f"scene_{scene_record['name']}_{camera.lower()}.gif"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header_font = _load_font(22)
    sub_font = _load_font(16)
    frames: list[Image.Image] = []
    sample_token = scene_record["first_sample_token"]
    sample_counter = 0
    emitted = 0
    while sample_token and emitted < max_frames:
        sample_record = nusc.get("sample", sample_token)
        if sample_counter % step == 0:
            sample_data = nusc.get("sample_data", sample_record["data"][camera])
            image_path = config.runtime.dataset_root / sample_data["filename"]
            frame = Image.open(image_path).convert("RGB")
            frame = ImageOps.fit(frame, (960, 540), method=Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(frame)
            draw.rounded_rectangle(
                [(18, 18), (440, 98)],
                radius=18,
                fill=(16, 17, 28),
                outline=(84, 75, 176),
                width=3,
            )
            draw.text(
                (34, 30),
                f"{scene_record['name']} · {camera}",
                fill=(231, 232, 243),
                font=header_font,
            )
            draw.text(
                (34, 60),
                (
                    f"frame {emitted + 1} · sample {sample_counter} · "
                    f"timestamp {sample_record['timestamp']}"
                ),
                fill=(181, 176, 207),
                font=sub_font,
            )
            frames.append(frame)
            emitted += 1
        sample_counter += 1
        sample_token = sample_record["next"] or None

    if not frames:
        raise ValueError(f"scene {scene_record['name']!r} did not yield frames for {camera}")

    frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=max(1, int(1000 / fps)),
        loop=0,
    )
    return RenderArtifact(
        artifact_type="scene_gif",
        output_path=output_path,
        metadata={
            "scene_name": scene_record["name"],
            "scene_token": scene_record["token"],
            "camera": camera,
            "frames": len(frames),
            "fps": fps,
            "step": step,
            "dataset_version": config.pipeline.dataset_version,
        },
    )
