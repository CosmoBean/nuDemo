# ruff: noqa: E501

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

VECTOR_DIM = 512
DEFAULT_MODALITY_WEIGHTS = {
    "lexical": 0.45,
    "fused": 0.85,
    "image": 0.20,
    "lidar": 0.20,
    "radar": 0.10,
    "metadata": 0.25,
    "positive": 0.70,
    "negative": 0.45,
}
MODALITY_PRESETS = {
    "balanced": DEFAULT_MODALITY_WEIGHTS,
    "image-heavy": {
        **DEFAULT_MODALITY_WEIGHTS,
        "image": 0.35,
        "lidar": 0.12,
        "radar": 0.08,
        "metadata": 0.20,
        "fused": 0.95,
    },
    "lidar-heavy": {
        **DEFAULT_MODALITY_WEIGHTS,
        "image": 0.10,
        "lidar": 0.38,
        "radar": 0.16,
        "metadata": 0.18,
        "fused": 0.95,
    },
    "metadata-heavy": {
        **DEFAULT_MODALITY_WEIGHTS,
        "image": 0.10,
        "lidar": 0.12,
        "radar": 0.06,
        "metadata": 0.42,
        "fused": 0.90,
    },
}

_TOKEN_PATTERN = re.compile(r"[a-z0-9_.-]+")


def normalize_modality_weights(weights: dict[str, float] | None) -> dict[str, float]:
    normalized = dict(DEFAULT_MODALITY_WEIGHTS)
    for key, value in (weights or {}).items():
        if key not in normalized:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        normalized[key] = max(0.0, min(numeric, 5.0))
    return normalized


def resolve_modality_weights(
    preset: str | None = None,
    overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    base = MODALITY_PRESETS.get(preset or "balanced", DEFAULT_MODALITY_WEIGHTS)
    merged = {**base, **(overrides or {})}
    return normalize_modality_weights(merged)


def build_metadata_text(sample_record: dict[str, Any]) -> str:
    annotations = sample_record.get("annotations") or []
    categories = [annotation.get("category", "") for annotation in annotations if annotation.get("category")]
    category_text = ", ".join(sorted(set(categories))[:18]) or "no annotations"
    scene_name = sample_record.get("scene_name") or sample_record.get("scene_token") or "unknown scene"
    location = sample_record.get("location") or "unknown location"
    parts = [
        f"scene {scene_name}",
        f"location {location}",
        f"sample token {sample_record.get('token', '')}",
        f"{sample_record.get('num_annotations', 0)} annotations",
        f"{sample_record.get('num_lidar_points', 0)} lidar points",
        f"categories {category_text}",
    ]
    return ". ".join(part for part in parts if part).strip()


@dataclass(slots=True)
class EncodedSampleVectors:
    image_vec: np.ndarray
    lidar_vec: np.ndarray
    radar_vec: np.ndarray
    metadata_vec: np.ndarray
    fused_vec: np.ndarray
    encoder_backend: str
    encoder_model: str
    has_image: bool
    has_lidar: bool
    has_radar: bool
    has_metadata: bool

    def as_document_fields(self) -> dict[str, object]:
        return {
            "image_vec": self.image_vec.tolist(),
            "lidar_vec": self.lidar_vec.tolist(),
            "radar_vec": self.radar_vec.tolist(),
            "metadata_vec": self.metadata_vec.tolist(),
            "fused_vec": self.fused_vec.tolist(),
            "encoder_backend": self.encoder_backend,
            "encoder_model": self.encoder_model,
            "has_image": self.has_image,
            "has_lidar": self.has_lidar,
            "has_radar": self.has_radar,
            "has_metadata": self.has_metadata,
        }


class MultimodalEmbeddingEncoder:
    def __init__(
        self,
        *,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self._backend = "fallback"
        self._ready = False
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._torch = None

    @property
    def backend(self) -> str:
        self._ensure_backend()
        return self._backend

    def encode_text(self, text: str) -> np.ndarray:
        self._ensure_backend()
        if self._backend == "openclip":
            tokens = self._tokenizer([text]).to(self.device)
            with self._torch.no_grad():
                vector = self._model.encode_text(tokens)
            return _to_numpy_unit(vector[0])
        return _hashed_text_vector(text)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        self._ensure_backend()
        if self._backend == "openclip":
            tensor = self._preprocess(image).unsqueeze(0).to(self.device)
            with self._torch.no_grad():
                vector = self._model.encode_image(tensor)
            return _to_numpy_unit(vector[0])
        return _fallback_image_vector(image)

    def encode_image_bytes(self, payload: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        return self.encode_image(image)

    def encode_sample_payloads(
        self,
        *,
        camera_payloads: dict[str, bytes],
        lidar_payload: bytes | None,
        radar_payloads: dict[str, bytes],
        metadata_text: str,
    ) -> EncodedSampleVectors:
        image_vecs = [self.encode_image_bytes(payload) for payload in camera_payloads.values() if payload]
        image_vec = _average_vectors(image_vecs)

        lidar_vec = _zero_vector()
        if lidar_payload:
            try:
                lidar_points = np.load(io.BytesIO(lidar_payload), allow_pickle=False)
                lidar_vec = self.encode_image(_lidar_to_bev_image(lidar_points))
            except Exception:
                lidar_vec = _zero_vector()

        radar_vec = _zero_vector()
        if radar_payloads:
            radar_arrays: list[np.ndarray] = []
            for payload in radar_payloads.values():
                if not payload:
                    continue
                try:
                    radar_arrays.append(np.load(io.BytesIO(payload), allow_pickle=False))
                except Exception:
                    continue
            if radar_arrays:
                radar_vec = self.encode_image(_radar_to_bev_image(radar_arrays))

        metadata_vec = self.encode_text(metadata_text) if metadata_text else _zero_vector()

        fused_vec = _average_vectors(
            [vector for vector in (image_vec, lidar_vec, radar_vec, metadata_vec) if np.linalg.norm(vector) > 0]
        )

        return EncodedSampleVectors(
            image_vec=image_vec,
            lidar_vec=lidar_vec,
            radar_vec=radar_vec,
            metadata_vec=metadata_vec,
            fused_vec=fused_vec,
            encoder_backend=self.backend,
            encoder_model=self.model_name if self.backend == "openclip" else "fallback-hash",
            has_image=bool(image_vecs),
            has_lidar=bool(lidar_payload),
            has_radar=bool(radar_payloads),
            has_metadata=bool(metadata_text),
        )

    def _ensure_backend(self) -> None:
        if self._ready:
            return
        self._ready = True
        try:
            import open_clip
            import torch

            self._torch = torch
            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device,
            )
            model.eval()
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            self._backend = "openclip"
        except Exception:
            self._backend = "fallback"
            self._model = None
            self._preprocess = None
            self._tokenizer = None
            self._torch = None


def _zero_vector() -> np.ndarray:
    return np.zeros(VECTOR_DIM, dtype=np.float32)


def _average_vectors(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        return _zero_vector()
    stacked = np.stack(vectors, axis=0).astype(np.float32)
    averaged = stacked.mean(axis=0)
    return _l2_normalize(averaged)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return _zero_vector()
    return (vector / norm).astype(np.float32)


def _to_numpy_unit(vector: Any) -> np.ndarray:
    return _l2_normalize(vector.detach().cpu().numpy().astype(np.float32))


def _hashed_text_vector(text: str) -> np.ndarray:
    tokens = _TOKEN_PATTERN.findall((text or "").lower())
    vector = _zero_vector()
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for offset in range(0, min(len(digest), 16), 2):
            idx = int.from_bytes(digest[offset:offset + 2], "little") % VECTOR_DIM
            sign = 1.0 if digest[offset] % 2 == 0 else -1.0
            vector[idx] += sign
    return _l2_normalize(vector)


def _fallback_image_vector(image: Image.Image) -> np.ndarray:
    resized = image.resize((32, 32)).convert("RGB")
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    features = np.concatenate(
        [
            pixels.mean(axis=(0, 1)),
            pixels.std(axis=(0, 1)),
            pixels[::4, ::4].reshape(-1),
        ]
    ).astype(np.float32)
    vector = _zero_vector()
    vector[: min(len(features), VECTOR_DIM)] = features[:VECTOR_DIM]
    return _l2_normalize(vector)


def _lidar_to_bev_image(points: np.ndarray, size: int = 224) -> Image.Image:
    return _points_to_bev_image(points[:, :3] if points.ndim == 2 and points.shape[1] >= 3 else np.zeros((0, 3), dtype=np.float32), size=size, point_radius=1)


def _radar_to_bev_image(radars: list[np.ndarray], size: int = 224) -> Image.Image:
    if not radars:
        return Image.new("RGB", (size, size), "#09090e")
    merged = np.concatenate(
        [radar[:, :3] for radar in radars if radar.ndim == 2 and radar.shape[1] >= 3],
        axis=0,
    ) if any(radar.ndim == 2 and radar.shape[1] >= 3 for radar in radars) else np.zeros((0, 3), dtype=np.float32)
    return _points_to_bev_image(merged, size=size, point_radius=2)


def _points_to_bev_image(points: np.ndarray, *, size: int, point_radius: int) -> Image.Image:
    image = Image.new("RGB", (size, size), "#09090e")
    if points.shape[0] == 0:
        return image
    draw = ImageDraw.Draw(image)
    xy = points[:, :2].astype(np.float32)
    max_extent = float(np.max(np.abs(xy))) or 1.0
    padded = max_extent * 1.05
    normalized_x = ((xy[:, 0] + padded) / (2 * padded)) * (size - 1)
    normalized_y = ((xy[:, 1] + padded) / (2 * padded)) * (size - 1)
    heights = points[:, 2].astype(np.float32) if points.shape[1] > 2 else np.zeros(points.shape[0], dtype=np.float32)
    z_min = float(heights.min()) if len(heights) else 0.0
    z_span = float(heights.max() - z_min) or 1.0
    for idx in range(points.shape[0]):
        hue = int(220 - (((float(heights[idx]) - z_min) / z_span) * 160))
        color = (max(40, hue), 100, min(255, 180 + point_radius * 10))
        x = int(np.clip(normalized_x[idx], 0, size - 1))
        y = int(np.clip(size - 1 - normalized_y[idx], 0, size - 1))
        draw.ellipse(
            (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
            fill=color,
        )
    return image
