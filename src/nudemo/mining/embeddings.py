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

    def encode_texts(self, texts: list[str], *, chunk_size: int = 32) -> list[np.ndarray]:
        if not texts:
            return []
        self._ensure_backend()
        if self._backend != "openclip":
            return [_hashed_text_vector(text) for text in texts]

        vectors: list[np.ndarray] = []
        for offset in range(0, len(texts), max(1, chunk_size)):
            chunk = texts[offset : offset + max(1, chunk_size)]
            tokens = self._tokenizer(chunk).to(self.device)
            with self._torch.no_grad():
                encoded = self._model.encode_text(tokens)
            vectors.extend(
                _to_numpy_unit(vector) for vector in encoded
            )
        return vectors

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

    def encode_image_list(
        self,
        images: list[Image.Image],
        *,
        chunk_size: int = 32,
    ) -> list[np.ndarray]:
        if not images:
            return []
        self._ensure_backend()
        if self._backend == "openclip":
            vectors: list[np.ndarray] = []
            effective_chunk = max(1, chunk_size)
            for offset in range(0, len(images), effective_chunk):
                chunk = images[offset : offset + effective_chunk]
                batch = self._torch.stack(
                    [self._preprocess(image.convert("RGB")) for image in chunk]
                ).to(self.device)
                with self._torch.no_grad():
                    encoded = self._model.encode_image(batch)
                vectors.extend(_to_numpy_unit(vector) for vector in encoded)
            return vectors
        return [_fallback_image_vector(image) for image in images]

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return _zero_vector()
        return _average_vectors(self.encode_image_list(images))

    def encode_sample_payloads(
        self,
        *,
        camera_payloads: dict[str, bytes],
        lidar_payload: bytes | None,
        radar_payloads: dict[str, bytes],
        metadata_text: str,
    ) -> EncodedSampleVectors:
        camera_images = [
            Image.open(io.BytesIO(payload)).convert("RGB")
            for payload in camera_payloads.values()
            if payload
        ]
        image_vec = self.encode_images(camera_images)

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
            has_image=bool(camera_images),
            has_lidar=bool(lidar_payload),
            has_radar=bool(radar_payloads),
            has_metadata=bool(metadata_text),
        )

    def encode_sample_payload_batch(
        self,
        payloads: list[dict[str, Any]],
    ) -> list[EncodedSampleVectors]:
        if not payloads:
            return []

        camera_groups: list[list[Image.Image]] = []
        grouped_camera_lengths: list[int] = []
        flat_camera_images: list[Image.Image] = []
        lidar_images: list[Image.Image | None] = []
        radar_images: list[Image.Image | None] = []
        metadata_texts: list[str] = []

        for payload in payloads:
            cameras = [
                Image.open(io.BytesIO(camera_payload)).convert("RGB")
                for camera_payload in (payload.get("camera_payloads") or {}).values()
                if camera_payload
            ]
            camera_groups.append(cameras)
            grouped_camera_lengths.append(len(cameras))
            flat_camera_images.extend(cameras)

            lidar_payload = payload.get("lidar_payload")
            if lidar_payload:
                try:
                    lidar_points = np.load(io.BytesIO(lidar_payload), allow_pickle=False)
                    lidar_images.append(_lidar_to_bev_image(lidar_points))
                except Exception:
                    lidar_images.append(None)
            else:
                lidar_images.append(None)

            radar_payloads = payload.get("radar_payloads") or {}
            radar_arrays: list[np.ndarray] = []
            for radar_payload in radar_payloads.values():
                if not radar_payload:
                    continue
                try:
                    radar_arrays.append(np.load(io.BytesIO(radar_payload), allow_pickle=False))
                except Exception:
                    continue
            radar_images.append(_radar_to_bev_image(radar_arrays) if radar_arrays else None)
            metadata_texts.append(str(payload.get("metadata_text") or ""))

        flat_camera_vectors = self.encode_image_list(flat_camera_images)
        camera_vectors: list[np.ndarray] = []
        cursor = 0
        for length in grouped_camera_lengths:
            group_vectors = flat_camera_vectors[cursor : cursor + length]
            cursor += length
            camera_vectors.append(_average_vectors(group_vectors))

        lidar_vectors = _encode_optional_images(self, lidar_images)
        radar_vectors = _encode_optional_images(self, radar_images)
        metadata_vectors = [
            vector if text else _zero_vector()
            for text, vector in zip(metadata_texts, self.encode_texts(metadata_texts), strict=False)
        ]

        results: list[EncodedSampleVectors] = []
        for idx in range(len(payloads)):
            image_vec = camera_vectors[idx]
            lidar_vec = lidar_vectors[idx]
            radar_vec = radar_vectors[idx]
            metadata_vec = metadata_vectors[idx]
            fused_vec = _average_vectors(
                [
                    vector
                    for vector in (image_vec, lidar_vec, radar_vec, metadata_vec)
                    if np.linalg.norm(vector) > 0
                ]
            )
            results.append(
                EncodedSampleVectors(
                    image_vec=image_vec,
                    lidar_vec=lidar_vec,
                    radar_vec=radar_vec,
                    metadata_vec=metadata_vec,
                    fused_vec=fused_vec,
                    encoder_backend=self.backend,
                    encoder_model=(
                        self.model_name if self.backend == "openclip" else "fallback-hash"
                    ),
                    has_image=bool(grouped_camera_lengths[idx]),
                    has_lidar=lidar_images[idx] is not None,
                    has_radar=radar_images[idx] is not None,
                    has_metadata=bool(metadata_texts[idx]),
                )
            )
        return results

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


def _encode_optional_images(
    encoder: MultimodalEmbeddingEncoder,
    images: list[Image.Image | None],
) -> list[np.ndarray]:
    actual_images = [image for image in images if image is not None]
    actual_vectors = encoder.encode_image_list(actual_images)
    result: list[np.ndarray] = []
    cursor = 0
    for image in images:
        if image is None:
            result.append(_zero_vector())
            continue
        result.append(actual_vectors[cursor])
        cursor += 1
    return result


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
