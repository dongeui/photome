"""
CLIP ViT-B/32 via open_clip_torch.

The image and text encoders share one in-process model to keep memory use lower
on NAS-class machines.
"""
from __future__ import annotations

import os
import struct

import numpy as np

_model = None
_preprocess = None
_tokenizer = None
_loading = False
_load_error: str | None = None


def _configure_torch_threads() -> None:
    import torch

    configured = os.environ.get("PHOTOME_TORCH_THREADS")
    index_workers = max(1, int(os.environ.get("PHOTOME_INDEX_WORKERS", "1")))
    cpu_count = os.cpu_count() or 1
    thread_count = int(configured) if configured else max(1, cpu_count // index_workers)
    torch.set_num_threads(thread_count)
    try:
        torch.set_num_interop_threads(max(1, min(2, thread_count)))
    except RuntimeError:
        pass


def is_ready() -> bool:
    return _model is not None and _preprocess is not None and _tokenizer is not None


def status() -> dict:
    return {
        "model_ready": is_ready(),
        "model_loading": _loading,
        "model_error": _load_error,
    }


def ensure_models() -> None:
    """Load CLIP model once. open_clip downloads and caches weights if needed."""
    global _model, _preprocess, _tokenizer, _loading, _load_error
    if is_ready():
        return

    _loading = True
    _load_error = None
    try:
        import open_clip
        _configure_torch_threads()

        print("[photome] Loading CLIP model (downloads ~350MB on first run) ...", flush=True)
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _model.eval()
        for p in _model.parameters():
            p.requires_grad_(False)
        _tokenizer = open_clip.get_tokenizer("ViT-B-32")
        print("[photome] Model ready.", flush=True)
    except Exception as exc:
        _load_error = str(exc)
        raise
    finally:
        _loading = False


def worker_init() -> None:
    """No-op kept for compatibility with earlier ProcessPool-based code."""
    pass


def load_text_encoder() -> None:
    """No-op: ensure_models() loads both image and text encoder state."""
    pass


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-12)


def embedding_to_bytes(v: np.ndarray) -> bytes:
    flat = _l2_normalize(v.flatten().astype(np.float32))
    return struct.pack(f"{len(flat)}f", *flat)


def embedding_from_bytes(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype=np.float32).copy()


def encode_image(image_path: str) -> bytes:
    """Encode image to embedding bytes."""
    if _model is None or _preprocess is None:
        raise RuntimeError("CLIP model is still loading")
    import torch
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    tensor = _preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = _model.encode_image(tensor)
    return embedding_to_bytes(features.detach().cpu().numpy())


def encode_text(query: str) -> bytes:
    """Encode text query to embedding bytes."""
    if _model is None or _tokenizer is None:
        raise RuntimeError("CLIP model is still loading")
    import torch

    tokens = _tokenizer([query])
    with torch.no_grad():
        features = _model.encode_text(tokens)
    return embedding_to_bytes(features.detach().cpu().numpy())
