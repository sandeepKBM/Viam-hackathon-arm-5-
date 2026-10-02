"""Load Moondream once. Photon first, Transformers on this Mac if Photon fails."""

from __future__ import annotations

import os

HF_ID = os.environ.get("MOONDREAM_HF_ID", "vikhyatk/moondream2")
HF_REV = os.environ.get("MOONDREAM_HF_REV", "2025-06-21")


class TransformersMoondream:
    def __init__(self, model) -> None:
        self._model = model

    def query(self, image, question: str) -> dict:
        return self._model.query(image, question)

    def detect(self, image, label: str) -> dict:
        return self._model.detect(image, label)

    def caption(self, image, **kwargs) -> dict:
        return self._model.caption(image, **kwargs)

    def encode_image(self, image):
        return self._model.encode_image(image)


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_photon(model_id: str):
    import moondream as md

    return md.photon(model_id)


def _patch_tied_weights() -> None:
    import torch.nn as nn

    orig = nn.Module.__getattr__

    def patched(self, name):
        if name == "all_tied_weights_keys":
            return {}
        return orig(self, name)

    nn.Module.__getattr__ = patched


def load_transformers():
    import torch
    from transformers import AutoModelForCausalLM

    _patch_tied_weights()
    device = _device()
    print(f"loading {HF_ID}@{HF_REV} via transformers on {device}…", flush=True)
    kwargs = {
        "revision": HF_REV,
        "trust_remote_code": True,
        "dtype": torch.float32 if device == "cpu" else torch.float16,
    }
    if device != "cpu":
        kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(HF_ID, **kwargs)
    if hasattr(model, "post_init"):
        try:
            model.post_init()
        except Exception:
            pass
    if device == "cpu":
        model = model.to(device)
    model.eval()
    try:
        device_name = str(next(model.parameters()).device)
    except Exception:
        device_name = device
    print(f"transformers device={device_name}", flush=True)
    return TransformersMoondream(model)


def load_moondream(model_id: str):
    # Photon is preferred, but current Metal wheels crash on this Mac
    # (k_scale_tensor FP32 scalar). Default stays transformers until that is fixed.
    backend = os.environ.get("MOONDREAM_BACKEND", "transformers").strip().lower()
    if backend == "photon":
        model = load_photon(model_id)
        print(f"{model_id} ready (photon)", flush=True)
        return model, "photon"
    if backend == "auto":
        try:
            model = load_photon(model_id)
            print(f"{model_id} ready (photon)", flush=True)
            return model, "photon"
        except Exception as exc:
            print(f"photon failed ({exc}); falling back to transformers", flush=True)
    model = load_transformers()
    print(f"{HF_ID} ready (transformers)", flush=True)
    return model, "transformers"
