"""Load TFT checkpoints that were saved on CUDA onto CPU/MPS hosts."""
from __future__ import annotations

import os
import torch
from pytorch_forecasting import TemporalFusionTransformer


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _force_metric_cpu(obj) -> None:
    if hasattr(obj, "_device") and "cuda" in str(getattr(obj, "_device")):
        obj._device = torch.device("cpu")
    if isinstance(obj, torch.nn.Module):
        for child in obj.children():
            _force_metric_cpu(child)


def load_tft_checkpoint(ckpt_path: str, device: str | None = None):
    """
    Load a TemporalFusionTransformer checkpoint portably.

    CUDA-trained checkpoints pickle torchmetrics with `_device=cuda:0`.
    `map_location="cpu"` remaps tensors but not `_device`, which crashes on
    non-CUDA builds. Remap metrics to CPU before load_from_checkpoint.
    """
    device = device or pick_device()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for bag_key in ("hyper_parameters", "__special_save__"):
        bag = ckpt.get(bag_key) or {}
        for key in ("loss", "logging_metrics"):
            if key in bag:
                _force_metric_cpu(bag[key])

    tmp_path = ckpt_path + ".portable.ckpt"
    torch.save(ckpt, tmp_path)
    try:
        model = TemporalFusionTransformer.load_from_checkpoint(
            tmp_path, map_location=device
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return model.to(device).eval()
