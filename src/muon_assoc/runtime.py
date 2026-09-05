"""Runtime helpers shared by local CPU development and CUDA experiments."""

from __future__ import annotations

from typing import Literal

import torch

DeviceChoice = Literal["auto", "cpu", "cuda", "mps"]


def resolve_device(requested: DeviceChoice | str = "auto") -> torch.device:
    """Resolve a requested device and fail clearly when it is unavailable."""
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is false. "
            "Install the CUDA environment with './scripts/setup.sh cuda' and "
            "check that the NVIDIA driver is visible."
        )
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested, but it is unavailable on this machine.")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unknown device {requested!r}; expected auto, cpu, cuda, or mps.")
    return torch.device(requested)


def runtime_summary(device: torch.device) -> dict[str, object]:
    """Return reproducibility metadata without assuming CUDA exists."""
    summary: dict[str, object] = {
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        summary["gpu_name"] = torch.cuda.get_device_name(device)
        summary["gpu_capability"] = list(torch.cuda.get_device_capability(device))
    return summary

