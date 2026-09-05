from __future__ import annotations

import pytest
import torch

from muon_assoc.runtime import resolve_device, runtime_summary


def test_cpu_is_always_available() -> None:
    device = resolve_device("cpu")
    assert device.type == "cpu"
    assert runtime_summary(device)["cuda_available"] == torch.cuda.is_available()


def test_auto_returns_an_available_device() -> None:
    device = resolve_device("auto")
    assert device.type in {"cpu", "cuda", "mps"}


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("quantum")

