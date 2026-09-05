"""Hydra-driven environment and autograd smoke test."""

from __future__ import annotations

import json

import hydra
import torch
from omegaconf import DictConfig

from muon_assoc.runtime import resolve_device, runtime_summary


@hydra.main(version_base="1.3", config_path="pkg://muon_assoc.conf", config_name="smoke")
def main(cfg: DictConfig) -> None:
    device = resolve_device(cfg.device)
    torch.manual_seed(int(cfg.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(cfg.seed))

    model = torch.nn.Linear(int(cfg.input_dim), int(cfg.output_dim), bias=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    inputs = torch.randn(int(cfg.batch_size), int(cfg.input_dim), device=device)
    targets = torch.randint(int(cfg.output_dim), (int(cfg.batch_size),), device=device)

    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(inputs), targets)
    loss.backward()
    optimizer.step()

    summary = runtime_summary(device)
    summary.update(
        {
            "hydra_version": hydra.__version__,
            "loss": float(loss.detach().cpu()),
            "finite": bool(torch.isfinite(loss).item()),
        }
    )
    if not summary["finite"]:
        raise RuntimeError("The smoke-test loss was not finite.")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
