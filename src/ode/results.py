"""Load a named run for notebooks without selecting dates or individual files."""

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from ode.observables import load_trajectory


def condition_names(dynamics: str) -> list[str]:
    """Share the runner's Hydra presets with notebook curve selection."""
    if dynamics not in {"main_text", "all", "gf_spectral"}:
        raise ValueError("dynamics must be 'main_text', 'all', or 'gf_spectral'.")
    resource = files("ode.conf").joinpath("dynamics", f"{dynamics}.yaml")
    return list(OmegaConf.create(resource.read_text()).conditions)


def condition_label(condition: str) -> str:
    if condition == "gf":
        return "GF"
    if condition == "spectral":
        return "Spectral GF"
    return "Sign GF (" + ",".join(condition.removeprefix("sign_").upper()) + ")"


@dataclass
class RecordedCase:
    config: DictConfig
    time: np.ndarray
    state: np.ndarray


@dataclass
class RecordedRun:
    snapshot: Path
    config: DictConfig
    cases: dict[str, RecordedCase]


def load_run(
    root: str | Path = "runs/experiment1",
    experiment: str = "pilot",
    dynamics: str = "main_text",
    plot_dynamics: str | None = None,
    pair: tuple[int, int] | None = None,
    snapshot: Path | None = None,
) -> RecordedRun:
    """Load selected curves from one completed, immutable snapshot.

    plot_dynamics='main_text' can select five curves from dynamics='all'.
    Cardinality runs require an explicit (S,R) pair; a single-pair pilot does not.
    Pass a previously loaded snapshot to keep a multi-pair analysis on that run.
    """
    selected = condition_names(plot_dynamics or dynamics)
    named = Path(root) / experiment / dynamics
    if snapshot is None and not named.exists():
        raise FileNotFoundError(
            f"No completed run at {named}. Run: uv run muon-ode "
            f"experiment={experiment} dynamics={dynamics}"
        )
    snapshot = (snapshot or named).resolve(strict=True)
    cfg = OmegaConf.load(snapshot / "config.yaml")
    missing = set(selected) - set(cfg.dynamics.conditions)
    if missing:
        raise ValueError(
            f"This run lacks {sorted(missing)}. Run 'uv run muon-ode "
            f"experiment={experiment} dynamics=all' and select dynamics='all' to load it."
        )
    if pair is None:
        if cfg.experiment.mode != "pairs" or len(cfg.experiment.pairs) != 1:
            raise ValueError("Select pair=(subjects, relations) for a multi-pair run.")
        pair = tuple(cfg.experiment.pairs[0])
    cases = {}
    for condition in selected:
        folder = snapshot / f"S{pair[0]}_R{pair[1]}" / condition
        if not folder.is_dir():
            raise ValueError(f"Pair {pair} is not present in this run.")
        status = json.loads((folder / "status.json").read_text())
        if status["status"] != "target_reached":
            raise ValueError(f"Incomplete case: {folder}: {status['status']}")
        time, state = load_trajectory(folder / "trajectory.npz")
        cases[condition] = RecordedCase(OmegaConf.load(folder / "config.yaml"), time, state)
    return RecordedRun(snapshot, cfg, cases)
