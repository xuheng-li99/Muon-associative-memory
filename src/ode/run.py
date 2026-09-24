"""Hydra runner: record raw trajectories for later notebook analysis."""

import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from ode.dynamics import Problem
from ode.observables import STATE_NAMES, margin_threshold
from ode.solve import SolverSettings, solve


def _git_info():
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def run(cfg: DictConfig, output: Path) -> Path:
    """Record an immutable snapshot and atomically update the named run on success.

    A notebook resolves the named link once, so rerunning cannot mix trajectories
    from two invocations. Failed/capped snapshots never replace a complete run.
    """
    settings = SolverSettings(**OmegaConf.to_container(cfg.solver, resolve=True))
    margin_threshold(cfg.experiment.delta)
    for name in (cfg.experiment.name, cfg.dynamics.name):
        if not name or name in {".", ".."} or Path(name).name != name:
            raise ValueError("Experiment and dynamics names must be single path components.")
    if cfg.experiment.mode == "grid":
        pairs = [(s, r) for s in cfg.experiment.sizes for r in cfg.experiment.sizes if s >= r]
    elif cfg.experiment.mode == "pairs":
        pairs = [tuple(pair) for pair in cfg.experiment.pairs]
    else:
        raise ValueError(f"Unknown experiment mode: {cfg.experiment.mode}")
    if not pairs or not cfg.dynamics.conditions:
        raise ValueError("A suite needs at least one problem and flow condition.")
    output = output.resolve()
    snapshot = output / ".snapshots" / f"{cfg.experiment.name}-{cfg.dynamics.name}-{uuid4().hex}"
    snapshot.mkdir(parents=True, exist_ok=False)
    with (snapshot / "config.yaml").open("x") as handle:
        handle.write(OmegaConf.to_yaml(cfg, resolve=True))
    metadata = {
        "schema_version": 2,
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("numpy", "scipy", "hydra-core")},
        "git": _git_info(),
        "state_order": list(STATE_NAMES),
        "time_convention": "native flow time; learning-rate multiplier 1",
    }
    (snapshot / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    failures = []
    incomplete = []
    for subjects, relations in pairs:
        problem = Problem(subjects, relations, cfg.initialization.alpha, cfg.initialization.beta)
        for condition in cfg.dynamics.conditions:
            if condition in {"gf", "spectral"}:
                flow, embedding = condition, "OOO"
            elif condition.startswith("sign_"):
                flow, embedding = "sign", condition.removeprefix("sign_").upper()
            else:
                raise ValueError(f"Unknown condition: {condition}")
            case_dir = snapshot / f"S{subjects}_R{relations}" / condition
            case_dir.mkdir(parents=True, exist_ok=False)
            case_config = {
                "problem": asdict(problem),
                "flow": flow,
                "embedding": embedding,
                "delta": cfg.experiment.delta,
                "solver": asdict(settings),
            }
            OmegaConf.save(OmegaConf.create(case_config), case_dir / "config.yaml")
            started = time.perf_counter()
            try:
                result = solve(problem, flow, embedding, cfg.experiment.delta, settings)
                np.savez_compressed(
                    case_dir / "trajectory.npz",
                    time=result.time,
                    **dict(zip(STATE_NAMES, result.state.T, strict=True)),
                )
                status = {
                    "status": result.status,
                    "message": result.message,
                    "nfev": result.nfev,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            except (ValueError, RuntimeError, FloatingPointError, OSError) as exc:
                status = {
                    "status": "failed",
                    "message": str(exc),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            (case_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n")
            print(f"S={subjects} R={relations} {condition}: {status['status']}", flush=True)
            if status["status"] == "failed":
                failures.append(str(case_dir))
            elif status["status"] != "target_reached":
                incomplete.append(str(case_dir))
    if failures:
        raise RuntimeError(f"Failed cases (see status.json): {failures}")
    if incomplete:
        raise RuntimeError(f"Time cap reached; named run was not updated. Inspect: {snapshot}")
    named = output / cfg.experiment.name / cfg.dynamics.name
    named.parent.mkdir(parents=True, exist_ok=True)
    temporary = named.with_name(f".{named.name}-{uuid4().hex}")
    temporary.symlink_to(os.path.relpath(snapshot, named.parent), target_is_directory=True)
    try:
        temporary.replace(named)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Notebook run: {named}\nSnapshot: {snapshot}", flush=True)
    return named


@hydra.main(version_base="1.3", config_path="pkg://ode.conf", config_name="experiment1")
def main(cfg: DictConfig):
    run(cfg, Path(hydra.utils.to_absolute_path(cfg.output_root)))


if __name__ == "__main__":
    main()
