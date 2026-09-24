"""Storage, stopping, and notebook interface checks (no matrix validation)."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from ode.dynamics import Problem, vector_field
from ode.observables import learning_times, load_trajectory, probabilities
from ode.results import condition_names, load_run
from ode.run import run
from ode.solve import SolverSettings, solve


@pytest.mark.parametrize("flow", ["gf", "spectral", "sign"])
def test_termination_and_cap(flow):
    problem = Problem(16, 4)
    result = solve(problem, flow, delta=0.1)
    assert result.status == "target_reached"
    assert np.all(np.diff(result.time) > 0)
    t_s, t_r, ratio = learning_times(result.time, result.state, 16, 4)
    assert np.isfinite([t_s, t_r, ratio]).all()
    assert max(t_s, t_r) == pytest.approx(result.time[-1], rel=1e-10)
    capped = solve(problem, flow, delta=0.1, settings=SolverSettings(max_time=0.1))
    assert capped.status == "max_time"
    assert capped.time[-1] == pytest.approx(0.1)
    assert np.isnan(learning_times(capped.time, capped.state, 16, 4)).all()


def test_notebook_metrics():
    # At S=R=2, both margins are t, crossing p=0.9 at log(9).
    time = np.array([0.0, 1.0, 3.0])
    state = np.column_stack([np.full(3, np.sqrt(2)), time] * 2)
    assert learning_times(time, state, 2, 2) == pytest.approx((np.log(9), np.log(9), 1))
    p_s, p_r, correct = probabilities(state, 2, 2)
    assert correct[0] == pytest.approx(0.25)
    assert np.allclose(correct, p_s * p_r)


def test_spectral_large_margins_remain_finite():
    rhs = vector_field(Problem(16, 16), "spectral")
    assert np.isfinite(rhs(0, np.full(4, 1e4))).all()


def suite_config():
    return OmegaConf.create(
        {
            "experiment": {"name": "pilot", "mode": "pairs", "pairs": [[16, 4]], "delta": 0.1},
            "dynamics": {
                "name": "main_text",
                "conditions": ["gf", "spectral", "sign_ooo", "sign_hoh", "sign_ohh"],
            },
            "initialization": {"alpha": 0.01, "beta": 0.01},
            "solver": {"samples_per_segment": 30},
        }
    )


def test_named_run_and_notebook_loading(tmp_path):
    cfg = suite_config()
    named = run(cfg, tmp_path)
    first = load_run(tmp_path)
    assert named == tmp_path / "pilot/main_text"
    assert len(first.cases) == 5
    path = named / "S16_R4/sign_ohh/trajectory.npz"
    with np.load(path) as data:
        assert set(data.files) == {"time", "alpha_s", "beta_s", "alpha_r", "beta_r"}
    time, state = load_trajectory(path)
    assert state.shape == (len(time), 4)
    assert learning_times(time, state, 16, 4)[2] < 1
    cfg.experiment.delta = 0.01
    assert run(cfg, tmp_path) == named
    second = load_run(tmp_path)
    assert second.snapshot != first.snapshot
    assert first.snapshot.is_dir()
    assert second.config.experiment.delta == 0.01
    # A loaded notebook run stays attached to its original complete snapshot.
    assert first.config.experiment.delta == 0.1
    with pytest.raises(ValueError, match="lacks"):
        load_run(tmp_path, plot_dynamics="all")


def test_incomplete_run_does_not_replace_completed_run(tmp_path):
    cfg = suite_config()
    named = run(cfg, tmp_path)
    completed = named.resolve()
    cfg.solver.max_time = 0.1
    with pytest.raises(RuntimeError, match="Time cap"):
        run(cfg, tmp_path)
    assert named.resolve() == completed
    assert load_run(tmp_path).snapshot == completed


def test_missing_notebook_run_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="uv run muon-ode"):
        load_run(tmp_path)


def test_appendix_selection_and_pinned_snapshot(tmp_path):
    cfg = suite_config()
    cfg.dynamics.name = "gf_spectral"
    cfg.dynamics.conditions = condition_names("gf_spectral")
    cfg.experiment.pairs = [[4, 4], [8, 4]]
    run(cfg, tmp_path)
    first = load_run(tmp_path, dynamics="gf_spectral", pair=(4, 4))
    assert set(first.cases) == {"gf", "spectral"}
    cfg.experiment.delta = 0.01
    run(cfg, tmp_path)
    pinned = load_run(tmp_path, dynamics="gf_spectral", pair=(8, 4), snapshot=first.snapshot)
    assert pinned.config.experiment.delta == 0.1
    assert pinned.snapshot == first.snapshot
