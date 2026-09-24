"""Notebook-facing observables computed from saved times and alpha/beta states."""

import numpy as np
from scipy.special import expit

STATE_NAMES = ("alpha_s", "beta_s", "alpha_r", "beta_r")


def margins(state, subjects: int, relations: int):
    """Accept one state (4,) or an array (..., 4)."""
    a_s, b_s, a_r, b_r = np.moveaxis(np.asarray(state), -1, 0)
    return (
        a_s * b_s / np.sqrt(relations) - np.log(subjects - 1),
        a_r * b_r / np.sqrt(subjects) - np.log(relations - 1),
    )


def probabilities(state, subjects: int, relations: int):
    """Return subject mass, relation mass, and correct-answer probability."""
    q_s, q_r = margins(state, subjects, relations)
    p_s, p_r = expit(q_s), expit(q_r)
    return p_s, p_r, p_s * p_r


def errors(state, subjects: int, relations: int):
    """Stable component errors, including when probabilities round to one."""
    q_s, q_r = margins(state, subjects, relations)
    return expit(-q_s), expit(-q_r)


def loss(state, subjects: int, relations: int):
    q_s, q_r = margins(state, subjects, relations)
    return np.logaddexp(0, -q_s) + np.logaddexp(0, -q_r)


def wrong_subject_mass(state, subjects: int, relations: int):
    q_s, q_r = margins(state, subjects, relations)
    return expit(q_r) * expit(-q_s)


def margin_threshold(delta: float) -> float:
    if not np.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must be finite and between zero and one.")
    return float(np.log1p(-delta) - np.log(delta))


def learning_times(time, state, subjects: int, relations: int, delta: float = 0.1):
    """First crossings via linear margin interpolation; unreached times are NaN.

    Configured threshold crossings are included in saved trajectories by the
    solver. Arbitrary notebook thresholds have sampling/interpolation error.
    Returns (subject time, relation time, subject/relation ratio).
    """
    time, state = np.asarray(time), np.asarray(state)
    if (
        time.ndim != 1
        or len(time) == 0
        or state.shape != (len(time), 4)
        or not np.all(np.isfinite(time))
        or not np.all(np.isfinite(state))
        or np.any(np.diff(time) <= 0)
    ):
        raise ValueError("Expected finite states (N,4) and strictly increasing times (N,).")
    target = margin_threshold(delta)
    times = []
    for q in margins(state, subjects, relations):
        # Event-root roundoff may place a terminal sample just below its target.
        hits = np.flatnonzero(q >= target - 32 * np.finfo(float).eps * max(1, abs(target)))
        if not len(hits):
            times.append(float("nan"))
        elif hits[0] == 0:
            times.append(float(time[0]))
        else:
            i = hits[0]
            weight = np.clip((target - q[i - 1]) / (q[i] - q[i - 1]), 0, 1)
            times.append(float(time[i - 1] + weight * (time[i] - time[i - 1])))
    t_s, t_r = times
    ratio = t_s / t_r if t_r > 0 else float("nan")
    return t_s, t_r, ratio


def load_trajectory(path):
    """Load trajectory.npz as (time, state); config supplies S and R."""
    with np.load(path, allow_pickle=False) as data:
        return data["time"].copy(), np.column_stack([data[name] for name in STATE_NAMES])
