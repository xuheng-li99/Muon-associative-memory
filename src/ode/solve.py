"""Integrate reduced flows and retain only times and parameter trajectories."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ode.dynamics import Problem, sign_rates, vector_field
from ode.observables import margin_threshold, margins


@dataclass(frozen=True)
class SolverSettings:
    method: str = "DOP853"
    rtol: float = 1e-9
    atol: float = 1e-12
    initial_horizon: float = 100.0
    max_time: float = 1e10
    horizon_growth: float = 10.0
    samples_per_segment: int = 2000

    def __post_init__(self):
        if not all(
            np.isfinite(x) and x > 0
            for x in (self.rtol, self.atol, self.initial_horizon, self.max_time)
        ):
            raise ValueError("Tolerances and time limits must be finite and positive.")
        if not np.isfinite(self.horizon_growth) or self.horizon_growth <= 1:
            raise ValueError("horizon_growth must be finite and greater than one.")
        if not isinstance(self.samples_per_segment, int) or self.samples_per_segment < 2:
            raise ValueError("samples_per_segment must be an integer >= 2.")


@dataclass
class Solution:
    time: np.ndarray
    state: np.ndarray
    status: str
    message: str
    nfev: int


def _sign_crossings(problem, rates, delta):
    """Positive roots of the quadratic alpha(t) beta(t) threshold equation."""
    roots = []
    target = margin_threshold(delta)
    for offset, count, other in (
        (0, problem.subjects, problem.relations),
        (2, problem.relations, problem.subjects),
    ):
        a, b = rates[offset : offset + 2]
        gap = np.sqrt(other) * (target + np.log(count - 1)) - problem.alpha0 * problem.beta0
        linear = a * problem.beta0 + b * problem.alpha0
        roots.append(0.0 if gap <= 0 else 2 * gap / (linear + np.sqrt(linear**2 + 4 * a * b * gap)))
    return np.asarray(roots)


def solve(
    problem: Problem,
    flow: str,
    embedding: str = "OOO",
    delta: float = 0.01,
    settings: SolverSettings | None = None,
) -> Solution:
    """Stop when both p_S and p_R reach 1-delta, or at the time cap.

    Root locations are saved as ordinary time/state samples, not derived metrics.
    Accepted solver steps and dense samples preserve the rest of each trajectory.
    """
    settings = settings or SolverSettings()
    target = margin_threshold(delta)
    rhs = vector_field(problem, flow, embedding)
    initial = problem.initial_state
    if flow == "sign":
        rates = sign_rates(problem, embedding)
        crossings = _sign_crossings(problem, rates, delta)
        finish = min(float(np.max(crossings)), settings.max_time)
        # Match numerical flows' logarithmic coverage while keeping exact roots.
        grid = np.expm1(np.linspace(0, np.log1p(finish), settings.samples_per_segment))
        time = np.unique(np.r_[0, np.clip(grid, 0, finish), crossings[crossings <= finish], finish])
        reached = np.max(crossings) <= settings.max_time
        return Solution(
            time,
            initial + time[:, None] * rates,
            "target_reached" if reached else "max_time",
            "Analytic Sign GF trajectory.",
            0,
        )

    if min(margins(initial, problem.subjects, problem.relations)) >= target:
        return Solution(
            np.array([0.0]),
            initial[None, :],
            "target_reached",
            "Initial state already reaches the target.",
            0,
        )

    events = []
    for component in (0, 1):

        def crossing(t, y, component=component):
            return margins(y, problem.subjects, problem.relations)[component] - target

        crossing.direction = 1
        events.append(crossing)

    def stop(t, y):
        return min(margins(y, problem.subjects, problem.relations)) - target

    stop.terminal = True
    stop.direction = 1
    events.append(stop)

    start, end, state, nfev = 0.0, min(settings.initial_horizon, settings.max_time), initial, 0
    all_time, all_state = [], []
    status, message = "max_time", "Integration cap reached before both components met the target."
    while start < settings.max_time:
        result = solve_ivp(
            rhs,
            (start, end),
            state,
            method=settings.method,
            rtol=settings.rtol,
            atol=settings.atol,
            dense_output=True,
            events=events,
        )
        nfev += result.nfev
        finish = float(result.t[-1])
        grid = np.expm1(
            np.linspace(np.log1p(start), np.log1p(finish), settings.samples_per_segment)
        )
        event_times = np.concatenate(result.t_events)
        time = np.unique(np.r_[result.t, np.clip(grid, start, finish), event_times])
        if all_time:
            time = time[time > start]
        all_time.append(time)
        all_state.append(result.sol(time).T if len(time) else np.empty((0, 4)))
        if not result.success:
            status, message = "failed", result.message
            break
        if result.status == 1:
            status, message = "target_reached", "Both components reached 1-delta."
            break
        if finish >= settings.max_time:
            break
        start, state = finish, result.y[:, -1]
        end = min(end * settings.horizon_growth, settings.max_time)

    return Solution(np.concatenate(all_time), np.concatenate(all_state), status, message, nfev)
