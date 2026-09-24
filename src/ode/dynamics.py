"""Positive-manifold dynamics; state order: alpha_s, beta_s, alpha_r, beta_r."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from ode.observables import margins


@dataclass(frozen=True)
class Problem:
    subjects: int = 256
    relations: int = 16
    alpha0: float = 0.01
    beta0: float = 0.01

    def __post_init__(self):
        for value in (self.subjects, self.relations):
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError("Subject and relation counts must be integers >= 2.")
        if not all(np.isfinite(v) and v > 0 for v in (self.alpha0, self.beta0)):
            raise ValueError("Initialization must be finite and positive.")

    @property
    def initial_state(self):
        return np.array([self.alpha0, self.beta0, self.alpha0, self.beta0], dtype=float)


def sign_rates(problem: Problem, embedding: str) -> np.ndarray:
    """Table 2 rates, with normalized Hadamard embeddings and one-hot EOS."""
    s, r = problem.subjects, problem.relations
    embedding = embedding.upper()
    if embedding not in {"OOO", "OOH", "OHO", "OHH", "HOO", "HOH", "HHO", "HHH"}:
        raise ValueError(f"Unknown Sign GF embedding tuple: {embedding}")
    if "H" in embedding and any(n & (n - 1) for n in (s, r)):
        raise ValueError("Hadamard conditions require power-of-two cardinalities.")
    a_s = {"OO": 2 * np.sqrt(r), "OH": np.sqrt(s), "HO": np.sqrt(s * r), "HH": 1.0}[
        embedding[0] + embedding[2]
    ]
    a_r = {"OO": 2 * np.sqrt(s), "OH": np.sqrt(r), "HO": np.sqrt(s * r), "HH": 1.0}[
        embedding[1] + embedding[2]
    ]
    return np.array(
        [
            a_s,
            1 if embedding[0] == "O" else 1 / np.sqrt(s),
            a_r,
            1 if embedding[1] == "O" else 1 / np.sqrt(r),
        ]
    )


def vector_field(problem: Problem, flow: str, embedding: str = "OOO"):
    s, r = problem.subjects, problem.relations
    if flow == "sign":
        rates = sign_rates(problem, embedding)
        return lambda t, y: rates.copy()
    if flow not in {"gf", "spectral"}:
        raise ValueError(f"Unknown flow: {flow}")

    def rhs(t, y):
        a_s, b_s, a_r, b_r = y
        q_s, q_r = margins(y, s, r)
        if flow == "gf":
            e_s, e_r = expit(-q_s), expit(-q_r)
            # Proposition 3.3: correct the two relation typos in the page 6 ODE.
            return np.array(
                [
                    b_s * e_s / ((s - 1) * np.sqrt(r)),
                    a_s * e_s / (s * np.sqrt(r)),
                    b_r * e_r / ((r - 1) * np.sqrt(s)),
                    a_r * e_r / (r * np.sqrt(s)),
                ]
            )
        # Normalize in log space: expit(-q) can underflow for large margins.
        log_u = np.array([np.log(a_s) - np.logaddexp(0, q_s), np.log(a_r) - np.logaddexp(0, q_r)])
        u = np.exp(log_u - np.max(log_u))
        u /= np.linalg.norm(u)
        return np.array([1.0, u[0] / np.sqrt(s), 1.0, u[1] / np.sqrt(r)])

    return rhs
