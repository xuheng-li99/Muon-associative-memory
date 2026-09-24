# Muon in Associative Memory

Code for controlled experiments on subject and relation learning dynamics under
SGD, AdamW, Muon, and related optimizers.

## Experiment 1: reduced ODEs

Experiment 1 lives in `src/ode`, a standalone NumPy/SciPy package. The existing
`src/muon_assoc` holds runtime and smoke-test utilities and is available for the
later attention-training experiments; it does not yet implement Experiment 2.

For ODE work alone (no PyTorch or CUDA required):

```bash
./scripts/setup.sh ode
uv run muon-ode                         # main-text pilot: S=64, R=16
uv run muon-ode experiment=cardinality
uv run muon-ode experiment=precision
```

The pilot supplies the main-text phase portrait: one shared plot with subject
probability `p_S` on the x-axis and relation probability `p_R` on the y-axis.
The five curves correspond to Table 1: GF, spectral GF, Sign GF (O,O,O),
Sign GF (H,O,H), and Sign GF (O,H,H). It uses only S=64, R=16; there is no
S=R control in this preset. Open `notebooks/experiment1.ipynb` and run all cells
to load the pilot data and generate the figure.

Select dynamics and the stopping error through Hydra:

```bash
uv run muon-ode dynamics=main_text       # default: five Table 1 dynamics
uv run muon-ode dynamics=all             # GF, spectral GF, all eight Sign GF variants
uv run muon-ode experiment.delta=0.001   # stop when both p_S and p_R >= 0.999
```

`experiment.delta` is a single stopping error, not a list of evaluation thresholds.
Its default is 0.001 for pilot, 0.01 for cardinality, and 1e-6 for precision. GF and spectral
GF use adaptive float64 integration; Sign GF uses analytic linear parameter
trajectories. `solver.max_time` caps integration. Other learning-time thresholds
are evaluated later from the saved data, with interpolation error; smaller errors
than the trajectory reaches remain unreached. The precision preset therefore
needs only one run to support later analysis at multiple error thresholds.

The notebook and runner share these stable locations:

```text
runs/experiment1/pilot/main_text/
runs/experiment1/pilot/all/
runs/experiment1/cardinality/main_text/
runs/experiment1/precision/main_text/
```

Each named location points to the latest complete snapshot for that preset and
dynamics selection. Successful reruns atomically update it. Earlier snapshots
remain under `runs/experiment1/.snapshots/`; old date-named runs are also untouched.
Failed or capped runs retain diagnostic files there, exit nonzero, and do not
replace the previous complete run. The runner prints the snapshot location.
A notebook resolves the link once to avoid mixing files if a rerun finishes while
it is loading. Overrides of sizes, delta, or solver settings update the same named
location, so the notebook displays the saved sizes and stopping error. Use
`output_root=...` to keep a separate collection (and set `OUTPUT_ROOT` in the notebook).
Hydra multirun jobs sharing a preset/dynamics name publish to the same alias;
the last successful completion becomes current, while all snapshots are retained.

Each snapshot contains the resolved config and environment/Git metadata. Each
problem/flow subdirectory contains `config.yaml`, `status.json`, and
`trajectory.npz`. The NPZ has only `time`, `alpha_s`, `beta_s`, `alpha_r`, and
`beta_r`; runners save no derived metrics or figures. The two component crossings
at the stopping error are retained as ordinary time/state samples, together with
solver steps and dense samples.

In the notebook, `DYNAMICS = "main_text"` loads/plots five curves; `DYNAMICS = "all"`
loads/plots all ten. Set `PLOT_DYNAMICS = "main_text"` to display five curves from
an all-ten run. The loader reports missing dynamics rather than silently showing
an incomplete plot. For cardinality runs, set `PAIR = (S, R)` explicitly.

The same interface can be used in other notebooks:

```python
from ode.results import load_run
from ode.observables import probabilities, learning_times

recorded = load_run("runs/experiment1", experiment="pilot", dynamics="main_text")
case = recorded.cases["gf"]
s, r = case.config.problem.subjects, case.config.problem.relations
p_subject, p_relation, p_correct = probabilities(case.state, s, r)
t_subject, t_relation, ratio = learning_times(case.time, case.state, s, r, delta=0.1)
```

These are probability masses, not greedy accuracies. Unreached learning times are
NaN. `observables.py` also exposes `margins`, `errors`, `loss`, and
`wrong_subject_mass`.

Notebook dependencies are separate from the runner:

```bash
uv run --group notebook jupyter lab notebooks/experiment1.ipynb
```

There is no separate plotting module, initialization sweep, or full-matrix
validation experiment. See [the experiment plan](docs/experiment1-plan.md) for
the equations and sweep design.

### Appendix: quantitative GF versus spectral GF comparison

Generate the notebook's four appendix plots with:

```bash
uv run muon-ode experiment=ratio_fixed_s dynamics=gf_spectral
uv run muon-ode experiment=ratio_fixed_r dynamics=gf_spectral
uv run muon-ode experiment=appendix_precision dynamics=gf_spectral
```

The first two presets fix S=1024 and R=4, respectively, varying the other count
over powers of two from 4 to 1024. They measure T_S/T_R at 90% component
probability (`experiment.delta=0.1`). The precision preset uses S=64, R=16 and
stops at error 1e-6; the notebook measures both components' learning times across
errors from 1e-6 to 0.5. The error-versus-learning-time results are plotted separately for GF and spectral GF.
All four plots are log-log and contain only GF and
spectral GF. Their PDFs are saved under
`runs/experiment1/figures/appendix/gf_spectral/`.

## Environment setup

The project uses [uv](https://docs.astral.sh/uv/) for Python and dependency
management, and [Hydra](https://hydra.cc/) for experiment configuration. You do
not need to edit `pyproject.toml` for ordinary use.

Install uv once using its official installation instructions, then run one of:

```bash
# Local development or a machine without an NVIDIA GPU
./scripts/setup.sh cpu

# Linux x86_64 server with a visible NVIDIA driver
./scripts/setup.sh cuda
```

Both commands create `.venv`, install the appropriate PyTorch build, and run a
Hydra-driven forward/backward smoke test. The CUDA environment uses the PyTorch
CUDA 12.6 wheel; it does not require a separately installed CUDA toolkit, but it
does require a sufficiently recent NVIDIA driver.

This repository currently supports Python 3.11 and 3.12. uv will install the
version named in `.python-version` when necessary.

### Manual verification

```bash
uv run pytest
uv run ruff check .
uv run muon-smoke device=auto
```

Device selection is a Hydra override. Production configurations should use
`device=auto` by default and may use `device=cuda` on the server to fail fast if
the scheduler allocates a node without a working GPU.

The runtime smoke test and runtime tests require one of the PyTorch extras above.
An ODE-only environment can run `uv run pytest tests/test_ode.py` and
`uv run ruff check .`. Run `uv run --extra cpu pytest` for the entire test suite on CPU.

## Why the PyTorch versions differ

Current PyTorch releases no longer publish Intel macOS wheels. On Intel Macs,
the CPU extra therefore resolves to PyTorch 2.2.2, the final compatible release.
Supported CPU platforms and the CUDA server resolve to a current pinned PyTorch
release. Experiment code must remain compatible with the minimum version and is
tested locally before CUDA sweeps are launched.
