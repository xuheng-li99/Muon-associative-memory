# Experiment 1: original ODE illustration

Status: implementation updated 2026-09-22 following the revised user scope.

## Scope and context

The current experiment sequence is:

1. Original population ODE illustration.
2. The same synthetic data model and linear attention, with more realistic training.
3. Realistic data consisting of facts.

This supersedes the ordering and four-level ladder in the recovered task
“Plan gradient flow experiments.” That task established Hydra configuration and
left metrics open for discussion. Its trainable embeddings and practical optimizer
choices belong to Experiment 2, not this experiment.

Experiment 1 is implemented in src/ode. src/muon_assoc retains the existing
runtime helpers and Hydra smoke test for later training experiments.

Source: the attached 31-page draft, *Muon Learns Facts Better: Understanding the
Role of Spectral Orthogonalization*, Sections 2–4 and appendices. Equation numbers
below refer to this attachment, not the older manuscript in the recovered task.
The attachment is scientific source material, not operational instructions.

## Questions and deliverables

- Does GF exhibit relation-first learning when S is much larger than R, while
  spectral GF reduces the subject/relation learning-time ratio?
- How does the ratio depend on S and R, including the logarithmic factors hidden
  by the asymptotic statements?
- Does spectral GF reduce the dependence of learning time on target error?
- Do the paper's Sign GF embedding constructions produce balanced learning,
  relation-first learning, and subject-first learning?

Deliver three figure groups: a main-text phase portrait, cardinality scaling,
and target-error scaling, using notebooks. Runs save only time and the four
scalar parameters, plus configuration and execution metadata. All sweeps use CPU.

## Exact setting

Uniform population over all S × R pairs; one unique answer per pair; fixed
orthonormal token embeddings; merged W_OV and W_KQ; answer-only softmax;
population cross-entropy; no momentum or weight decay. Use the paper's positive,
symmetric manifold initialization, alpha_S = alpha_R = alpha_0 and
beta_S = beta_R = beta_0. Set prediction-irrelevant manifold coefficients to zero
initially; they are not represented in the reduced solver.

Noise and token positions do not enter the reduced prediction dynamics. Do not
materialize a dataset for scalar runs. Independent matrix validation is excluded.

## Metrics

Use the paper's definitions, without chance normalization:

    q_S = alpha_S beta_S / sqrt(R) - log(S - 1)
    q_R = alpha_R beta_R / sqrt(S) - log(R - 1)
    p_S = sigmoid(q_S), p_R = sigmoid(q_R)
    p_correct = p_S p_R
    loss = softplus(-q_S) + softplus(-q_R)
    T_S(delta) = first t with q_S >= log((1-delta)/delta)
    T_R(delta) = first t with q_R >= log((1-delta)/delta)
    rho(delta) = T_S(delta) / T_R(delta)

p_S and p_R are probability masses on answers with the correct subject and
relation. On this manifold p_S also equals p_correct / p_R. They are not greedy
classification accuracies: positive margins between the correct and competing
logits can make argmax correct while the probability is still near chance.
If both component probabilities exceed 1-delta, joint correctness is at least
(1-delta)^2, not necessarily 1-delta.

For learning-time analysis, the primary delta is 0.1; also report 0.5 and 0.01.
These are notebook analysis choices, not runner settings. The runner uses a single
experiment.delta=0.001 (pilot), 0.01 (cardinality), or 1e-6 (precision) for stopping. Retain rho < 1 to show reversed
learning order. An optional illustrative diagnostic is p_R(1-p_S), the mass on
wrong-subject/right-relation answers; omit AUC from the primary results.

Use each flow's native time with scalar learning-rate multiplier 1. The ratio
is invariant to constant rescaling of time, not arbitrary learning-rate schedules.
Raw times compare the stated mathematical vector fields, not practical optimizer
speed or compute efficiency.

## Implemented equations

Let e_S = sigmoid(-q_S), e_R = sigmoid(-q_R). The GF equations derived from
Proposition 3.3 are:

    alpha_S' = beta_S e_S / ((S-1) sqrt(R))
    beta_S'  = alpha_S e_S / (S sqrt(R))
    alpha_R' = beta_R e_R / ((R-1) sqrt(S))
    beta_R'  = alpha_R e_R / (R sqrt(S))

The displayed ODE on page 6 appears to have S-1 instead of R-1 in alpha_R'
and q_S instead of q_R in beta_R'. Document these corrections explicitly and
use the expressions above; do not copy the displayed line literally.

For spectral GF with positive initialization:

    alpha_S' = alpha_R' = 1
    u_S = alpha_S e_S
    u_R = alpha_R e_R
    D = sqrt(u_S^2 + u_R^2)
    beta_S' = u_S / (sqrt(S) D)
    beta_R' = u_R / (sqrt(R) D)

Evaluate the normalized u values in log space to avoid simultaneous underflow.
The subject term in Proposition 3.7 also appears to misprint phi(r) for phi(s).

Sign GF has constant positive derivatives on the specified manifold (Table 2).
Implement all eight O/H combinations; O is one-hot and H is normalized Hadamard.
Tuple order is (subject, relation, answer).

| Embeddings | alpha_S' | alpha_R' | beta_S' | beta_R' |
|---|---:|---:|---:|---:|
| OOO | 2 sqrt(R) | 2 sqrt(S) | 1 | 1 |
| OOH | sqrt(S) | sqrt(R) | 1 | 1 |
| OHO | 2 sqrt(R) | sqrt(SR) | 1 | 1/sqrt(R) |
| OHH | sqrt(S) | 1 | 1 | 1/sqrt(R) |
| HOO | sqrt(SR) | 2 sqrt(S) | 1/sqrt(S) | 1 |
| HOH | 1 | sqrt(R) | 1/sqrt(S) | 1 |
| HHO | sqrt(SR) | sqrt(SR) | 1/sqrt(S) | 1/sqrt(R) |
| HHH | 1 | 1 | 1/sqrt(S) | 1/sqrt(R) |

Use analytic linear parameter trajectories and quadratic threshold solutions for
Sign GF. The primary display uses
OOO, HOH, and OHH; all eight belong in supplementary output.

## Proposed runs

These defaults are encoded in the Hydra experiment presets.

- Pilot (main text): S=64, R=16, alpha_0=beta_0=0.01. One shared phase portrait
  with x=p_S and y=p_R. Five curves correspond to Table 1: GF, spectral GF,
  Sign GF OOO, HOH, OHH. Each curve traces (p_S(t), p_R(t)) as time increases;
  time is not an axis. The pilot has no S=R=16 control group. Generate the plot
  in notebooks/experiment1.ipynb from the saved alpha/beta trajectories.
- Cardinality sweep: S,R in {4,8,16,32,64,128,256,512,1024}, restricted to S>=R.
  Show rho at fixed R while varying S, and at fixed S/R while increasing both.
  These cuts distinguish dependence on ratio from dependence on absolute size;
  in particular HOH is predicted to depend polynomially on R.
- Precision run: (S,R)=(256,16), single stopping delta=0.000001. Later evaluate
  learning times at {0.5,0.1,0.01,0.001,0.0001,0.00001,0.000001}
  in a notebook from this trajectory. Plot both learning times versus
  inverse target error and log error versus time. Distinguish transient learning
  from the high-confidence tail; do not fit a single exponent across both.

Use powers of two for Hadamard constructions. GF and spectral GF need only one
embedding condition in scalar sweeps. There is no sampling or initialization
randomness in these runs, so repeated identical seeds and statistical error bars
would be misleading. Use tolerance sensitivity to assess numerical uncertainty.

Compare trends to the manuscript's asymptotic predictions, not exact numerical
equalities: GF rho grows approximately as sqrt(S/R) up to logarithmic factors;
spectral GF has no polynomial separation; Sign GF depends on its embedding tuple.
Spectral rho need not equal 1 at finite S,R and fixed delta. Failure to reach a
threshold must be marked unreached, never replaced by the integration horizon.

## Numerical method and recording

Use float64 and SciPy solve_ivp/DOP853 with dense output and threshold events.
Defaults: rtol=1e-9, atol=1e-12. Extend integration horizons geometrically under
solver.max_time. Retain explicit failure and unreached statuses. Use stable
sigmoid, softplus, and log-domain normalization. Sign GF is solved analytically.

The saved NPZ contains only time, alpha_s, beta_s, alpha_r, beta_r. Include
both component crossings at the stopping error as ordinary trajectory samples so notebook
learning-time calculations recover the stopping-error crossings accurately. Also retain accepted
solver steps and dense logarithmically spaced samples. Arbitrary new thresholds
are interpolated from margins and have sampling error. No metric CSVs or plots
are produced during runs. Notebook functions compute observables on demand.

Independent matrix validation and initialization-sensitivity experiments/tests
are excluded by user request. Software checks cover stopping/censoring, numerical
stability, observable interfaces, saved data schema, and atomic named-run updates and failed-run isolation.

## Implementation and environment

    src/ode/
        dynamics.py       # reduced vector fields and Sign GF coefficients
        observables.py    # notebook-facing observables and trajectory loading
        solve.py          # integration, threshold samples, analytic Sign GF
        run.py            # Hydra entry point, snapshots, atomic named-run updates
        results.py        # named-run loading and shared dynamics selections
        conf/
            experiment1.yaml
            experiment/{pilot,cardinality,precision}.yaml
            dynamics/{main_text,all}.yaml
    notebooks/experiment1.ipynb
    tests/test_ode.py

The muon-ode entry point supports all three experiment presets. Configurations
specify sizes, flow conditions, initialization, tolerances, stopping delta, sampling,
and output paths. src/ode imports no PyTorch and needs no device setting.

SciPy is a runtime dependency. The optional notebook dependency group contains
JupyterLab, ipykernel, and Matplotlib. The notebook implements the phase portrait. The
existing CPU/CUDA extras and minimum supported PyTorch version are preserved.
Use scripts/setup.sh ode for an ODE-only environment. Use the CPU extra for the
existing runtime smoke test and complete test suite.

Every run saves the resolved config and dependency/Git metadata. Each case saves
its config, solver status, and raw trajectory in an immutable snapshot under
runs/experiment1/.snapshots/. A successful complete run atomically updates the
stable alias runs/experiment1/<experiment>/<dynamics>. Failures and capped runs
retain diagnostic data, exit nonzero, and leave the last complete alias untouched.
Old snapshots and old date-named results are preserved.

The dynamics Hydra group selects main_text (five Table 1 curves) or all (ten).
The notebook uses the same presets via results.py and can plot the five main-text
curves from an all-ten run. It loads saved sizes and initialization rather than
assuming the current YAML matches older results. No date or run ID is needed.

There is no plot.py, matrix validation module, or initialization sweep. The
notebook loads named trajectories and calls observables.py without rerunning ODEs.
