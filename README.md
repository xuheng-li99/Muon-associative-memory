# Muon in Associative Memory

Code for controlled experiments on subject and relation learning dynamics under
SGD, AdamW, Muon, and related optimizers.

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

## Why the PyTorch versions differ

Current PyTorch releases no longer publish Intel macOS wheels. On Intel Macs,
the CPU extra therefore resolves to PyTorch 2.2.2, the final compatible release.
Supported CPU platforms and the CUDA server resolve to a current pinned PyTorch
release. Experiment code must remain compatible with the minimum version and is
tested locally before CUDA sweeps are launched.
