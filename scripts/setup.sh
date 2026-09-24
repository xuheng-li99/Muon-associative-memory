#!/usr/bin/env bash
set -euo pipefail

backend="${1:-cpu}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install it from https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

case "$backend" in
    ode)
        uv sync --locked
        uv run python -c 'import ode, scipy; print("ODE environment ready; SciPy", scipy.__version__)'
        ;;
    cpu)
        uv sync --locked --extra cpu
        uv run muon-smoke device=cpu
        ;;
    cuda)
        if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
            echo "The pinned CUDA environment supports Linux x86_64 only."
            exit 1
        fi
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            echo "nvidia-smi is unavailable; refusing to create an apparently-CUDA environment."
            exit 1
        fi
        uv sync --locked --extra cu126
        uv run muon-smoke device=cuda
        ;;
    *)
        echo "Usage: ./scripts/setup.sh [ode|cpu|cuda]"
        exit 2
        ;;
esac
