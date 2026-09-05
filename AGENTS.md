# Repository guidance

- Use Hydra configuration rather than hard-coded experiment settings.
- Keep device handling backend-agnostic; never call `.cuda()` directly.
- Run `uv run pytest` and `uv run ruff check .` before committing.
- Use `uv run muon-smoke device=cpu` for local environment verification.
- Use `uv run muon-smoke device=cuda` on an allocated GPU before launching a sweep.
- Do not commit `runs/`, checkpoints, Hydra output directories, or large result files.
- Keep code compatible with the minimum supported PyTorch version (2.2.2).
