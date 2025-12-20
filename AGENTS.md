# Repository Guidelines

## Project Structure & Module Organization
- Core library lives in `src/lerobot/`, with subpackages for policies, envs, data tools, and CLI scripts registered under `lerobot-*`.
- Reusable notebooks, scripts, and small demos sit in `examples/`, while production docs are authored in `docs/` and synced to huggingface.co.
- Comprehensive regression assets are collected under `tests/` (unit and e2e), with heavyweight artifacts cached inside `tests/outputs/`.
- Hardware, Docker, and benchmark utilities live in `docker/`, `benchmarks/`, and `media/`.

## Build, Test, and Development Commands
- `pip install -e ".[dev,test]"` enables editable installs with lint/test extras; use `pip install -e ".[aloha,pusht]"` when you need the gym environments locally.
- `pytest tests -m "not slow"` runs the default test suite; add `-k policy_name` when validating a focused module.
- `make test-end-to-end DEVICE=cpu` chains the ACT, diffusion, TDMPC, and SmolVLA training/eval smoke tests using the lightweight configs in `tests/outputs/`.
- `lerobot-train --config_path <json>` and `lerobot-eval --policy.path <dir>` drive CLI experiments when validating new configs.

## Coding Style & Naming Conventions
- The repo targets Python 3.10+; keep imports typed and prefer `pathlib` over raw strings for filesystem code.
- Ruff enforces linting and formatting (see `pyproject.toml`), with a 110-char limit and `E,W,F,I,UP,B,C4,A` rules. Run `ruff check src tests`.
- Modules follow `snake_case` filenames, classes use `PascalCase`, and CLI entry points remain `lerobot_<verb>.py`.
- Configuration constants (paths, repo IDs) belong in `lerobot/configs` or Hydra/Draccus config files rather than inline literals.

## Testing Guidelines
- Write fast pytest units near their targets (e.g., `tests/policies/test_act.py`) and mirror directory names from `src/`.
- Mark integration or hardware tests with `@pytest.mark.slow` so they can be excluded by default and triggered explicitly in CI.
- Keep fixtures deterministic by seeding PyTorch/NumPy and using the dataset stubs inside `tests/fixtures/`.
- End-to-end policies must export checkpoints into `tests/outputs/<policy>/checkpoints/` so `make test-end-to-end` can resume/eval without recomputing.

## Commit & Pull Request Guidelines
- Follow the existing `type(scope): summary` style (`fix(dataset)`, `chore(dependencies)`) and keep the subject ≤72 chars; reference issues with `#NNNN` when relevant.
- Commits should bundle code + tests + docs for a single change; leave refactors and generated artifacts in separate commits for easier reviews.
- PRs need a short changelog, reproduction steps (commands or config paths), screenshots for UI/docs, and confirmation that lint/pytest/Makefile goals were run.
- Link any new dataset/model cards hosted on Hugging Face and mention required hardware so reviewers can scope verification.
