# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeRobot is a PyTorch-based robotics library from Hugging Face that provides models, datasets, and tools for real-world robotics. It supports imitation learning, reinforcement learning, and vision-language-action (VLA) models.

## Build and Development Commands

### Installation
```bash
pip install -e ".[dev,test]"   # Development install with test dependencies
uv sync --extra "test"         # Using uv (preferred in CI)
```

### Code Quality
```bash
pre-commit install             # Install pre-commit hooks (run once)
pre-commit run --all-files     # Run all linting/formatting checks
```

Pre-commit runs: ruff (format + lint), typos, pyupgrade, mypy, bandit, gitleaks.

### Testing
```bash
git lfs install && git lfs pull       # Required for test artifacts
pytest tests -vv                       # Run all tests
pytest tests/test_specific.py -sv      # Run single test file
pytest tests -vv --maxfail=10          # Stop after 10 failures (CI default)
```

### CLI Entry Points
All scripts are exposed as CLI commands:
- `lerobot-train` - Train a policy
- `lerobot-eval` - Evaluate a policy
- `lerobot-record` - Record robot data
- `lerobot-replay` - Replay recorded episodes
- `lerobot-calibrate` - Calibrate robot motors
- `lerobot-teleoperate` - Teleoperate a robot
- `lerobot-dataset-viz` - Visualize datasets
- `lerobot-info` - Display system/package info

## Architecture

### Source Layout (`src/lerobot/`)

**Core Abstractions:**
- `policies/` - Policy implementations (ACT, Diffusion, TDMPC, VQBeT, Pi0, SmolVLA, Gr00t, XVLA, etc.)
- `datasets/` - `LeRobotDataset` format (Parquet + MP4), data loading, streaming
- `robots/` - Hardware-agnostic `Robot` interface (SO100, Koch, Reachy2, LeKiwi, etc.)
- `teleoperators/` - Teleoperation devices (gamepad, phone, leader arms)
- `cameras/` - Camera interfaces (OpenCV, Intel RealSense)
- `motors/` - Motor controllers (Dynamixel, Feetech)

**Configuration System:**
- `configs/` - Draccus-based dataclass configs for training, evaluation, policies
- `PreTrainedConfig` (`configs/policies.py`) - Base class for all policy configs, uses `draccus.ChoiceRegistry` for dynamic policy type selection
- `TrainPipelineConfig` (`configs/train.py`) - Training configuration with dataset, policy, optimizer settings

**Key Modules:**
- `envs/` - Simulation environment wrappers (gym-aloha, gym-pusht, LIBERO, MetaWorld)
- `optim/` - Optimizer and learning rate scheduler configs
- `processor/` - Data preprocessing for VLA models
- `async_inference/` - Async policy inference for real-time control

### Policy Architecture

Each policy lives in `policies/{name}/`:
- `configuration_{name}.py` - Config dataclass inheriting `PreTrainedConfig`
- `modeling_{name}.py` - PyTorch model implementation

Policies must implement:
- `observation_delta_indices`, `action_delta_indices`, `reward_delta_indices` properties
- `get_optimizer_preset()`, `get_scheduler_preset()` methods
- `validate_features()` method

### Dataset Format

LeRobotDataset uses:
- Parquet files for state/action data
- MP4 videos (or images) for visual observations
- Metadata stored on Hugging Face Hub

### Adding New Components

When adding a new policy:
1. Update `available_policies` and `available_policies_per_env` in `lerobot/__init__.py`
2. Set the required `name` class attribute
3. Update `tests/test_available.py`

When adding a new environment:
1. Update `available_tasks_per_env` and `available_datasets_per_env` in `lerobot/__init__.py`

## Optional Dependencies

Many features require specific extras:
```bash
pip install -e ".[smolvla]"     # SmolVLA policy
pip install -e ".[groot]"       # Gr00t N1.5 (requires flash-attn)
pip install -e ".[aloha]"       # ALOHA simulation
pip install -e ".[libero]"      # LIBERO benchmark
pip install -e ".[feetech]"     # Feetech motor support
pip install -e ".[dynamixel]"   # Dynamixel motor support
```

Note: `wallx` and `pi` have conflicting transformer versions and cannot be installed together with other VLA policies.
