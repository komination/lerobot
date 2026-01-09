#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("groot_n1d6")
@dataclass
class GrootN1d6Config(PreTrainedConfig):
    """Configuration for GR00T N1.6 policy wrapper."""

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    max_state_dim: int = 128
    max_action_dim: int = 128
    action_horizon: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    image_size: tuple[int, int] = (256, 256)

    base_model_path: str = "nvidia/GR00T-N1.6-3B"
    model_name: str = "nvidia/Eagle-Block2A-2B-v2"

    embodiment_tag: str = "new_embodiment"

    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True
    tune_top_llm_layers: int = 0
    backbone_trainable_params_fp32: bool = True

    use_bf16: bool = True

    use_relative_action: bool = False
    apply_sincos_state_encoding: bool = False
    formalize_language: bool = True

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-5
    warmup_ratio: float = 0.05

    video_backend: str = "torchvision_av"
    balance_dataset_weights: bool = True
    balance_trajectory_weights: bool = True

    dataset_paths: list[str] | None = None
    output_dir: str = "./tmp/gr00t_n1d6"
    save_steps: int = 1000
    max_steps: int = 10000
    batch_size: int = 32
    dataloader_num_workers: int = 8
    report_to: str = "wandb"
    resume: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})"
            )

    def validate_features(self) -> None:
        image_features = [
            key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL
        ]
        if not image_features:
            raise ValueError(
                "Groot N1.6 policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(type=FeatureType.STATE, shape=(self.max_state_dim,))
            self.input_features["observation.state"] = state_feature
        else:
            state_shape = self.input_features["observation.state"].shape
            state_dim = state_shape[0] if state_shape else 0
            if state_dim > self.max_state_dim:
                raise ValueError(
                    f"State dimension {state_dim} exceeds max_state_dim {self.max_state_dim}. "
                    "Either reduce state dimension or increase max_state_dim in config."
                )

        if "action" not in self.output_features:
            action_feature = PolicyFeature(type=FeatureType.ACTION, shape=(self.max_action_dim,))
            self.output_features["action"] = action_feature
        else:
            action_shape = self.output_features["action"].shape
            action_dim = action_shape[0] if action_shape else 0
            if action_dim > self.max_action_dim:
                raise ValueError(
                    f"Action dimension {action_dim} exceeds max_action_dim {self.max_action_dim}. "
                    "Either reduce action dimension or increase max_action_dim in config."
                )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=int(10000 * self.warmup_ratio),
            num_decay_steps=10000,
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_lr * 0.1,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(min(self.chunk_size, self.action_horizon)))

    @property
    def reward_delta_indices(self) -> None:
        return None
