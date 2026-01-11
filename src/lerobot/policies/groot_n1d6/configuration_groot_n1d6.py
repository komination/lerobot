#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""Configuration for GR00T N1.6 policy."""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


# N1.6 Embodiment tag to projector index mapping
EMBODIMENT_TAG_TO_PROJECTOR_INDEX = {
    "oxe_google": 0,
    "oxe_widowx": 1,
    "libero_panda": 2,
    "unitree_g1": 8,
    "new_embodiment": 10,
    "robocasa_panda_omron": 13,
    "gr1": 20,
    "behavior_r1_pro": 24,
}


@PreTrainedConfig.register_subclass("groot_n1d6")
@dataclass
class GrootN1d6Config(PreTrainedConfig):
    """Configuration for GR00T N1.6 policy wrapper.

    GR00T N1.6 is an improved version with:
    - 32-layer DiT (vs 16 in N1.5)
    - AlternateVLDiT with image/text attention separation
    - Eagle-Block2A-2B-v2 (Eagle 3) backbone
    - Larger max dimensions (128 vs 64/32)
    - State augmentation features (dropout, noise)
    """

    # Basic policy settings
    n_obs_steps: int = 1
    chunk_size: int = 40  # N1.6 uses 40 as max
    n_action_steps: int = 40

    # Dimension settings (N1.6 increased limits)
    max_state_dim: int = 128  # Changed from 64 in N1.5
    max_action_dim: int = 128  # Changed from 32 in N1.5
    action_horizon: int = 16  # Prediction horizon within chunk

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Image preprocessing
    image_size: tuple[int, int] = (256, 256)  # N1.6 default

    # Model paths
    base_model_path: str = "nvidia/GR00T-N1.6-3B"
    model_name: str = "nvidia/Eagle-Block2A-2B-v2"
    tokenizer_assets_repo: str = "nvidia/Eagle-Block2A-2B-v2"

    # Embodiment configuration
    embodiment_tag: str = "new_embodiment"
    max_num_embodiments: int = 32

    # Backbone configuration
    backbone_embedding_dim: int = 2048
    select_layer: int = 16
    use_flash_attention: bool = True
    load_bf16: bool = True
    tune_top_llm_layers: int = 4  # N1.6 feature: tune top LLM layers
    backbone_trainable_params_fp32: bool = True

    # Fine-tuning control
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True  # N1.6 specific

    # DiT configuration (N1.6 uses 32 layers)
    hidden_size: int = 1024
    input_embedding_dim: int = 1536
    use_alternate_vl_dit: bool = True  # N1.6 default
    attend_text_every_n_blocks: int = 2
    add_pos_embed: bool = True
    use_vlln: bool = True
    max_seq_len: int = 1024

    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 32,  # N1.6: 32 layers (vs 16 in N1.5)
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 1024,
            "interleave_self_attention": True,
        }
    )

    # Flow matching parameters
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # State augmentation (N1.6 features)
    state_dropout_prob: float = 0.0
    state_additive_noise_scale: float = 0.0

    # LoRA parameters
    lora_rank: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_full_model: bool = False

    # Training parameters
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    use_bf16: bool = True

    # Dataset parameters
    video_backend: str = "torchvision_av"
    balance_dataset_weights: bool = True
    balance_trajectory_weights: bool = True

    # Training output
    dataset_paths: list[str] | None = None
    output_dir: str = "./tmp/groot_n1d6"
    save_steps: int = 1000
    max_steps: int = 10000
    batch_size: int = 32
    dataloader_num_workers: int = 8
    report_to: str = "wandb"
    resume: bool = False

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})"
            )

        # Validate embodiment tag
        if self.embodiment_tag not in EMBODIMENT_TAG_TO_PROJECTOR_INDEX:
            raise ValueError(
                f"Unknown embodiment_tag: {self.embodiment_tag}. "
                f"Supported tags: {list(EMBODIMENT_TAG_TO_PROJECTOR_INDEX.keys())}"
            )

    def validate_features(self) -> None:
        """Validate and set up input/output features for GR00T N1.6."""
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "GR00T N1.6 policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features["observation.state"] = state_feature
        else:
            state_shape = self.input_features["observation.state"].shape
            state_dim = state_shape[0] if state_shape else 0
            if state_dim > self.max_state_dim:
                raise ValueError(
                    f"State dimension {state_dim} exceeds max_state_dim {self.max_state_dim}. "
                    f"Either reduce state dimension or increase max_state_dim in config."
                )

        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
            self.output_features["action"] = action_feature
        else:
            action_shape = self.output_features["action"].shape
            action_dim = action_shape[0] if action_shape else 0
            if action_dim > self.max_action_dim:
                raise ValueError(
                    f"Action dimension {action_dim} exceeds max_action_dim {self.max_action_dim}. "
                    f"Either reduce action dimension or increase max_action_dim in config."
                )

    def get_optimizer_preset(self) -> AdamWConfig:
        """Return optimizer configuration."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        """Return scheduler configuration."""
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=int(self.max_steps * self.warmup_ratio),
            num_decay_steps=self.max_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_lr * 0.1,
        )

    @property
    def observation_delta_indices(self) -> None:
        """Return indices for delta observations (None for GR00T)."""
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Return indices for delta actions."""
        return list(range(min(self.chunk_size, self.action_horizon)))

    @property
    def reward_delta_indices(self) -> None:
        """Return indices for delta rewards (None for GR00T)."""
        return None

    @property
    def embodiment_id(self) -> int:
        """Get the embodiment ID for the configured embodiment tag."""
        return EMBODIMENT_TAG_TO_PROJECTOR_INDEX[self.embodiment_tag]
