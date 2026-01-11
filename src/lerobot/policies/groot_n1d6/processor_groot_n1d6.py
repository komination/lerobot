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

"""Processor pipeline for GR00T N1.6 policy.

This module provides data preprocessing and postprocessing for the GR00T N1.6
Vision-Language-Action model, adapted for LeRobot's processor pipeline system.
"""

import logging
from typing import Any

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.groot_n1d6.configuration_groot_n1d6 import (
    EMBODIMENT_TAG_TO_PROJECTOR_INDEX,
    GrootN1d6Config,
)
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
)
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    HF_LEROBOT_HOME,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

logger = logging.getLogger(__name__)

# Check for transformers availability
try:
    from transformers import AutoProcessor, ProcessorMixin
    from transformers.feature_extraction_utils import BatchFeature

    _transformers_available = True
except ImportError:
    _transformers_available = False
    AutoProcessor = None
    ProcessorMixin = object
    BatchFeature = dict


def make_groot_n1d6_pre_post_processors(
    config: GrootN1d6Config, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create preprocessor and postprocessor for GR00T N1.6 policy.

    This creates a processing pipeline that transforms LeRobot data format into
    the format expected by GR00T N1.6:

    Preprocessing steps:
    1. Optional key renaming (dataset-specific key mapping)
    2. Add batch dimension to unbatched data
    3. Pack video/state/action/language/embodiment with normalization and padding
    4. Create VLM content for Eagle 3 backbone
    5. Move tensors to device (GPU)

    Args:
        config: GR00T N1.6 configuration
        dataset_stats: Optional per-key min/max statistics for normalization

    Returns:
        Tuple of (preprocessor, postprocessor) pipelines
    """
    # Get horizon/dimension parameters from config
    state_horizon = 1
    action_horizon = min(config.chunk_size, config.action_horizon)

    # Build preprocessing steps
    pre_steps: list[ProcessorStep] = []

    # 1. Rename observations if needed
    observation_rename_map = {}  # Can be populated from config if needed
    if observation_rename_map:
        pre_steps.append(RenameObservationsProcessorStep(observation_rename_map))

    # 2. Add batch dimension for unbatched inputs
    pre_steps.append(AddBatchDimensionProcessorStep())

    # 3. Pack inputs step - handles state/action padding, normalization, masking
    pre_steps.append(
        GrootN1d6PackInputsStep(
            config=config,
            state_horizon=state_horizon,
            action_horizon=action_horizon,
            dataset_stats=dataset_stats,
        )
    )

    # 4. VLM encoding step - creates input for Eagle 3 backbone
    pre_steps.append(GrootN1d6VLMEncodeStep(config=config))

    # 5. Device step
    pre_steps.append(DeviceProcessorStep())

    # Build postprocessing steps
    post_steps: list[ProcessorStep] = []

    # Unpack actions step - removes padding, denormalizes
    post_steps.append(
        GrootN1d6ActionUnpackStep(
            config=config,
            action_horizon=action_horizon,
            dataset_stats=dataset_stats,
        )
    )

    preprocessor = PolicyProcessorPipeline(pre_steps, name=POLICY_PREPROCESSOR_DEFAULT_NAME)
    postprocessor = PolicyProcessorPipeline(post_steps, name=POLICY_POSTPROCESSOR_DEFAULT_NAME)

    return preprocessor, postprocessor


@ProcessorStepRegistry.register("groot_n1d6_pack_inputs")
class GrootN1d6PackInputsStep(ProcessorStep):
    """Pack and normalize inputs for GR00T N1.6.

    Handles:
    - State padding to max_state_dim
    - Action padding to max_action_dim
    - Optional min-max normalization
    - Creating state_mask and action_mask
    - Embodiment ID mapping
    """

    def __init__(
        self,
        config: GrootN1d6Config,
        state_horizon: int = 1,
        action_horizon: int = 16,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        self.config = config
        self.max_state_dim = config.max_state_dim
        self.max_action_dim = config.max_action_dim
        self.state_horizon = state_horizon
        self.action_horizon = action_horizon
        self.dataset_stats = dataset_stats
        self.embodiment_tag = config.embodiment_tag
        self.embodiment_id = EMBODIMENT_TAG_TO_PROJECTOR_INDEX.get(config.embodiment_tag, 10)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Pack inputs for GR00T N1.6."""
        result = dict(data)

        # Get state from observation.state
        state = data.get("observation.state")
        if state is not None:
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state)
            state = state.float()

            # Handle shape
            if state.dim() == 1:
                state = state.unsqueeze(0)  # Add batch dim
            if state.dim() == 2:
                B, D = state.shape
                T = self.state_horizon
            else:
                B, T, D = state.shape

            # Normalize if stats available
            if self.dataset_stats and "observation.state" in self.dataset_stats:
                stats = self.dataset_stats["observation.state"]
                mean = stats.get("mean", torch.zeros(D))
                std = stats.get("std", torch.ones(D))
                state = (state - mean) / (std + 1e-8)

            # Pad to max_state_dim
            if D < self.max_state_dim:
                padding = torch.zeros(*state.shape[:-1], self.max_state_dim - D, device=state.device)
                state = torch.cat([state, padding], dim=-1)

            # Create state mask
            state_mask = torch.zeros(B, self.max_state_dim, dtype=torch.bool, device=state.device)
            state_mask[:, :D] = True

            result["state"] = state
            result["state_mask"] = state_mask

        # Get action if present (for training)
        action = data.get("action")
        if action is not None:
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action)
            action = action.float()

            # Handle shape
            if action.dim() == 2:
                B, D = action.shape
                action = action.unsqueeze(1)  # Add time dim
                T = 1
            else:
                B, T, D = action.shape

            # Normalize if stats available
            if self.dataset_stats and "action" in self.dataset_stats:
                stats = self.dataset_stats["action"]
                mean = stats.get("mean", torch.zeros(D))
                std = stats.get("std", torch.ones(D))
                action = (action - mean) / (std + 1e-8)

            # Pad time dimension to action_horizon
            if T < self.action_horizon:
                time_padding = action[:, -1:, :].repeat(1, self.action_horizon - T, 1)
                action = torch.cat([action, time_padding], dim=1)
            elif T > self.action_horizon:
                action = action[:, : self.action_horizon, :]

            # Pad to max_action_dim
            if D < self.max_action_dim:
                padding = torch.zeros(B, self.action_horizon, self.max_action_dim - D, device=action.device)
                action = torch.cat([action, padding], dim=-1)

            # Create action mask
            action_mask = torch.zeros(
                B, self.action_horizon, self.max_action_dim, dtype=torch.float32, device=action.device
            )
            action_mask[:, :T, :D] = 1.0

            result["action"] = action
            result["action_mask"] = action_mask

        # Add embodiment_id
        batch_size = state.shape[0] if state is not None else 1
        result["embodiment_id"] = torch.full((batch_size,), self.embodiment_id, dtype=torch.long)

        return result


@ProcessorStepRegistry.register("groot_n1d6_vlm_encode")
class GrootN1d6VLMEncodeStep(ProcessorStep):
    """Create VLM content for Eagle 3 backbone.

    Handles:
    - Image preprocessing
    - Text/language instruction formatting
    - Creating conversation format for Eagle
    """

    def __init__(self, config: GrootN1d6Config):
        self.config = config
        self.image_size = config.image_size
        self._processor = None

    @property
    def processor(self):
        """Lazy load the VLM processor."""
        if self._processor is None and _transformers_available:
            try:
                self._processor = AutoProcessor.from_pretrained(
                    self.config.tokenizer_assets_repo, trust_remote_code=True
                )
                self._processor.tokenizer.padding_side = "left"
            except Exception as e:
                logger.warning(f"Could not load Eagle processor: {e}")
                self._processor = None
        return self._processor

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create VLM content from images and language."""
        result = dict(data)

        # Collect image features
        images = []
        for key, value in data.items():
            if isinstance(value, torch.Tensor) and value.dim() >= 3:
                # Check if this looks like an image
                if value.shape[-1] == 3 or value.shape[-3] == 3:
                    images.append(value)
            elif isinstance(value, np.ndarray) and value.ndim >= 3:
                if value.shape[-1] == 3 or value.shape[-3] == 3:
                    images.append(torch.from_numpy(value))

        # Get language instruction
        language = data.get("language_instruction", data.get("task", ""))
        if isinstance(language, bytes):
            language = language.decode("utf-8")
        if not language:
            language = "Complete the task."

        # If we have a processor, use it
        if self.processor is not None and images:
            try:
                # Convert tensors to PIL images
                pil_images = []
                for img in images:
                    if img.dim() == 4:
                        img = img[0]  # Take first in batch
                    if img.dim() == 3:
                        if img.shape[0] == 3:
                            img = rearrange(img, "c h w -> h w c")
                        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
                        pil_images.append(Image.fromarray(img_np))

                # Create conversation format
                image_tags = "<image>" * len(pil_images)
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"} for _ in pil_images
                        ]
                        + [{"type": "text", "text": language}],
                    }
                ]

                # Process with Eagle
                text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = self.processor(text=text, images=pil_images, return_tensors="pt", padding=True)

                for k, v in inputs.items():
                    result[k] = v

            except Exception as e:
                logger.warning(f"VLM encoding failed: {e}")

        return result


@ProcessorStepRegistry.register("groot_n1d6_action_unpack")
class GrootN1d6ActionUnpackStep(ProcessorStep):
    """Unpack and denormalize actions from GR00T N1.6 output.

    Handles:
    - Removing action dimension padding
    - Denormalization using dataset stats
    """

    def __init__(
        self,
        config: GrootN1d6Config,
        action_horizon: int = 16,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        self.config = config
        self.action_horizon = action_horizon
        self.dataset_stats = dataset_stats

    def __call__(self, action: PolicyAction) -> PolicyAction:
        """Unpack actions from GR00T N1.6 output."""
        if isinstance(action, dict):
            actions = action.get("action", action.get("action_pred"))
        else:
            actions = action

        if actions is None:
            return action

        # Get original action dimension
        original_dim = self.config.output_features.get("action", PolicyFeature(type=FeatureType.ACTION, shape=(6,))).shape[0]

        # Trim padding
        actions = actions[..., :original_dim]

        # Denormalize if stats available
        if self.dataset_stats and "action" in self.dataset_stats:
            stats = self.dataset_stats["action"]
            mean = stats.get("mean", torch.zeros(original_dim))
            std = stats.get("std", torch.ones(original_dim))
            if isinstance(actions, torch.Tensor):
                mean = mean.to(actions.device)
                std = std.to(actions.device)
            actions = actions * (std + 1e-8) + mean

        if isinstance(action, dict):
            action["action"] = actions
            return action
        return actions
