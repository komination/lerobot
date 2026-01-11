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

"""GR00T N1.6 Policy Wrapper for LeRobot Integration.

This module provides the GrootN1d6Policy class that wraps the GR00T N1.6
Vision-Language-Action model for use with LeRobot's training and inference
pipelines.

Key features of N1.6 vs N1.5:
- 32-layer DiT (vs 16 in N1.5) for improved action quality
- AlternateVLDiT with image/text attention separation
- Eagle 3 (Eagle-Block2A-2B-v2) backbone
- Larger max dimensions (128 vs 64/32)
- State dropout and noise augmentation for training
"""

import logging
import os
from collections import deque

import torch
from torch import Tensor

from lerobot.policies.groot_n1d6.configuration_groot_n1d6 import GrootN1d6Config
from lerobot.policies.groot_n1d6.groot_n1d6 import Gr00tN1d6
from lerobot.policies.pretrained import PreTrainedPolicy

logger = logging.getLogger(__name__)


class GrootN1d6Policy(PreTrainedPolicy):
    """LeRobot policy wrapper for GR00T N1.6 model.

    This class provides the interface required by LeRobot's training and
    inference pipelines while delegating to the underlying Gr00tN1d6 model
    for the actual computation.
    """

    name = "groot_n1d6"
    config_class = GrootN1d6Config

    def __init__(self, config: GrootN1d6Config, **kwargs):
        """Initialize GR00T N1.6 policy wrapper.

        Args:
            config: Policy configuration
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Handle Flash Attention compatibility
        self._handle_flash_attention_compatibility()

        # Initialize GR00T N1.6 model
        self._groot_model = self._create_groot_model()

        self.reset()

    def _create_groot_model(self) -> Gr00tN1d6:
        """Create and initialize the GR00T N1.6 model.

        Returns:
            Initialized Gr00tN1d6 model
        """
        logger.info(f"Loading GR00T N1.6 model from {self.config.base_model_path}")

        # For now, create model from config
        # In the future, we can add support for loading pretrained weights
        model = Gr00tN1d6(self.config)

        logger.info(
            f"GR00T N1.6 model initialized with {sum(p.numel() for p in model.parameters()):,} parameters"
        )
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {trainable_params:,}")

        return model

    def reset(self):
        """Reset policy state when environment resets."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        """Get parameters for optimization."""
        return self.parameters()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass.

        Args:
            batch: Dictionary containing training batch with keys:
                - state: [B, state_dim] state observations
                - action: [B, action_horizon, action_dim] target actions
                - action_mask: [B, action_horizon, action_dim] action mask
                - embodiment_id: [B] embodiment IDs
                - input_ids, attention_mask, pixel_values: VLM inputs

        Returns:
            Tuple of (loss tensor, loss dictionary)
        """
        # Build input dict for GR00T N1.6
        # Keep only tensors that the model consumes
        allowed_base = {"state", "state_mask", "action", "action_mask", "embodiment_id"}
        vlm_keys = {"input_ids", "attention_mask", "pixel_values"}

        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k in vlm_keys) and not (k.startswith("next.") or k == "info")
        }

        # Get device from model parameters
        device = next(self.parameters()).device

        # Run forward under bf16 autocast when enabled
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.forward(groot_inputs)

        loss = outputs.get("loss")

        loss_dict = {"loss": loss.item()}

        # Add additional losses if available
        if "action_loss" in outputs:
            action_loss = outputs["action_loss"]
            loss_dict["action_loss"] = action_loss.sum().item() / (
                outputs.get("action_mask", torch.ones_like(action_loss)).sum().item() + 1e-6
            )

        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions for inference.

        Args:
            batch: Dictionary containing inference inputs

        Returns:
            Tensor of shape (B, n_action_steps, action_dim)
        """
        self.eval()

        # Build input dict for inference
        # Note: During inference, we should NOT pass action/action_mask
        allowed_base = {"state", "state_mask", "embodiment_id"}
        vlm_keys = {"input_ids", "attention_mask", "pixel_values"}

        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k in vlm_keys) and not (k.startswith("next.") or k == "info")
        }

        # Get device from model parameters
        device = next(self.parameters()).device

        # Use bf16 autocast for inference
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.get_action(groot_inputs)

        actions = outputs.get("action_pred")

        # Trim to original action dimension (remove padding)
        original_action_dim = self.config.output_features["action"].shape[0]
        actions = actions[:, :, :original_action_dim]

        # Limit to n_action_steps
        actions = actions[:, : self.config.n_action_steps, :]

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select single action from action queue.

        Uses temporal ensembling by maintaining an action queue and returning
        actions one at a time.

        Args:
            batch: Dictionary containing observation inputs

        Returns:
            Tensor of shape (B, action_dim) - single action to execute
        """
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            # Transpose to (n_action_steps, B, action_dim) and extend queue
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    def _handle_flash_attention_compatibility(self) -> None:
        """Handle Flash Attention compatibility issues.

        This addresses common 'undefined symbol' errors that occur when Flash
        Attention is compiled against a different PyTorch version.
        """
        # Set environment variables for Flash Attention compatibility
        os.environ.setdefault("FLASH_ATTENTION_FORCE_BUILD", "0")
        os.environ.setdefault("FLASH_ATTENTION_SKIP_CUDA_BUILD", "0")

        try:
            import flash_attn

            logger.info(f"[GROOT N1.6] Flash Attention version: {flash_attn.__version__}")
        except ImportError as e:
            logger.warning(f"[GROOT N1.6] Flash Attention not available: {e}")
            logger.warning("[GROOT N1.6] GR00T N1.6 requires Flash Attention for Eagle 3 backbone")
            logger.warning("[GROOT N1.6] Install with: pip install flash-attn>=2.7.0")
        except Exception as e:
            if "undefined symbol" in str(e):
                logger.error(f"[GROOT N1.6] Flash Attention compatibility issue: {e}")
                logger.error("[GROOT N1.6] This is likely due to PyTorch/Flash Attention version mismatch")
                logger.error("[GROOT N1.6] Try reinstalling Flash Attention:")
                logger.error("  pip uninstall flash-attn")
                logger.error("  pip install --no-build-isolation flash-attn>=2.7.0")
            else:
                logger.error(f"[GROOT N1.6] Flash Attention error: {e}")
