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

"""Eagle 3 Vision-Language Backbone for GR00T N1.6."""

import logging
import os

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

logger = logging.getLogger(__name__)


class EagleBackbone(nn.Module):
    """Eagle 3 Vision-Language backbone for GR00T N1.6.

    This backbone uses NVIDIA's Eagle-Block2A-2B-v2 model for vision-language
    understanding, producing embeddings that condition the action generation.
    """

    def __init__(
        self,
        model_name: str = "nvidia/Eagle-Block2A-2B-v2",
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = 16,
        reproject_vision: bool = False,
        use_flash_attention: bool = True,
        projector_dim: int = -1,
        load_bf16: bool = True,
        tune_top_llm_layers: int = 4,
        trainable_params_fp32: bool = True,
        transformers_loading_kwargs: dict | None = None,
    ):
        """Initialize Eagle backbone.

        Args:
            model_name: HuggingFace model name or local path (default: nvidia/Eagle-Block2A-2B-v2)
            tune_llm: Whether to tune the entire LLM backbone (default: False)
            tune_visual: Whether to tune the vision encoder (default: False)
            select_layer: Which hidden layer to use for features (default: 16)
            reproject_vision: Whether to reproject vision features (default: False)
            use_flash_attention: Use Flash Attention 2 (required for Eagle 3) (default: True)
            projector_dim: Dimension for projection layer (default: -1, no projection)
            load_bf16: Load model in bfloat16 (required for Eagle 3) (default: True)
            tune_top_llm_layers: Number of top LLM layers to tune (default: 4)
            trainable_params_fp32: Cast trainable parameters to fp32 (default: True)
            transformers_loading_kwargs: Additional kwargs for transformers loading
        """
        super().__init__()

        if transformers_loading_kwargs is None:
            transformers_loading_kwargs = {"trust_remote_code": True}

        # Add attention kwargs
        extra_kwargs = {}
        if use_flash_attention:
            extra_kwargs["attn_implementation"] = "flash_attention_2"
        if load_bf16:
            extra_kwargs["torch_dtype"] = torch.bfloat16

        # Try to load from local vendored path first, then from HuggingFace Hub
        if model_name == "nvidia/Eagle-Block2A-2B-v2":
            assert use_flash_attention, "nvidia/Eagle-Block2A-2B-v2 requires flash attention"
            assert load_bf16, "nvidia/Eagle-Block2A-2B-v2 requires bfloat16"

            # Try local path first (vendored model)
            local_path = os.path.join(os.path.dirname(__file__), "eagle3_hg_model")
            if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
                logger.info(f"Loading Eagle model from local path: {local_path}")
                config = AutoConfig.from_pretrained(local_path, **transformers_loading_kwargs)
                self.model = AutoModel.from_config(config, **transformers_loading_kwargs, **extra_kwargs)
            else:
                # Fall back to HuggingFace Hub
                logger.info(f"Loading Eagle model from HuggingFace Hub: {model_name}")
                config = AutoConfig.from_pretrained(model_name, **transformers_loading_kwargs)
                self.model = AutoModel.from_pretrained(
                    model_name, config=config, **transformers_loading_kwargs, **extra_kwargs
                )
        else:
            # Load from specified path or Hub
            logger.info(f"Loading Eagle model: {model_name}")
            config = AutoConfig.from_pretrained(model_name, **transformers_loading_kwargs)
            self.model = AutoModel.from_pretrained(
                model_name, config=config, **transformers_loading_kwargs, **extra_kwargs
            )

        # Truncate LLM layers to save compute (we only need up to select_layer)
        if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "model"):
            while len(self.model.language_model.model.layers) > select_layer:
                self.model.language_model.model.layers.pop(-1)
            logger.info(f"Truncated LLM to {select_layer} layers")

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual, tune_top_llm_layers)

        if load_bf16 and trainable_params_fp32:
            # Cast trainable parameters to fp32 for better training stability
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)
                    logger.debug(f"Casting trainable parameter {n} to fp32")

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool, tune_top_llm_layers: int):
        """Set which parameters are trainable.

        Args:
            tune_llm: Whether to tune the entire LLM
            tune_visual: Whether to tune the vision encoder
            tune_top_llm_layers: Number of top LLM layers to tune
        """
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual

        # First freeze everything
        for p in self.parameters():
            p.requires_grad = False

        # Then selectively unfreeze
        if tune_llm:
            if hasattr(self.model, "language_model"):
                self.model.language_model.requires_grad_(True)

        if tune_visual:
            if hasattr(self.model, "vision_model"):
                self.model.vision_model.requires_grad_(True)
            if hasattr(self.model, "mlp1"):
                self.model.mlp1.requires_grad_(True)

        # Tune top N LLM layers (N1.6 feature)
        if tune_top_llm_layers > 0 and hasattr(self.model, "language_model"):
            if hasattr(self.model.language_model, "model") and hasattr(
                self.model.language_model.model, "layers"
            ):
                for layer in self.model.language_model.model.layers[-tune_top_llm_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                logger.info(f"Tuning top {tune_top_llm_layers} LLM layers")

        # Log trainable parameters
        trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in self.parameters())
        logger.info(f"Backbone trainable parameters: {trainable_count:,} / {total_count:,}")

        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """Set frozen modules to eval mode for proper dropout/batchnorm behavior.

        HuggingFace calls model.train() at each training step, so we need to
        manually set frozen modules to eval mode.
        """
        if self.training:
            if hasattr(self.model, "language_model") and not self.tune_llm:
                self.model.language_model.eval()
            if hasattr(self.model, "vision_model") and not self.tune_visual:
                self.model.vision_model.eval()
                if hasattr(self.model, "mlp1"):
                    self.model.mlp1.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for forward pass."""
        return BatchFeature(data=batch)

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        """Forward pass through the Eagle backbone.

        Args:
            vl_input: BatchFeature containing input_ids, attention_mask, pixel_values

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, hidden_size] vision-language features
                - backbone_attention_mask: [B, seq_len] attention mask
                - image_mask: [B, seq_len] mask indicating image token positions
        """
        self.set_frozen_modules_to_eval_mode()

        # Extract required keys
        keys_to_use = ["input_ids", "attention_mask", "pixel_values"]
        vl_input_filtered = {k: vl_input[k] for k in keys_to_use if k in vl_input}

        # Forward through model
        outputs = self.model(**vl_input_filtered, output_hidden_states=True)
        hidden_states = outputs["hidden_states"][-1]

        # Create masks
        image_token_index = getattr(self.model.config, "image_token_index", -200)
        image_mask = vl_input["input_ids"] == image_token_index
        attention_mask = vl_input["attention_mask"] == 1

        return BatchFeature(
            data={
                "backbone_features": hidden_states,
                "backbone_attention_mask": attention_mask,
                "image_mask": image_mask,
            }
        )
