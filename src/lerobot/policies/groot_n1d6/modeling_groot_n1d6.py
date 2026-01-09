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

import os
from collections import deque

import torch
from torch import Tensor

from lerobot.policies.groot_n1d6.configuration_groot_n1d6 import GrootN1d6Config
from lerobot.policies.groot_n1d6.groot_n1d6 import Gr00tN1d6
from lerobot.policies.pretrained import PreTrainedPolicy


class GrootN1d6Policy(PreTrainedPolicy):
    """Wrapper around GR00T N1.6 model for LeRobot integration."""

    name = "groot_n1d6"
    config_class = GrootN1d6Config

    def __init__(self, config: GrootN1d6Config, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._groot_model = self._create_groot_model()

        self.reset()

    def _create_groot_model(self):
        self._handle_flash_attention_compatibility()

        model = Gr00tN1d6.from_pretrained(
            pretrained_model_name_or_path=self.config.base_model_path,
            tune_llm=self.config.tune_llm,
            tune_visual=self.config.tune_visual,
            tune_projector=self.config.tune_projector,
            tune_diffusion_model=self.config.tune_diffusion_model,
            tune_vlln=self.config.tune_vlln,
            tune_top_llm_layers=self.config.tune_top_llm_layers,
            backbone_trainable_params_fp32=self.config.backbone_trainable_params_fp32,
        )

        return model

    def reset(self):
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        allowed_base = {"state", "action", "action_mask", "embodiment_id"}
        allowed_vlm = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "vlm_content"}
        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k in allowed_vlm) and not (k.startswith("next.") or k == "info")
        }

        device = next(self.parameters()).device
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.forward(groot_inputs)

        loss = outputs.get("loss")
        loss_dict = {"loss": loss.item()}

        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        allowed_base = {"state", "embodiment_id"}
        allowed_vlm = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "vlm_content"}
        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k in allowed_vlm) and not (k.startswith("next.") or k == "info")
        }

        device = next(self.parameters()).device
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.get_action(groot_inputs)

        actions = outputs.get("action_pred")
        original_action_dim = self.config.output_features["action"].shape[0]
        actions = actions[:, :, :original_action_dim]
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def _handle_flash_attention_compatibility(self) -> None:
        os.environ.setdefault("FLASH_ATTENTION_FORCE_BUILD", "0")
        os.environ.setdefault("FLASH_ATTENTION_SKIP_CUDA_BUILD", "0")

        try:
            import flash_attn

            print(f"[GROOT N1.6] Flash Attention version: {flash_attn.__version__}")
        except ImportError as exc:
            print(f"[GROOT N1.6] Flash Attention not available: {exc}")
            print("[GROOT N1.6] Flash Attention is required for Eagle-Block2A-2B-v2")
        except Exception as exc:
            if "undefined symbol" in str(exc):
                print(f"[GROOT N1.6] Flash Attention compatibility issue detected: {exc}")
                print("[GROOT N1.6] This is likely due to PyTorch/Flash Attention version mismatch")
            else:
                print(f"[GROOT N1.6] Flash Attention error: {exc}")
