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
from typing import Any, TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from lerobot.policies.groot_n1d6.configuration_groot_n1d6 import GrootN1d6Config
from lerobot.policies.groot_n1d6.processing_gr00t_n1d6 import Gr00tN1d6DataCollator
from lerobot.policies.groot_n1d6.utils import build_eagle_processor
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

if TYPE_CHECKING:
    from transformers import ProcessorMixin
else:
    ProcessorMixin = object


DEFAULT_EMBODIMENT_MAPPING = {
    "robocasa_panda_omron": 13,
    "gr1": 20,
    "behavior_r1_pro": 24,
    "unitree_g1": 8,
    "libero_panda": 2,
    "oxe_google": 0,
    "oxe_widowx": 1,
    "new_embodiment": 10,
}


def _to_uint8_np_bhwc(img_t: torch.Tensor) -> np.ndarray:
    if img_t.dtype.is_floating_point:
        img_t = (img_t.clamp(0, 1) * 255.0).to(torch.uint8)
    return img_t.permute(0, 2, 3, 1).cpu().numpy()


def _formalize_language(text: str) -> str:
    lowered = text.lower()
    return "".join(ch for ch in lowered if ch.isalnum() or ch.isspace())


def make_groot_n1d6_pre_post_processors(
    config: GrootN1d6Config, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    state_horizon = 1
    action_horizon = config.action_horizon
    max_state_dim = config.max_state_dim
    max_action_dim = config.max_action_dim

    padded_stats = dataset_stats or {}

    try:
        env_action_dim = int(config.output_features["action"].shape[0])
    except Exception:
        env_action_dim = 0

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        GrootN1d6PackInputsStep(
            state_horizon=state_horizon,
            action_horizon=action_horizon,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            embodiment_tag=config.embodiment_tag,
            normalize_min_max=True,
            stats=padded_stats,
        ),
        GrootN1d6VlmContentStep(
            language_key="task",
            formalize_language=config.formalize_language,
        ),
        GrootN1d6CollateStep(),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        GrootN1d6ActionUnpackUnnormalizeStep(
            env_action_dim=env_action_dim,
            stats=padded_stats,
            normalize_min_max=True,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


@dataclass
@ProcessorStepRegistry.register(name="groot_n1d6_pack_inputs_v1")
class GrootN1d6PackInputsStep(ProcessorStep):
    state_horizon: int = 1
    action_horizon: int = 50
    max_state_dim: int = 128
    max_action_dim: int = 128
    embodiment_tag: str = "new_embodiment"
    embodiment_mapping: dict[str, int] = field(default_factory=lambda: DEFAULT_EMBODIMENT_MAPPING)
    normalize_min_max: bool = True
    stats: dict[str, dict[str, Any]] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}

        def _align_vec(vec: Any, target_dim: int, *, default: float) -> torch.Tensor:
            t = torch.as_tensor(vec)
            t = t.flatten().to(
                dtype=torch.float32,
                device=next(
                    (v.device for v in obs.values() if isinstance(v, torch.Tensor)), torch.device("cpu")
                ),
            )
            d = int(t.shape[-1]) if t.numel() > 0 else 0
            if d == target_dim:
                return t
            if d < target_dim:
                pad = torch.full((target_dim - d,), default, dtype=t.dtype, device=t.device)
                return torch.cat([t, pad], dim=0)
            return t[:target_dim]

        def _min_max_norm(x: torch.Tensor, key: str) -> torch.Tensor:
            if not self.normalize_min_max:
                return x
            if self.stats is None or key not in self.stats:
                return x
            stats_k = self.stats[key]
            last_dim = x.shape[-1]
            min_v = _align_vec(stats_k.get("min", torch.zeros(last_dim)), last_dim, default=0.0)
            max_v = _align_vec(stats_k.get("max", torch.ones(last_dim)), last_dim, default=1.0)
            denom = max_v - min_v
            mask = denom != 0
            safe_denom = torch.where(mask, denom, torch.ones_like(denom))
            mapped = 2 * (x - min_v) / safe_denom - 1
            return torch.where(mask, mapped, torch.zeros_like(mapped))

        if "observation.state" in obs:
            state = obs["observation.state"]
            if state.dim() != 2:
                raise ValueError(f"state must be (B, D), got {tuple(state.shape)}")
            bsz, d = state.shape
            state = _min_max_norm(state, "observation.state")
            state = state.unsqueeze(1)
            if d > self.max_state_dim:
                state = state[:, :, : self.max_state_dim]
                d = self.max_state_dim
            elif d < self.max_state_dim:
                pad = torch.zeros(bsz, 1, self.max_state_dim - d, dtype=state.dtype, device=state.device)
                state = torch.cat([state, pad], dim=2)
            obs["state"] = state

        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            if self.normalize_min_max:
                if action.dim() == 2:
                    action = _min_max_norm(action, "action")
                elif action.dim() == 3:
                    b, t, d = action.shape
                    flat = action.reshape(b * t, d)
                    flat = _min_max_norm(flat, "action")
                    action = flat.view(b, t, d)
            if action.dim() == 2:
                action = action.unsqueeze(1)
            elif action.dim() != 3:
                raise ValueError(f"action must be (B, D) or (B, T, D), got {tuple(action.shape)}")

            b, t, d = action.shape
            valid_horizon = min(t, self.action_horizon)
            if t < self.action_horizon:
                last = action[:, -1:, :]
                pad = last.repeat(1, self.action_horizon - t, 1)
                action = torch.cat([action, pad], dim=1)
            elif t > self.action_horizon:
                action = action[:, : self.action_horizon, :]

            if d > self.max_action_dim:
                action = action[:, :, : self.max_action_dim]
                d = self.max_action_dim
            elif d < self.max_action_dim:
                pad = torch.zeros(b, self.action_horizon, self.max_action_dim - d, dtype=action.dtype, device=action.device)
                action = torch.cat([action, pad], dim=2)

            action_mask = torch.zeros(
                b,
                self.action_horizon,
                self.max_action_dim,
                dtype=action.dtype,
                device=action.device,
            )
            action_mask[:, :valid_horizon, :d] = 1
            transition[TransitionKey.ACTION] = action
            comp["action_mask"] = action_mask

        emb_id = self.embodiment_mapping.get(self.embodiment_tag, 0)
        bsz = None
        device = torch.device("cpu")
        for v in list(obs.values()) + [transition.get(TransitionKey.ACTION)]:
            if isinstance(v, torch.Tensor):
                bsz = v.shape[0]
                device = v.device
                break
        if bsz is None:
            bsz = 1
        comp["embodiment_id"] = torch.full((bsz,), emb_id, dtype=torch.long, device=device)

        transition[TransitionKey.OBSERVATION] = obs
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return transition

    def transform_features(self, features):
        return features


@dataclass
@ProcessorStepRegistry.register(name="groot_n1d6_vlm_content_v1")
class GrootN1d6VlmContentStep(ProcessorStep):
    language_key: str = "task"
    formalize_language: bool = True
    _proc: ProcessorMixin | None = field(default=None, init=False, repr=False)

    @property
    def proc(self) -> ProcessorMixin:
        if self._proc is None:
            self._proc = build_eagle_processor({"trust_remote_code": True})
        return self._proc

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}

        img_keys = sorted([k for k in obs if k.startswith("observation.images.")])
        if not img_keys and "observation.image" in obs:
            img_keys = ["observation.image"]
        if not img_keys:
            return transition

        language = comp.get(self.language_key) or "Perform the task."
        lang_list: list[str]
        if isinstance(language, list):
            lang_list = language
        else:
            lang_list = [language]

        cams = [_to_uint8_np_bhwc(obs[k]) for k in img_keys]
        bsz = cams[0].shape[0]
        if len(lang_list) == 1:
            lang_list = lang_list * bsz
        elif len(lang_list) != bsz:
            lang_list = [lang_list[0] if lang_list else "Perform the task."] * bsz

        vlm_contents: list[dict[str, Any]] = []
        for b in range(bsz):
            images = [Image.fromarray(cams[v][b]) for v in range(len(cams))]
            text = lang_list[b] or "Perform the task."
            if self.formalize_language:
                text = _formalize_language(text)
            content = [{"type": "text", "text": text}] + [
                {"type": "image", "image": img} for img in images
            ]
            conversation = [{"role": "user", "content": content}]
            chat_text = self.proc.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
            vlm_contents.append(
                {
                    "text": chat_text,
                    "images": images,
                    "conversation": conversation,
                }
            )

        comp["vlm_content"] = vlm_contents
        for k in img_keys:
            obs.pop(k, None)

        transition[TransitionKey.OBSERVATION] = obs
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return transition

    def transform_features(self, features):
        return features


@dataclass
@ProcessorStepRegistry.register(name="groot_n1d6_collate_v1")
class GrootN1d6CollateStep(ProcessorStep):
    _collator: Gr00tN1d6DataCollator | None = field(default=None, init=False, repr=False)

    @property
    def collator(self) -> Gr00tN1d6DataCollator:
        if self._collator is None:
            self._collator = Gr00tN1d6DataCollator()
        return self._collator

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
        contents = comp.get("vlm_content")
        if not contents:
            return transition

        features = [{"vlm_content": content} for content in contents]
        batched = self.collator(features)["inputs"]
        for k, v in batched.items():
            comp[k] = v
        comp.pop("vlm_content", None)

        transition[TransitionKey.OBSERVATION] = obs
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return transition

    def transform_features(self, features):
        return features


@dataclass
@ProcessorStepRegistry.register(name="groot_n1d6_action_unpack_unnormalize_v1")
class GrootN1d6ActionUnpackUnnormalizeStep(ProcessorStep):
    env_action_dim: int = 0
    normalize_min_max: bool = True
    stats: dict[str, dict[str, Any]] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, torch.Tensor):
            return transition

        if action.dim() == 3:
            action = action[:, -1, :]
        if self.env_action_dim and action.shape[-1] >= self.env_action_dim:
            action = action[..., : self.env_action_dim]

        if self.normalize_min_max and self.stats is not None:
            stats_k = self.stats.get("action", {})
            d = action.shape[-1]
            min_v = torch.as_tensor(
                stats_k.get("min", torch.zeros(d)), dtype=action.dtype, device=action.device
            )
            max_v = torch.as_tensor(
                stats_k.get("max", torch.ones(d)), dtype=action.dtype, device=action.device
            )
            if min_v.numel() != d:
                min_v = torch.nn.functional.pad(min_v.flatten()[:d], (0, max(0, d - min_v.numel())))
                min_v = min_v.to(action.device, dtype=action.dtype)
            if max_v.numel() != d:
                max_v = torch.nn.functional.pad(max_v.flatten()[:d], (0, max(0, d - max_v.numel())))
                max_v = max_v.to(action.device, dtype=action.dtype)
            denom = max_v - min_v
            mask = denom != 0
            safe_denom = torch.where(mask, denom, torch.ones_like(denom))
            inv = (action + 1.0) * 0.5 * safe_denom + min_v
            action = torch.where(mask, inv, min_v)

        transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "env_action_dim": self.env_action_dim,
            "normalize_min_max": self.normalize_min_max,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        if not self.stats:
            return {}

        flat: dict[str, torch.Tensor] = {}
        for key, sub in self.stats.items():
            for stat_name, value in sub.items():
                tensor = torch.as_tensor(value).cpu()
                flat[f"{key}.{stat_name}"] = tensor
        return flat

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if not state:
            return

        reconstructed: dict[str, dict[str, Any]] = {}
        for flat_key, tensor in state.items():
            if "." in flat_key:
                key, stat_name = flat_key.rsplit(".", 1)
                if key not in reconstructed:
                    reconstructed[key] = {}
                reconstructed[key][stat_name] = tensor

        if reconstructed:
            self.stats = reconstructed
