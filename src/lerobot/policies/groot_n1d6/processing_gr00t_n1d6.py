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
from typing import Any

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from lerobot.policies.groot_n1d6.utils import build_eagle_processor


@dataclass
class Gr00tN1d6DataCollator:
    model_type: str = "eagle"
    transformers_loading_kwargs: dict = field(default_factory=lambda: {"trust_remote_code": True})
    _proc: Any | None = field(default=None, init=False, repr=False)

    @property
    def proc(self):
        if self._proc is None:
            self._proc = build_eagle_processor(self.transformers_loading_kwargs)
        return self._proc

    def __call__(self, features: list[dict[str, Any]]) -> BatchFeature:
        batch: dict[str, Any] = {}
        keys = list(set().union(*(elem.keys() for elem in features)))

        for key in keys:
            values = [elem[key] for elem in features if key in elem]
            if key == "vlm_content":
                text_list: list[str] = []
                image_inputs: list[Any] = []
                conversations = []
                for v in values:
                    text_list.append(v["text"])
                    image_inputs += v["images"]
                    conversations.append(v["conversation"])

                if self.model_type == "eagle":
                    image_inputs, _ = self.proc.process_vision_info(conversations)
                vlm_inputs = self.proc(
                    text=text_list, images=image_inputs, return_tensors="pt", padding=True
                )
                for k, v in vlm_inputs.items():
                    batch[k] = v
            elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
                raise ValueError("Unexpected VLM keys present before collation")
            else:
                batch[key] = torch.from_numpy(np.stack(values))
        return BatchFeature(data={"inputs": batch})
