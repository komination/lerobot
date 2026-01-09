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

from pathlib import Path
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor, ProcessorMixin
else:
    AutoProcessor = None
    ProcessorMixin = object

DEFAULT_VENDOR_EAGLE_PATH = Path(__file__).resolve().parent / "eagle3_hg_model"


def build_eagle_processor(transformers_loading_kwargs: dict | None = None) -> ProcessorMixin:
    if AutoProcessor is None:
        raise ImportError("transformers is required to build the Eagle processor")
    kwargs = transformers_loading_kwargs or {"trust_remote_code": True}
    proc = AutoProcessor.from_pretrained(str(DEFAULT_VENDOR_EAGLE_PATH), **kwargs)
    proc.tokenizer.padding_side = "left"
    return proc
