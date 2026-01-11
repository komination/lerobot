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

"""GR00T N1.6 Vision-Language-Action Model."""

import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

try:
    import tree
except ImportError:
    tree = None

from lerobot.policies.groot_n1d6.action_head.action_encoder import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
from lerobot.policies.groot_n1d6.action_head.cross_attention_dit import AlternateVLDiT, DiT
from lerobot.policies.groot_n1d6.configuration_groot_n1d6 import GrootN1d6Config
from lerobot.policies.groot_n1d6.eagle_backbone_n1d6 import EagleBackbone

logger = logging.getLogger(__name__)


class Gr00tN1d6ActionHead(nn.Module):
    """Action head component for GR00T N1.6 using flow matching diffusion policy.

    This is the N1.6 version which uses:
    - 32-layer DiT (vs 16 in N1.5)
    - AlternateVLDiT for image/text attention separation
    - State dropout and noise augmentation
    """

    supports_gradient_checkpointing = True

    def __init__(self, config: GrootN1d6Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        # Get diffusion model config
        diffusion_cfg = config.diffusion_model_cfg.copy()

        # Initialize DiT or AlternateVLDiT
        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **diffusion_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            logger.info("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(**diffusion_cfg, cross_attention_dim=config.backbone_embedding_dim)
            logger.info("Using DiT for diffusion model")

        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        # State and action encoders/decoders
        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        # Vision-language layer norm
        self.vlln = nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()

        # Position embedding
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        else:
            self.position_embedding = None

        # State dropout parameters (N1.6 feature)
        self.state_dropout_prob = config.state_dropout_prob
        self.mask_token = (
            nn.Parameter(0.02 * torch.randn(1, 1, self.input_embedding_dim))
            if self.state_dropout_prob > 0
            else None
        )

        # State noise parameters (N1.6 feature)
        self.state_additive_noise_scale = config.state_additive_noise_scale

        # Flow matching parameters
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.noise_s = config.noise_s

        # Set trainable parameters
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model, config.tune_vlln)

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        """Configure which parameters are trainable."""
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln

        # First enable all gradients
        for p in self.parameters():
            p.requires_grad = True

        # Then disable based on config
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.position_embedding is not None:
                self.position_embedding.requires_grad_(False)
            if self.mask_token is not None:
                self.mask_token.requires_grad_(False)

        if not tune_diffusion_model:
            self.model.requires_grad_(False)

        if not tune_vlln:
            self.vlln.requires_grad_(False)

        # Log trainable status
        trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Action head trainable parameters: {trainable_count:,}")

    def set_frozen_modules_to_eval_mode(self):
        """Set frozen modules to eval mode for proper dropout/batchnorm behavior."""
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.position_embedding is not None:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        """Sample timesteps from beta distribution for flow matching."""
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.noise_s
        return sample

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        """Apply layer norm to backbone features."""
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """Training forward pass through the action head.

        Args:
            backbone_output: Output from backbone containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
                - image_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim]
                - embodiment_id: [B]
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing loss and other outputs
        """
        self.set_frozen_modules_to_eval_mode()
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device
        embodiment_id = action_input.embodiment_id

        # Embed state
        state = action_input.state
        if state.dim() == 2:
            state = state.unsqueeze(1)  # [B, state_dim] -> [B, 1, state_dim]
        state_features = self.state_encoder(state, embodiment_id)

        # State dropout (N1.6 feature)
        if self.state_dropout_prob > 0 and self.training:
            do_dropout = torch.rand(state_features.shape[0], device=device) < self.state_dropout_prob
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout) + self.mask_token * do_dropout

        # Add Gaussian noise to state features (N1.6 feature)
        if self.training and self.state_additive_noise_scale > 0:
            noise = torch.randn_like(state_features) * self.state_additive_noise_scale
            state_features = state_features + noise

        # Embed noised action trajectory
        actions = action_input.action
        noise = torch.randn(actions.shape, device=device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=device, dtype=actions.dtype)
        t = t[:, None, None]  # [B, 1, 1] for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Discretize timestep
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Add position embedding
        if self.position_embedding is not None:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Concatenate state and action features
        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask

        # Run through DiT
        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        # Decode to actions
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Compute loss
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """Encode features for inference."""
        backbone_output = self.process_backbone_output(backbone_output)
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state
        state = action_input.state
        if state.dim() == 2:
            state = state.unsqueeze(1)
        state_features = self.state_encoder(state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """Generate actions using flow matching diffusion.

        Args:
            backbone_output: Output from backbone
            action_input: Input containing state and embodiment_id

        Returns:
            BatchFeature containing action_pred: [B, action_horizon, action_dim]
        """
        features = self._encode_features(backbone_output, action_input)
        vl_embeds = features.backbone_features
        state_features = features.state_features
        embodiment_id = action_input.embodiment_id

        # Initialize actions as random noise
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps

        # Iteratively denoise
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Encode actions at current timestep
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)

            # Add position embedding
            if self.position_embedding is not None:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Concatenate state and action features
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run through DiT
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )

            # Decode velocity prediction
            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon :]

            # Euler integration
            actions = actions + dt * pred_velocity

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class Gr00tN1d6(PreTrainedModel):
    """GR00T N1.6 Vision-Language-Action model.

    This is the main model class that combines:
    - Eagle 3 Vision-Language backbone
    - Flow matching diffusion action head with 32-layer DiT
    """

    config_class = GrootN1d6Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: GrootN1d6Config,
        transformers_loading_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize GR00T N1.6 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Additional kwargs for transformers loading
        """
        super().__init__(config)
        self.config = config

        if transformers_loading_kwargs is None:
            transformers_loading_kwargs = {"trust_remote_code": True}

        # Initialize backbone
        self.backbone = EagleBackbone(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d6ActionHead(config)

        logger.info(
            f"Initialized Gr00tN1d6 model with {sum(p.numel() for p in self.parameters()):,} parameters"
        )

    def prepare_input(self, inputs: dict) -> tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head.

        Args:
            inputs: Dictionary containing all input tensors

        Returns:
            Tuple of (backbone_inputs, action_inputs)
        """
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = BatchFeature(data=inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if isinstance(x, torch.Tensor):
                if torch.is_floating_point(x):
                    return x.to(self.device, dtype=self.dtype)
                else:
                    return x.to(self.device)
            return x

        if tree is not None:
            backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
            action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)
        else:
            # Fallback without tree
            backbone_inputs = BatchFeature(
                data={k: to_device_with_dtype(v) for k, v in backbone_inputs.items()}
            )
            action_inputs = BatchFeature(
                data={k: to_device_with_dtype(v) for k, v in action_inputs.items()}
            )

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """Training forward pass.

        Args:
            inputs: Dictionary containing all input tensors

        Returns:
            BatchFeature containing loss and other outputs
        """
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)
        return action_outputs

    def get_action(self, inputs: dict) -> BatchFeature:
        """Inference forward pass to generate actions.

        Args:
            inputs: Dictionary containing observation inputs

        Returns:
            BatchFeature containing action predictions
        """
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
