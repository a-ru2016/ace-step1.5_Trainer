"""
FixedLoRAModule -- Corrected adapter training step for ACE-Step V2.

This module contains the ``FixedLoRAModule`` (nn.Module) responsible for
the per-step training logic: CFG dropout, logit-normal timestep sampling,
flow-matching interpolation, and the decoder forward pass.

Also includes small device/dtype/precision helpers used by both the
Fabric and basic training loops.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ACE-Step utilities
from acestep.training.lora_utils import (
    check_peft_available,
    inject_lora_into_dit,
)
from acestep.training.lokr_utils import (
    check_lycoris_available,
    inject_lokr_into_dit,
)

# V2 modules
from acestep.training_v2.configs import LoRAConfigV2, LoKRConfigV2, TrainingConfigV2
from acestep.training_v2.timestep_sampling import apply_cfg_dropout, sample_timesteps
from acestep.training_v2.ui import TrainingUpdate

# Union type for adapter configs
AdapterConfig = Union[LoRAConfigV2, LoKRConfigV2]


class _LastLossAccessor:
    """Lightweight wrapper that provides ``[-1]`` and bool access.

    Avoids storing an unbounded list of floats while keeping backward
    compatibility with code that reads ``module.training_losses[-1]``
    or checks ``if module.training_losses:``.
    """

    def __init__(self, module: "FixedLoRAModule") -> None:
        self._module = module
        self._has_value = False

    def append(self, value: float) -> None:
        self._module.last_training_loss = value
        self._has_value = True

    def __getitem__(self, idx: int) -> float:
        if idx == -1 or idx == 0:
            return self._module.last_training_loss
        raise IndexError("only index -1 or 0 is supported")

    def __bool__(self) -> bool:
        return self._has_value

    def __len__(self) -> int:
        return 1 if self._has_value else 0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_device_type(device: Any) -> str:
    if isinstance(device, torch.device):
        return device.type
    if isinstance(device, str):
        return device.split(":", 1)[0]
    return str(device)


def _select_compute_dtype(device_type: str) -> torch.dtype:
    if device_type in ("cuda", "xpu"):
        return torch.bfloat16
    if device_type == "mps":
        return torch.float16
    return torch.float32


def _select_fabric_precision(device_type: str) -> str:
    if device_type in ("cuda", "xpu"):
        return "bf16-mixed"
    if device_type == "mps":
        return "16-mixed"
    return "32-true"


# ===========================================================================
# FixedLoRAModule -- corrected training step
# ===========================================================================

class FixedLoRAModule(nn.Module):
    """Adapter training module with corrected timestep sampling and CFG dropout.

    Supports both LoRA (PEFT) and LoKR (LyCORIS) adapters.  The training
    step is identical for both -- only the injection and weight format differ.

    Training flow (per step):
        1. Load pre-computed tensors (from ``PreprocessedDataModule``).
        2. Apply **CFG dropout** on ``encoder_hidden_states``.
        3. Sample noise ``x1`` and continuous timestep ``t`` via
           ``sample_timesteps()`` (logit-normal).
        4. Interpolate ``x_t = t * x1 + (1 - t) * x0``.
        5. Forward through decoder, compute flow matching loss.
    """

    def __init__(
        self,
        model: nn.Module,
        adapter_config: AdapterConfig,
        training_config: TrainingConfigV2,
        device: torch.device,
        dtype: torch.dtype,
        dit_handler: Any = None,
    ) -> None:
        super().__init__()

        self.adapter_config = adapter_config
        self.adapter_type = training_config.adapter_type
        self.training_config = training_config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.device_type = _normalize_device_type(self.device)
        self.dtype = _select_compute_dtype(self.device_type)
        self.transfer_non_blocking = self.device_type in ("cuda", "xpu")
        self.dit_handler = dit_handler

        # Granular flags
        self.train_te = getattr(adapter_config, "train_text_encoder", False)
        self.train_dit = getattr(adapter_config, "train_dit", True)
        
        # Backward compatibility
        if getattr(adapter_config, "train_text_encoder_only", False):
            self.train_te = True
            self.train_dit = False

        # LyCORIS network reference (only set for LoKR)
        self.lycoris_net: Any = None
        self.adapter_info: Dict[str, Any] = {}

        # -- Adapter injection -----------------------------------------------
        if self.adapter_type == "lokr":
            self._inject_lokr(model, adapter_config)  # type: ignore[arg-type]
        else:
            self._inject_lora(model, adapter_config)  # type: ignore[arg-type]

        # Backward-compat alias
        self.lora_info = self.adapter_info

        # Model config (for timestep params read at runtime)
        # If we injected at root, model.config is correct. 
        # If we injected into sub-component, it might also be correct or available via dit_handler.
        self.config = getattr(model, "config", getattr(dit_handler, "config", None))

        # -- Null condition embedding for CFG dropout ------------------------
        if hasattr(model, "null_condition_emb"):
            self._null_cond_emb = model.null_condition_emb
        elif hasattr(model, "decoder") and hasattr(model.decoder, "null_condition_emb"):
            self._null_cond_emb = model.decoder.null_condition_emb
        elif dit_handler is not None and hasattr(dit_handler.model, "null_condition_emb"):
            self._null_cond_emb = dit_handler.model.null_condition_emb
        else:
            self._null_cond_emb = None
            logger.warning(
                "[WARN] model.null_condition_emb not found -- CFG dropout disabled"
            )

        # -- Timestep sampling params ----------------------------------------
        self._timestep_mu = training_config.timestep_mu
        self._timestep_sigma = training_config.timestep_sigma
        self._data_proportion = training_config.data_proportion
        self._cfg_ratio = training_config.cfg_ratio

        self.force_input_grads_for_checkpointing: bool = False
        self.last_training_loss: float = 0.0
        self.training_losses = _LastLossAccessor(self)

    # -----------------------------------------------------------------------
    # Adapter injection helpers
    # -----------------------------------------------------------------------

    def _inject_lora(self, model: nn.Module, cfg: LoRAConfigV2) -> None:
        """Inject LoRA adapters via PEFT."""
        if not check_peft_available():
            raise RuntimeError("PEFT is required for LoRA training.")
        self.model, self.adapter_info = inject_lora_into_dit(model, cfg)
        logger.info("[OK] LoRA injected: %s trainable params", f"{self.adapter_info['trainable_params']:,}")

    def _inject_lokr(self, model: nn.Module, cfg: LoKRConfigV2) -> None:
        """Inject LoKR adapters via LyCORIS."""
        if not check_lycoris_available():
            raise RuntimeError("LyCORIS is required for LoKR training.")
        
        # Use common inject function from lokr_utils which now handles TE+DiT
        self.model, self.lycoris_net, self.adapter_info = inject_lokr_into_dit(model, cfg)
        
        self.model = self.model.to(self.device)
        logger.info(
            "[OK] LoKR injected (TE=%s, DiT=%s): %s trainable params (moved to %s)",
            self.train_te, self.train_dit,
            f"{self.adapter_info['trainable_params']:,}",
            self.device,
        )

    # -----------------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------------

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single training step with corrected timestep sampling + CFG dropout."""
        import torch.utils.checkpoint as torch_ckpt
        from math import ceil

        if self.device_type in ("cuda", "xpu", "mps"):
            autocast_ctx = torch.autocast(device_type=self.device_type, dtype=self.dtype)
        else:
            autocast_ctx = nullcontext()

        with autocast_ctx:
            nb = self.transfer_non_blocking

            target_latents = batch["target_latents"].to(self.device, dtype=self.dtype, non_blocking=nb)
            attention_mask = batch["attention_mask"].to(self.device, dtype=self.dtype, non_blocking=nb)
            context_latents = batch["context_latents"].to(self.device, dtype=self.dtype, non_blocking=nb)

            # --- Text Encoder Conditioning (TE Trainable or Frozen) ---
            if self.train_te or (self.dit_handler is not None and getattr(self.adapter_config, "train_text_encoder_only", False)):
                if self.dit_handler is None:
                    raise ValueError("dit_handler is required for text encoder related training")
                
                metadata_list = batch.get("metadata", [])
                if not metadata_list:
                    raise ValueError("metadata missing in batch, required for TE training.")
                
                bsz = len(metadata_list)
                captions = [m.get("caption", "") for m in metadata_list]
                lyrics = [m.get("lyrics", "") for m in metadata_list]
                parsed_metas = [m.get("parsed_meta", "") for m in metadata_list]
                vocal_languages = [m.get("language", "") for m in metadata_list]

                for i in range(bsz):
                    if not parsed_metas[i]:
                        meta_dict = metadata_list[i]
                        parsed_metas[i] = f"- bpm: {meta_dict.get('bpm','N/A')}\n- timesignature: {meta_dict.get('timesignature','N/A')}\n- keyscale: {meta_dict.get('keyscale','N/A')}\n- duration: {ceil(float(meta_dict.get('duration',30)))} seconds"

                (
                    text_inputs,
                    text_token_ids,
                    text_attention_mask,
                    lyric_token_ids,
                    lyric_attention_mask,
                    _, _
                ) = self.dit_handler._prepare_text_conditioning_inputs(
                    batch_size=bsz,
                    instructions=[""] * bsz,
                    captions=captions,
                    lyrics=lyrics,
                    parsed_metas=parsed_metas,
                    vocal_languages=vocal_languages,
                    audio_cover_strength=1.0,
                )
                
                text_token_ids = text_token_ids.to(self.device, non_blocking=nb)
                text_attention_mask = text_attention_mask.to(self.device, non_blocking=nb)

                # Which TE model to use?
                te_model = getattr(self.model, "text_encoder", self.model if not hasattr(self.model, "decoder") else None)
                if te_model is None:
                    raise ValueError("Could not find text_encoder in self.model")

                embed_layer = getattr(te_model, "embed_tokens", None)
                if embed_layer is None and hasattr(te_model, "model") and hasattr(te_model.model, "embed_tokens"):
                    embed_layer = te_model.model.embed_tokens
                
                def _text_encoder_forward(tokens, mask):
                    return te_model(
                        inputs_embeds=embed_layer(tokens),
                        attention_mask=mask,
                        output_hidden_states=False,
                    )[0]

                # Use checkpointing for TE to save VRAM
                prompt_hidden_states = torch_ckpt.checkpoint(
                    _text_encoder_forward,
                    text_token_ids,
                    text_attention_mask,
                    use_reentrant=False
                )
                
                # Project (always frozen or handled by dit_handler)
                dit_encoder = getattr(self.dit_handler.model, "encoder", None)
                text_projector = dit_encoder.text_projector
                text_hidden_states = text_projector(prompt_hidden_states)

                # Reconstruct encoder_hidden_states
                original_ehs = batch["encoder_hidden_states"].to(self.device, dtype=self.dtype, non_blocking=nb)
                original_mask = batch["encoder_attention_mask"].to(self.device, dtype=self.dtype, non_blocking=nb)
                
                reconstructed_ehs = []
                reconstructed_mask = []
                
                for i in range(bsz):
                    valid_total = int(original_mask[i].sum().item())
                    text_len = int(text_attention_mask[i].sum().item())
                    lyric_timbre_len = max(0, valid_total - text_len)
                    
                    lyric_timbre_ehs = original_ehs[i, :lyric_timbre_len]
                    lyric_timbre_mask = original_mask[i, :lyric_timbre_len]
                    new_text_ehs = text_hidden_states[i, :text_len]
                    new_text_mask = text_attention_mask[i, :text_len]
                    
                    reconstructed_ehs.append(torch.cat([lyric_timbre_ehs, new_text_ehs], dim=0))
                    reconstructed_mask.append(torch.cat([lyric_timbre_mask, new_text_mask], dim=0))

                max_new_len = max(len(x) for x in reconstructed_ehs)
                final_ehs = []
                final_mask = []
                for i in range(bsz):
                    ehs_i = reconstructed_ehs[i]
                    mask_i = reconstructed_mask[i]
                    pad_len = max_new_len - len(ehs_i)
                    if pad_len > 0:
                        ehs_i = torch.cat([ehs_i, torch.zeros((pad_len, ehs_i.shape[1]), device=ehs_i.device, dtype=ehs_i.dtype)], dim=0)
                        mask_i = torch.cat([mask_i, torch.zeros(pad_len, device=mask_i.device, dtype=mask_i.dtype)], dim=0)
                    final_ehs.append(ehs_i)
                    final_mask.append(mask_i)

                encoder_hidden_states = torch.stack(final_ehs)
                encoder_attention_mask = torch.stack(final_mask)
            else:
                encoder_hidden_states = batch["encoder_hidden_states"].to(self.device, dtype=self.dtype, non_blocking=nb)
                encoder_attention_mask = batch["encoder_attention_mask"].to(self.device, dtype=self.dtype, non_blocking=nb)

            bsz = target_latents.shape[0]

            if self._null_cond_emb is not None and self._cfg_ratio > 0.0:
                encoder_hidden_states = apply_cfg_dropout(
                    encoder_hidden_states,
                    self._null_cond_emb,
                    cfg_ratio=self._cfg_ratio,
                )

            x1 = torch.randn_like(target_latents)
            x0 = target_latents

            t, r = sample_timesteps(
                batch_size=bsz,
                device=self.device,
                dtype=self.dtype,
                data_proportion=self._data_proportion,
                timestep_mu=self._timestep_mu,
                timestep_sigma=self._timestep_sigma,
                use_meanflow=False,
            )
            t_ = t.unsqueeze(-1).unsqueeze(-1)
            xt = t_ * x1 + (1.0 - t_) * x0
            if self.force_input_grads_for_checkpointing:
                xt = xt.requires_grad_(True)

            # --- DiT Forward ---
            decoder = getattr(self.model, "decoder", self.model if hasattr(self.model, "decoder") else None)
            if decoder is None and self.dit_handler is not None:
                decoder = self.dit_handler.model.decoder
            
            if decoder is None:
                raise ValueError("Could not find decoder in self.model or dit_handler")

            # Use checkpointing for decoder if DiT is NOT being trained (frozen) OR if explicitly forced
            # Actually, we always use it if training TE to save VRAM.
            if self.train_te or self.force_input_grads_for_checkpointing:
                 decoder_outputs = torch_ckpt.checkpoint(
                    decoder,
                    xt, t, t, attention_mask,
                    encoder_hidden_states, encoder_attention_mask,
                    context_latents,
                    use_reentrant=False
                )
            else:
                decoder_outputs = decoder(
                    hidden_states=xt,
                    timestep=t,
                    timestep_r=t,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    context_latents=context_latents,
                )

            flow = x1 - x0
            diffusion_loss = F.mse_loss(decoder_outputs[0], flow)

        diffusion_loss = diffusion_loss.float()
        self.training_losses.append(diffusion_loss.item())
        return diffusion_loss
