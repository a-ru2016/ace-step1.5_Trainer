"""
LoKr utilities for ACE-Step training and inference.

This module integrates LyCORIS LoKr adapters with the ACE-Step decoder.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
from loguru import logger

from acestep.training.configs import LoKRConfig
from acestep.training.path_safety import safe_path

try:
    from lycoris import LycorisNetwork, create_lycoris

    LYCORIS_AVAILABLE = True
except ImportError:
    LYCORIS_AVAILABLE = False
    LycorisNetwork = Any  # type: ignore[assignment,misc]
    logger.warning(
        "LyCORIS library not installed. LoKr training/inference unavailable. "
        "Install with: pip install lycoris-lora"
    )


def check_lycoris_available() -> bool:
    """Check if LyCORIS is importable."""
    return LYCORIS_AVAILABLE


def _matches_target_module_name(module_name: str, target_modules) -> bool:
    """Return True if a LyCORIS module name maps to one of target module suffixes."""
    if not module_name:
        return False
    name = module_name.lower()
    for target in target_modules or []:
        t = str(target).strip().lower()
        if not t:
            continue
        if name.endswith(t) or f"_{t}" in name or f".{t}" in name:
            return True
    return False


def apply_pseudo_fp8(target_model: torch.nn.Module):
    """Apply pseudo FP8 compression to a model's Linear and Conv2d layers to save VRAM."""

    from torch.nn import functional as F

    def _make_pseudo_fp8_forward(module: torch.nn.Module):
        """Create a forward closure that captures the module reference.

        For frozen weights (requires_grad=False) we avoid saving weight/bias in
        save_for_backward entirely.  Instead, backward accesses them via the
        captured ``module`` reference.  This prevents PyTorch from adding
        the (potentially large) FP8 weight tensor into the autograd graph's
        saved-tensor storage, saving significant VRAM during backward.
        """

        is_conv = isinstance(module, torch.nn.Conv2d)

        class _FrozenFP8(torch.autograd.Function):
            """Pseudo FP8 for frozen (non-trainable) layers.

            forward takes only ``inp`` so backward returns only ``(grad_input,)``.
            Weight is accessed via the captured closure – nothing is saved.
            """
            @staticmethod
            def forward(ctx, inp):
                weight = module.weight
                bias = getattr(module, "bias", None)
                ctx.save_for_backward()  # nothing saved – minimal VRAM
                
                # Proactive casting to compute dtype (usually bf16/fp16)
                w_comp = weight.to(inp.dtype)
                b_comp = bias.to(inp.dtype) if bias is not None else None
                
                if is_conv:
                    res = F.conv2d(
                        inp, w_comp, b_comp,
                        module.stride, module.padding, module.dilation, module.groups
                    )
                else:
                    res = F.linear(inp, w_comp, b_comp)
                
                # Explictly delete to free memory earlier
                del w_comp, b_comp
                return res

            @staticmethod
            def backward(ctx, grad_output):
                grad_input = None
                if ctx.needs_input_grad[0]:
                    weight = module.weight
                    w_comp = weight.to(grad_output.dtype)
                    
                    if is_conv:
                        # Conv2d backward for input
                        grad_input = torch.nn.grad.conv2d_input(
                            module.input_size if hasattr(module, 'input_size') else None, # Placeholder if needed
                            w_comp, grad_output,
                            module.stride, module.padding, module.dilation, module.groups
                        )
                        # Fallback for standard autograd if input_size is unknown/complex
                        if grad_input is None:
                            # We can use F.conv_transpose2d for input grad if needed, 
                            # but usually standard torch matmul logic for Linear is fine.
                            # For Conv2d, we rely on the fact that we can't easily do it without input shape.
                            # Standard way is to use a dummy with grad and backward it if we really need to save weight vram.
                            # BUT, most DiT layers we train are Linear. For Conv2d, we might just use standard if frozen.
                            pass
                    
                    # If we don't have a specialized fast path for Conv2d input grad while frozen weight,
                    # we might skip the 'Frozen' optimization for Conv2d if it's too complex, 
                    # but let's try to keep it simple for Linear which is 99% of DiT.
                    if not is_conv:
                        grad_input = grad_output.matmul(w_comp)
                    else:
                        # Fallback: if we can't do custom backward easily for conv without saved input, 
                        # just use the trainable path which saves input.
                        pass

                    del w_comp
                return (grad_input,)

        class _TrainableFP8(torch.autograd.Function):
            """Pseudo FP8 for trainable layers (e.g. LoKr-injected).

            Weight and bias are explicit forward arguments so PyTorch can
            properly propagate their gradients.
            """
            @staticmethod
            def forward(ctx, inp, weight, bias=None):
                ctx.save_for_backward(inp, weight, bias)
                ctx.bias_requires_grad = bias is not None and bias.requires_grad
                w_comp = weight.to(inp.dtype)
                b_comp = bias.to(inp.dtype) if bias is not None else None
                
                if is_conv:
                    res = F.conv2d(
                        inp, w_comp, b_comp,
                        module.stride, module.padding, module.dilation, module.groups
                    )
                else:
                    res = F.linear(inp, w_comp, b_comp)
                
                del w_comp, b_comp
                return res

            @staticmethod
            def backward(ctx, grad_output):
                inp, weight, bias = ctx.saved_tensors
                w_comp = weight.to(grad_output.dtype)
                grad_input = grad_weight = grad_bias = None
                
                if is_conv:
                    # For Conv2d, we use the standard autograd results by re-running if needed, 
                    # but here we just need to calculate grads.
                    # Since we have inp and weight, we can use torch.nn.grad functions.
                    if ctx.needs_input_grad[0]:
                        grad_input = torch.nn.grad.conv2d_input(inp.shape, w_comp, grad_output, module.stride, module.padding, module.dilation, module.groups)
                    if ctx.needs_input_grad[1]:
                        grad_weight = torch.nn.grad.conv2d_weight(inp, weight.shape, grad_output, module.stride, module.padding, module.dilation, module.groups)
                    if ctx.bias_requires_grad and ctx.needs_input_grad[2]:
                        grad_bias = grad_output.sum(dim=(0, 2, 3))
                else:
                    if ctx.needs_input_grad[0]:
                        grad_input = grad_output.matmul(w_comp)
                    if ctx.needs_input_grad[1]:
                        g_flat = grad_output.reshape(-1, grad_output.size(-1))
                        i_flat = inp.reshape(-1, inp.size(-1))
                        grad_weight = g_flat.transpose(-2, -1).matmul(i_flat)
                    if ctx.bias_requires_grad and ctx.needs_input_grad[2]:
                        grad_bias = grad_output.reshape(-1, grad_output.size(-1)).sum(dim=0)
                
                del w_comp
                return grad_input, grad_weight, grad_bias

        def _forward(self_module, inp: torch.Tensor) -> torch.Tensor:  # noqa: N805
            weight = self_module.weight
            bias = getattr(self_module, "bias", None)
            
            # Conv2d often has non-trainable weights that we want to save VRAM on.
            # But the 'Frozen' optimization (not saving weight/bias) depends on matmul availability in backward.
            # For Conv2d, it's safer to use _TrainableFP8 if we don't want to deal with input_size tracking for Frozen path.
            # However, most DiT layers are Linear.
            if weight.requires_grad or (bias is not None and bias.requires_grad) or is_conv:
                return _TrainableFP8.apply(inp, weight, bias)
            else:
                return _FrozenFP8.apply(inp)

        return _forward


    for module in target_model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            if hasattr(module.weight, "dtype") and module.weight.dtype != torch.float8_e4m3fn:
                with torch.no_grad():
                    module.weight.data = module.weight.data.to(torch.float8_e4m3fn)
                    if getattr(module, "bias", None) is not None:
                        module.bias.data = module.bias.data.to(torch.float8_e4m3fn)
                # Override forward with closure-based version
                module.forward = _make_pseudo_fp8_forward(module).__get__(module, module.__class__)


def inject_lokr_into_dit(
    model,
    lokr_config: LoKRConfig,
    multiplier: float = 1.0,
) -> Tuple[Any, "LycorisNetwork", Dict[str, Any]]:
    """
    Inject LoKr adapters into the model (TE and/or DiT).

    Returns:
        Tuple: (model, lycoris_network, info_dict)
    """
    if not LYCORIS_AVAILABLE:
        raise ImportError(
            "LyCORIS library is required for LoKr training. "
            "Install with: pip install lycoris-lora"
        )

    # Granular flags
    train_te = getattr(lokr_config, "train_text_encoder", False)
    train_dit = getattr(lokr_config, "train_dit", True)
    
    # Backward compatibility with train_text_encoder_only
    if getattr(lokr_config, "train_text_encoder_only", False):
        train_te = True
        train_dit = False

    if getattr(lokr_config, "use_fp8", False):
        # Apply to frozen components to save VRAM
        # We avoid applying to trained components as it can cause issues with some kernels (mul_cpu_reduced_float)
        
        # Handle TE (if not being trained)
        if not train_te:
            te = getattr(model, "text_encoder", None)
            if te is None and hasattr(model, "text_encoders"):
                te = model.text_encoders
            if te is not None:
                apply_pseudo_fp8(te)
        
        # Handle DiT (if not being trained)
        if not train_dit:
            decoder = getattr(model, "decoder", None)
            if decoder is not None:
                apply_pseudo_fp8(decoder)
        
        logger.info("Pseudo FP8 applied to frozen components.")

    # Freeze strategy
    if train_te and train_dit:
        target_root = model
    elif train_te:
        target_root = getattr(model, "text_encoder", model)
    else:
        target_root = getattr(model, "decoder", model)

    # Freeze all existing params in the target tree
    for _, param in target_root.named_parameters():
        param.requires_grad = False

    LycorisNetwork.apply_preset(
        {
            "unet_target_name": lokr_config.target_modules,
            "target_name": lokr_config.target_modules,
        }
    )

    lycoris_net = create_lycoris(
        target_root,
        multiplier,
        linear_dim=lokr_config.linear_dim,
        linear_alpha=lokr_config.linear_alpha,
        algo="lokr",
        factor=lokr_config.factor,
        decompose_both=lokr_config.decompose_both,
        use_tucker=lokr_config.use_tucker,
        use_scalar=lokr_config.use_scalar,
        full_matrix=lokr_config.full_matrix,
        bypass_mode=lokr_config.bypass_mode,
        rs_lora=lokr_config.rs_lora,
        unbalanced_factorization=lokr_config.unbalanced_factorization,
    )

    if lokr_config.weight_decompose:
        try:
            lycoris_net = create_lycoris(
                target_root,
                multiplier,
                linear_dim=lokr_config.linear_dim,
                linear_alpha=lokr_config.linear_alpha,
                algo="lokr",
                factor=lokr_config.factor,
                decompose_both=lokr_config.decompose_both,
                use_tucker=lokr_config.use_tucker,
                use_scalar=lokr_config.use_scalar,
                full_matrix=lokr_config.full_matrix,
                bypass_mode=lokr_config.bypass_mode,
                rs_lora=lokr_config.rs_lora,
                unbalanced_factorization=lokr_config.unbalanced_factorization,
                dora_wd=True,
            )
        except Exception as exc:
            logger.warning(f"DoRA mode not supported in current LyCORIS build: {exc}")

    lycoris_net.apply_to()

    # Keep a reference on target_root so it stays discoverable after wrappers.
    target_root._lycoris_net = lycoris_net

    lokr_param_list = []
    enabled_module_count = 0
    disabled_module_count = 0
    disabled_examples = []

    for idx, module in enumerate(getattr(lycoris_net, "loras", []) or []):
        module_name = (
            getattr(module, "lora_name", None)
            or getattr(module, "name", None)
            or f"{module.__class__.__name__}#{idx}"
        )
        enabled = _matches_target_module_name(module_name, lokr_config.target_modules)

        if enabled:
            enabled_module_count += 1
        else:
            disabled_module_count += 1
            if len(disabled_examples) < 8:
                disabled_examples.append(module_name)

        for param in module.parameters():
            param.requires_grad = enabled
            if enabled:
                lokr_param_list.append(param)

    logger.info(
        f"LoKr target filter: enabled {enabled_module_count} LyCORIS modules "
        f"(disabled {disabled_module_count}) for targets={lokr_config.target_modules}"
    )

    if not lokr_param_list:
        for param in lycoris_net.parameters():
            param.requires_grad = True
            lokr_param_list.append(param)

    # De-duplicate possible shared params.
    unique_params = {id(p): p for p in lokr_param_list}
    total_params = sum(p.numel() for p in target_root.parameters())
    lokr_params = sum(p.numel() for p in unique_params.values())
    trainable_params = sum(p.numel() for p in unique_params.values() if p.requires_grad)

    info = {
        "total_params": total_params,
        "lokr_params": lokr_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0.0,
        "linear_dim": lokr_config.linear_dim,
        "linear_alpha": lokr_config.linear_alpha,
        "factor": lokr_config.factor,
        "algo": "lokr",
        "target_modules": lokr_config.target_modules,
        "train_te": train_te,
        "train_dit": train_dit,
    }

    logger.info(f"LoKr injected (TE={train_te}, DiT={train_dit})")
    logger.info(
        f"LoKr trainable params: {trainable_params:,}/{total_params:,} "
        f"({info['trainable_ratio']:.2%})"
    )
    return model, lycoris_net, info


def save_lokr_weights(
    lycoris_net: "LycorisNetwork",
    output_dir: str,
    dtype: Optional[torch.dtype] = None,
    metadata: Optional[Dict[str, str]] = None,
    is_text_encoder_only: bool = False, # Deprecated
) -> str:
    """Save LoKr weights to safetensors, converting keys to ComfyUI format."""
    output_dir = safe_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "lokr_weights.safetensors")

    save_metadata: Dict[str, str] = {"algo": "lokr", "format": "lycoris", "converted_to": "comfyui_acestep"}
    if metadata:
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, str):
                save_metadata[key] = value
            else:
                save_metadata[key] = json.dumps(value, ensure_ascii=True)

    # First use LyCORIS native save_weights to ensure all internal formatting/alpha passes
    lycoris_net.save_weights(weights_path, dtype=dtype, metadata=save_metadata)

    # ── Key conversion for ComfyUI compatibility ──
    #
    # ACEStep 1.5 ComfyUI lora.py registers key mappings in model_lora_keys_unet:
    #   key_map["lycoris_{}".format(key_lora.replace(".", "_"))] = k
    #   where key_lora = k[len("diffusion_model.decoder."):-len(".weight")]
    #
    # This means ComfyUI already expects lycoris_ prefixed keys that match the
    # decoder's module path with dots replaced by underscores. For example:
    #   lycoris_layers_0_self_attn_q_proj  →  diffusion_model.decoder.layers.0.self_attn.q_proj.weight
    #   lycoris_condition_embedder         →  diffusion_model.decoder.condition_embedder.weight
    #
    # LyCORIS generates keys in exactly this format (lycoris_<module_path_underscored>.suffix),
    # so for decoder training, NO key conversion is needed at all.
    #
    # For text encoder training, ComfyUI's model_lora_keys_clip registers:
    #   key_map["text_encoders.{}".format(k[:-len(".weight")])] = k
    # where k comes from ACE15TEModel.state_dict(), e.g.:
    #   "qwen3_06b.transformer.model.layers.0.self_attn.q_proj.weight"
    # So the expected lora key is:
    #   "text_encoders.qwen3_06b.transformer.model.layers.0.self_attn.q_proj"
    #
    # LyCORIS wraps the text encoder directly, producing keys like:
    #   "lycoris_model_layers_0_self_attn_q_proj.lokr_w1"
    # These need to be converted to:
    #   "text_encoders.qwen3_06b.transformer.model.layers.0.self_attn.q_proj.lokr_w1"

    import re
    from collections import OrderedDict
    from safetensors.torch import save_file
    from safetensors import safe_open

    with safe_open(weights_path, framework="pt", device="cpu") as sf:
        raw_tensors = {k: sf.get_tensor(k) for k in sf.keys()}
        updated_metadata = sf.metadata() or {}

    converted_state_dict = OrderedDict()

    # --- Key conversion logic ---
    def _restore_dots_te(underscored_path: str) -> str:
        # Split "layers_0_self_attn_q_proj.lokr_w1" → ("layers_0_self_attn_q_proj", "lokr_w1")
        dot_pos = underscored_path.rfind(".")
        if dot_pos == -1:
            module_part = underscored_path
            suffix = ""
        else:
            module_part = underscored_path[:dot_pos]
            suffix = underscored_path[dot_pos:]

        # Restore dots at known structural boundaries
        restored = re.sub(r"layers_(\d+)_", r"layers.\1.", module_part)
        restored = re.sub(r"self_attn_([qkvo]_(?:proj|norm))", r"self_attn.\1", restored)
        restored = re.sub(r"self_attn_([qk]_norm)", r"self_attn.\1", restored)
        restored = re.sub(r"cross_attn_([qkvo]_(?:proj|norm))", r"cross_attn.\1", restored)
        restored = re.sub(r"cross_attn_([qk]_norm)", r"cross_attn.\1", restored)
        restored = re.sub(r"mlp_(gate_proj|up_proj|down_proj)", r"mlp.\1", restored)
        
        # model_norm -> model.norm
        restored = restored.replace("model_norm", "model.norm")
        # model_embed_tokens -> model.embed_tokens
        restored = restored.replace("model_embed_tokens", "model.embed_tokens")

        return restored + suffix

    te_model_prefix = "text_encoders.qwen3_06b.transformer.model."
    
    # regex for Identifying components when injected at root
    # lycoris_text_encoder_model_layers_... or lycoris_text_encoder_layers_...
    # We want to capture everything after lycoris_text_encoder_ and if it starts with model_, we can optionally strip it or keep it depending on prefix.
    # Actually, let's keep it simple: capture everything after lycoris_text_encoder_ and then decide on stripping.
    _ROOT_TE_RE = re.compile(r"^lycoris_text_encoder_(.+)$")
    _ROOT_DIT_RE = re.compile(r"^lycoris_decoder_(.+)$")
    
    # 2. Direct injection (standalone TE training or fallback)
    _DIRECT_TE_RE = re.compile(r"^lycoris_(?:(?:qwen3_[0-9a-z]+_)?model_|model_)?(.+)$")
    
    # Determine if we are in dual mode by checking keys
    has_decoder_keys = any(k.startswith("lycoris_decoder_") for k in raw_tensors.keys())
    has_te_keys = any(k.startswith("lycoris_text_encoder_") for k in raw_tensors.keys())
    dual_mode = has_decoder_keys or has_te_keys

    for lycoris_key, tensor in raw_tensors.items():
        comfyui_key = None
        
        if dual_mode:
            # Component-prefixed keys (Root injection)
            m_te = _ROOT_TE_RE.match(lycoris_key)
            if m_te:
                inner_path = m_te.group(1)
                # Strip leading 'model_' if present to match te_model_prefix which ALREADY has it
                if inner_path.startswith("model_"):
                    inner_path = inner_path[len("model_"):]
                comfyui_key = te_model_prefix + _restore_dots_te(inner_path)
            else:
                m_dit = _ROOT_DIT_RE.match(lycoris_key)
                if m_dit:
                    # ComfyUI expects lycoris_ prefix for decoder
                    comfyui_key = "lycoris_" + m_dit.group(1)
        
        if comfyui_key is None:
            # Traditional/Fallback mode (Direct injection)
            # If it's TE, it should look like lycoris_model_...
            # If it's Decoder, it just stays as is (it's already lycoris_...)
            
            # Use metadata to guess if it's TE-only if we can't tell from key
            is_te_from_meta = False
            if metadata and "lokr_config" in metadata:
                try:
                    cfg = metadata["lokr_config"]
                    if isinstance(cfg, str): cfg = json.loads(cfg)
                    if cfg.get("train_text_encoder_only", False) or cfg.get("train_text_encoder", False):
                        is_te_from_meta = True
                except Exception: pass

            m_te_dir = _DIRECT_TE_RE.match(lycoris_key)
            if m_te_dir and (is_te_from_meta or lycoris_key.startswith("lycoris_model_") or lycoris_key.startswith("lycoris_qwen3_")):
                comfyui_key = te_model_prefix + _restore_dots_te(m_te_dir.group(1))
            else:
                comfyui_key = lycoris_key

        converted_state_dict[comfyui_key] = tensor

    updated_metadata["converted_to"] = "comfyui_acestep"
    tmp_path = weights_path + ".tmp"
    save_file(converted_state_dict, tmp_path, metadata=updated_metadata)

    # Delete references to safely release memory map on Windows
    del converted_state_dict
    del raw_tensors
    import gc
    gc.collect()

    os.replace(tmp_path, weights_path)
    logger.info(f"LoKr weights saved for ComfyUI (0.6B compatible) at {weights_path}.")
    return weights_path


def load_lokr_weights(lycoris_net: "LycorisNetwork", weights_path: str) -> Dict[str, Any]:
    """Load LoKr weights into an injected LyCORIS network."""
    weights_path = safe_path(weights_path)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"LoKr weights not found: {weights_path}")
    result = lycoris_net.load_weights(weights_path)
    logger.info(f"LoKr weights loaded from {weights_path}")
    return result


def save_lokr_training_checkpoint(
    lycoris_net: "LycorisNetwork",
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    output_dir: str,
    lokr_config: Optional[LoKRConfig] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save LoKr weights plus optimizer/scheduler state."""
    output_dir = safe_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    metadata: Dict[str, Any] = {}
    if lokr_config is not None:
        metadata["lokr_config"] = lokr_config.to_dict()
    if run_metadata is not None:
        metadata["run_metadata"] = run_metadata
    metadata = metadata or None
    save_lokr_weights(lycoris_net, output_dir, metadata=metadata)

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if lokr_config is not None:
        state["lokr_config"] = lokr_config.to_dict()
    if run_metadata is not None:
        state["run_metadata"] = run_metadata

    state_path = os.path.join(output_dir, "training_state.pt")
    torch.save(state, state_path)
    logger.info(f"LoKr checkpoint saved to {output_dir} (epoch={epoch}, step={global_step})")
    return output_dir


def load_lokr_training_checkpoint(
    checkpoint_dir: str,
    optimizer=None,
    scheduler=None,
    device: torch.device = None,
) -> Dict[str, Any]:
    """Load LoKr training checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        optimizer: Optimizer instance to load state into (optional)
        scheduler: Scheduler instance to load state into (optional)
        device: Device to load tensors to

    Returns:
        Dictionary with checkpoint info:
        - epoch: Saved epoch number
        - global_step: Saved global step
        - adapter_path: Path to adapter weights
        - loaded_optimizer: Whether optimizer state was loaded
        - loaded_scheduler: Whether scheduler state was loaded
    """
    result = {
        "epoch": 0,
        "global_step": 0,
        "adapter_path": None,
        "loaded_optimizer": False,
        "loaded_scheduler": False,
    }

    # Validate checkpoint directory
    try:
        safe_dir = safe_path(checkpoint_dir)
    except ValueError:
        logger.warning(f"Rejected unsafe checkpoint directory: {checkpoint_dir!r}")
        return result

    # Find adapter path (safe_dir is already validated)
    adapter_path = os.path.join(safe_dir, "adapter")
    if os.path.isdir(adapter_path):
        result["adapter_path"] = adapter_path
    elif os.path.isdir(safe_dir):
        result["adapter_path"] = safe_dir

    # Load training state
    state_path = os.path.join(safe_dir, "training_state.pt")
    if os.path.isfile(state_path):
        try:
            training_state = torch.load(state_path, map_location=device, weights_only=True)

            if "epoch" in training_state:
                result["epoch"] = int(training_state["epoch"])
            if "global_step" in training_state:
                result["global_step"] = int(training_state["global_step"])

            # Load optimizer state if provided
            if optimizer is not None and "optimizer_state_dict" in training_state:
                optimizer.load_state_dict(training_state["optimizer_state_dict"])
                result["loaded_optimizer"] = True

            # Load scheduler state if provided
            if scheduler is not None and "scheduler_state_dict" in training_state:
                scheduler.load_state_dict(training_state["scheduler_state_dict"])
                result["loaded_scheduler"] = True

            logger.info(
                f"Loaded LoKr checkpoint from epoch {result['epoch']}, step {result['global_step']}"
            )
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning(f"Failed to load training_state.pt: {e}")
    else:
        # Fallback: extract epoch from path
        import re
        match = re.search(r"epoch_(\d+)", safe_dir)
        if match:
            result["epoch"] = int(match.group(1))
            logger.info(
                f"No training_state.pt found, extracted epoch {result['epoch']} from path"
            )

    return result
