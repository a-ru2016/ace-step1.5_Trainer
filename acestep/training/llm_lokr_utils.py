"""
LLM LoKr Utilities for ACE-Step 1.5

Utilities for injecting LoKr adapters into the 5Hz-LM (Qwen2.5-based) language model.

The 5Hz-LM generates audio semantic tokens in the format:
  <think>
  bpm: 120
  caption: A calm piano melody
  duration: 180
  keyscale: C major
  language: en
  timesignature: 4
  </think>
  <|audio_code_12345|><|audio_code_67890|>...

Audio codes are discrete tokens representing quantized audio latents (codebook size: 64000).
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

import torch

try:
    from lycoris import LycorisNetwork, create_lycoris
    LYCORIS_AVAILABLE = True
except ImportError:
    LYCORIS_AVAILABLE = False
    logger.warning("LyCORIS library not installed. Install with: pip install lycoris-lora")


def check_lycoris_available() -> bool:
    """Check if LyCORIS library is available."""
    return LYCORIS_AVAILABLE


def apply_pseudo_fp8(target_model: torch.nn.Module, compute_dtype: torch.dtype = None):
    """
    Apply pseudo FP8 compression to a model's Linear layers to save VRAM.

    This is optimized for LLM models (Qwen2.5-based) which primarily use Linear layers.
    Frozen weights are compressed to FP8 and accessed via closure in backward,
    avoiding saving them in the autograd graph.
    
    Note: Only Linear layers are converted to FP8. Normalization layers (RMSNorm, LayerNorm)
    remain in compute_dtype to avoid dtype promotion issues.
    
    Args:
        target_model: Model to apply FP8 to
        compute_dtype: Compute dtype for weight conversion (default: infer from model)
    """
    from torch.nn import functional as F
    
    # Determine compute dtype
    if compute_dtype is None:
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def _make_pseudo_fp8_forward(module: torch.nn.Module):
        """Create a forward closure that captures the module reference."""

        class _FrozenFP8(torch.autograd.Function):
            """
            Pseudo FP8 for frozen (non-trainable) layers.

            forward takes only ``inp`` so backward returns only ``(grad_input,)``.
            Weight is accessed via the captured closure – nothing is saved.
            """
            @staticmethod
            def forward(ctx, inp):
                weight = module.weight
                bias = getattr(module, "bias", None)
                ctx.save_for_backward()  # nothing saved – minimal VRAM

                # Convert FP8 weight to compute dtype first, then to input dtype
                w_comp = weight.to(compute_dtype).to(inp.dtype)
                b_comp = bias.to(compute_dtype).to(inp.dtype) if bias is not None else None

                res = F.linear(inp, w_comp, b_comp)

                # Explicitly delete to free memory earlier
                del w_comp, b_comp
                return res

            @staticmethod
            def backward(ctx, grad_output):
                grad_input = None
                if ctx.needs_input_grad[0]:
                    weight = module.weight
                    # Convert FP8 weight to compute dtype first
                    w_comp = weight.to(compute_dtype).to(grad_output.dtype)
                    grad_input = grad_output.matmul(w_comp)
                    del w_comp
                return (grad_input,)

        class _TrainableFP8(torch.autograd.Function):
            """
            Pseudo FP8 for trainable layers (e.g. LoKr-injected).

            Weight and bias are explicit forward arguments so PyTorch can
            properly propagate their gradients.
            """
            @staticmethod
            def forward(ctx, inp, weight, bias=None):
                ctx.save_for_backward(inp, weight, bias)
                ctx.bias_requires_grad = bias is not None and bias.requires_grad
                # Convert FP8 weight to compute dtype first, then to input dtype
                w_comp = weight.to(compute_dtype).to(inp.dtype)
                b_comp = bias.to(compute_dtype).to(inp.dtype) if bias is not None else None

                res = F.linear(inp, w_comp, b_comp)

                del w_comp, b_comp
                return res

            @staticmethod
            def backward(ctx, grad_output):
                inp, weight, bias = ctx.saved_tensors
                # Convert FP8 weight to compute dtype first
                w_comp = weight.to(compute_dtype).to(grad_output.dtype)
                grad_input = grad_weight = grad_bias = None

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

        def _forward(self_module, inp: torch.Tensor) -> torch.Tensor:
            weight = self_module.weight
            bias = getattr(self_module, "bias", None)

            if weight.requires_grad or (bias is not None and bias.requires_grad):
                return _TrainableFP8.apply(inp, weight, bias)
            else:
                return _FrozenFP8.apply(inp)

        return _forward

    # Apply pseudo FP8 to Linear layers only
    # Normalize non-Linear layers to compute_dtype to avoid dtype issues
    for module in target_model.modules():
        if isinstance(module, torch.nn.Linear):
            if hasattr(module.weight, "dtype") and module.weight.dtype != torch.float8_e4m3fn:
                with torch.no_grad():
                    module.weight.data = module.weight.data.to(torch.float8_e4m3fn)
                    if getattr(module, "bias", None) is not None:
                        module.bias.data = module.bias.data.to(torch.float8_e4m3fn)
                # Override forward with closure-based version
                module.forward = _make_pseudo_fp8_forward(module).__get__(module, module.__class__)
        elif isinstance(module, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
            # Ensure normalization layers stay in compute_dtype
            with torch.no_grad():
                module.weight.data = module.weight.data.to(compute_dtype)
                if getattr(module, "bias", None) is not None:
                    module.bias.data = module.bias.data.to(compute_dtype)
    
    logger.info(f"Pseudo FP8 applied to Linear layers for VRAM savings (compute_dtype={compute_dtype})")


def _matches_target_module_name(module_name: str, target_modules: List[str]) -> bool:
    """Check if a module name matches any of the target module suffixes."""
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


def inject_lokr_into_llm(
    llm_model: torch.nn.Module,
    lokr_config: Any,
    multiplier: float = 1.0,
    use_fp8: bool = False,
    compute_dtype: torch.dtype = None,
) -> Tuple[torch.nn.Module, Any, Dict[str, Any]]:
    """
    Inject LoKr adapters into an LLM model (Qwen3ForCausalLM).

    Args:
        llm_model: The LLM model (Qwen3ForCausalLM or similar)
        lokr_config: LoKRConfig with adapter settings
        multiplier: Adapter multiplier strength
        use_fp8: Whether to apply pseudo FP8 to frozen layers for VRAM savings
        compute_dtype: Compute dtype for FP8 conversion (default: bfloat16)

    Returns:
        Tuple: (llm_model, lycoris_network, info_dict)
    """
    if not LYCORIS_AVAILABLE:
        raise ImportError(
            "LyCORIS library is required for LoKr training. "
            "Install with: pip install lycoris-lora"
        )

    logger.info("Injecting LoKr adapters into LLM...")

    # Apply pseudo FP8 to frozen model before LoKr injection
    # This saves significant VRAM by compressing frozen weights to FP8
    if use_fp8:
        logger.info("Applying pseudo FP8 to frozen LLM layers...")
        apply_pseudo_fp8(llm_model, compute_dtype=compute_dtype)

    # Freeze all existing parameters (after FP8 conversion if enabled)
    for param in llm_model.parameters():
        param.requires_grad = False

    # Apply LyCORIS preset
    LycorisNetwork.apply_preset({
        "unet_target_name": lokr_config.target_modules,
        "target_name": lokr_config.target_modules,
    })

    # Create LyCORIS network
    try:
        lycoris_net = create_lycoris(
            llm_model,
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
    except Exception as e:
        logger.warning(f"Initial LoKr creation failed: {e}")
        raise

    # Apply to model
    lycoris_net.apply_to()

    # Move LoKr weights to the same device as the model
    device = next(llm_model.parameters()).device
    logger.info(f"Moving LoKr weights to {device}")
    
    # Move the entire LyCORIS network to the device
    lycoris_net = lycoris_net.to(device)
    
    # Also move any LoKr parameters that were directly added to modules
    for module in llm_model.modules():
        if hasattr(module, 'lokr_w1_a'):
            for attr_name in ['lokr_w1_a', 'lokr_w1_b', 'lokr_w2_a', 'lokr_w2_b', 'lokr_t2', 'lokr_w1', 'lokr_w2']:
                if hasattr(module, attr_name):
                    param = getattr(module, attr_name)
                    if param is not None and isinstance(param, torch.nn.Parameter):
                        setattr(module, attr_name, torch.nn.Parameter(param.data.to(device)))
    
    # Store reference on model
    llm_model._lycoris_net = lycoris_net

    # Enable only target modules
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
        logger.warning("No LoKr modules enabled! Enabling all parameters.")
        for param in lycoris_net.parameters():
            param.requires_grad = True
            lokr_param_list.append(param)

    # Deduplicate params
    unique_params = {id(p): p for p in lokr_param_list}
    total_params = sum(p.numel() for p in llm_model.parameters())
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
        "enabled_modules": enabled_module_count,
    }

    logger.info(f"LoKr injected into LLM")
    logger.info(
        f"LoKr trainable params: {trainable_params:,}/{total_params:,} "
        f"({info['trainable_ratio']:.2%})"
    )

    return llm_model, lycoris_net, info


def save_lokr_weights(
    lycoris_net: Any,
    output_dir: str,
    dtype: Optional[torch.dtype] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> str:
    """
    Save LoKr weights to safetensors file.

    Args:
        lycoris_net: LyCORIS network instance
        output_dir: Output directory
        dtype: Data type for saved weights
        metadata: Optional metadata dictionary

    Returns:
        Path to saved weights file
    """
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "lokr_weights.safetensors")

    save_metadata: Dict[str, str] = {"algo": "lokr", "format": "lycoris"}
    if metadata:
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, str):
                save_metadata[key] = value
            else:
                save_metadata[key] = json.dumps(value, ensure_ascii=True)

    lycoris_net.save_weights(weights_path, dtype=dtype, metadata=save_metadata)
    logger.info(f"LoKr weights saved to {weights_path}")

    return weights_path


def load_lokr_weights(
    lycoris_net: Any,
    weights_path: str,
) -> Dict[str, Any]:
    """
    Load LoKr weights into LyCORIS network.

    Args:
        lycoris_net: LyCORIS network instance
        weights_path: Path to weights file

    Returns:
        Load result dictionary
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"LoKr weights not found: {weights_path}")

    result = lycoris_net.load_weights(weights_path)
    logger.info(f"LoKr weights loaded from {weights_path}")

    return result


def save_lokr_training_checkpoint(
    lycoris_net: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    output_dir: str,
    lokr_config: Optional[Any] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save LoKr training checkpoint with optimizer/scheduler state.

    Args:
        lycoris_net: LyCORIS network
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        epoch: Current epoch
        global_step: Current training step
        output_dir: Output directory
        lokr_config: LoKR configuration
        run_metadata: Additional metadata

    Returns:
        Checkpoint directory path
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save weights
    metadata: Dict[str, Any] = {}
    if lokr_config is not None:
        metadata["lokr_config"] = {
            "linear_dim": lokr_config.linear_dim,
            "linear_alpha": lokr_config.linear_alpha,
            "factor": lokr_config.factor,
            "target_modules": lokr_config.target_modules,
        }
    if run_metadata is not None:
        metadata["run_metadata"] = run_metadata

    save_lokr_weights(lycoris_net, output_dir, metadata=metadata)

    # Save training state
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }
    if lokr_config is not None:
        state["lokr_config"] = metadata["lokr_config"]

    state_path = os.path.join(output_dir, "training_state.pt")
    torch.save(state, state_path)

    logger.info(f"Checkpoint saved to {output_dir} (epoch={epoch}, step={global_step})")

    return output_dir


def load_lokr_training_checkpoint(
    checkpoint_dir: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = None,
) -> Dict[str, Any]:
    """
    Load LoKr training checkpoint.

    Args:
        checkpoint_dir: Checkpoint directory
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load tensors to

    Returns:
        Dictionary with loaded state
    """
    result = {
        "epoch": 0,
        "global_step": 0,
        "adapter_path": None,
        "loaded_optimizer": False,
        "loaded_scheduler": False,
    }

    # Find adapter weights
    adapter_path = os.path.join(checkpoint_dir, "lokr_weights.safetensors")
    if os.path.exists(adapter_path):
        result["adapter_path"] = adapter_path
    else:
        result["adapter_path"] = checkpoint_dir

    # Load training state
    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if os.path.exists(state_path):
        try:
            training_state = torch.load(state_path, map_location=device, weights_only=True)

            if "epoch" in training_state:
                result["epoch"] = int(training_state["epoch"])
            if "global_step" in training_state:
                result["global_step"] = int(training_state["global_step"])

            if optimizer is not None and "optimizer_state_dict" in training_state:
                optimizer.load_state_dict(training_state["optimizer_state_dict"])
                result["loaded_optimizer"] = True

            if scheduler is not None and training_state.get("scheduler_state_dict"):
                scheduler.load_state_dict(training_state["scheduler_state_dict"])
                result["loaded_scheduler"] = True

            logger.info(
                f"Loaded checkpoint from epoch {result['epoch']}, step {result['global_step']}"
            )
        except Exception as e:
            logger.warning(f"Failed to load training state: {e}")

    return result
