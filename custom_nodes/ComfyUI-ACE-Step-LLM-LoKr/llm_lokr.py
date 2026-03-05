"""
Custom LoKr (Low-Rank Kronecker Product) Implementation for LLM
ComfyUI ACE-Step 1.5 LLM LoKr Node

LoKr decomposes weight updates using Kronecker products:
ΔW = (A ⊗ B) where A and B are low-rank matrices

This implementation supports:
- Standard LoKr with Kronecker product decomposition
- Bypass mode for stable training
- Pseudo FP8 quantization for VRAM reduction

Compatible with acestep/training/train_llm_lokr.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


def _quantize_to_fp8(x: torch.Tensor, fp8_scale: float = 448.0) -> torch.Tensor:
    """Quantize tensor to pseudo FP8 format (int8 storage)."""
    x_clamped = torch.clamp(x, -fp8_scale, fp8_scale)
    x_normalized = x_clamped / fp8_scale * 127.0
    return x_normalized.round().to(torch.int8)


def _dequantize_from_fp8(x_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize int8 tensor back to FP8 range."""
    return x_int8.to(torch.float32) * scale


class LoKrLayer(nn.Module):
    """
    LoKr (Low-Rank Kronecker) layer for weight decomposition.
    
    For a weight matrix W of shape (out_features, in_features):
    - Decompose into two smaller matrices A and B
    - W ≈ A ⊗ B (Kronecker product)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
        factor: int = -1,
        decompose_both: bool = False,
        bypass_mode: bool = True,
        use_fp8: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.factor = factor
        self.decompose_both = decompose_both
        self.bypass_mode = bypass_mode
        self.use_fp8 = use_fp8
        self.fp8_scale = 448.0

        # Determine factor automatically if not specified
        if factor == -1:
            factor = self._find_optimal_factor(in_features, out_features, rank)
        self.factor = factor

        # Calculate decomposed dimensions
        if decompose_both:
            self.out_dim_a = out_features // factor
            self.out_dim_b = factor
            self.in_dim_a = in_features // factor
            self.in_dim_b = factor
        else:
            self.out_dim_a = out_features // factor
            self.out_dim_b = factor
            self.in_dim_a = in_features
            self.in_dim_b = 1

        # Validate dimensions
        assert self.out_dim_a * self.out_dim_b == out_features, \
            f"Output dimension mismatch: {self.out_dim_a} * {self.out_dim_b} != {out_features}"
        assert self.in_dim_a * self.in_dim_b == in_features, \
            f"Input dimension mismatch: {self.in_dim_a} * {self.in_dim_b} != {in_features}"

        # LoKr matrices A and B
        if use_fp8:
            self.lokr_A = nn.Parameter(torch.zeros(self.out_dim_a, self.in_dim_a))
            self.lokr_B = nn.Parameter(torch.zeros(self.out_dim_b, self.in_dim_b))
            self.register_buffer('scale_A', torch.tensor(1.0))
            self.register_buffer('scale_B', torch.tensor(1.0))
            self.register_buffer('lokr_A_int8', torch.zeros(self.out_dim_a, self.in_dim_a, dtype=torch.int8))
            self.register_buffer('lokr_B_int8', torch.zeros(self.out_dim_b, self.in_dim_b, dtype=torch.int8))
        else:
            self.lokr_A = nn.Parameter(torch.Tensor(self.out_dim_a, self.in_dim_a))
            self.lokr_B = nn.Parameter(torch.Tensor(self.out_dim_b, self.in_dim_b))
            self.register_buffer('scale_A', torch.tensor(1.0))
            self.register_buffer('scale_B', torch.tensor(1.0))

        # Initialize
        if not use_fp8:
            nn.init.kaiming_uniform_(self.lokr_A, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.lokr_A, a=math.sqrt(5))
            self.lokr_A_int8 = _quantize_to_fp8(self.lokr_A.data, self.fp8_scale)
            self.scale_A.fill_(1.0)

        nn.init.zeros_(self.lokr_B)
        if use_fp8:
            self.lokr_B_int8 = _quantize_to_fp8(self.lokr_B.data, self.fp8_scale)
            self.scale_B.fill_(1.0)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Learnable scalar for bypass mode
        if bypass_mode:
            self.scalar = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('scalar', torch.tensor(1.0))
        
        # Scaling factor (alpha/rank) - store as buffer for device movement
        self.register_buffer('scaling', torch.tensor(alpha / rank if rank > 0 else 1.0))

    def _find_optimal_factor(self, in_dim: int, out_dim: int, rank: int) -> int:
        """Find optimal Kronecker factor."""
        head_dim = 128  # Qwen3 5Hz-LM head dimension
        
        for factor in [16, 8, 4, 2, 1]:
            if out_dim % factor == 0 and in_dim % factor == 0:
                out_dim_a = out_dim // factor
                in_dim_a = in_dim // factor
                if out_dim_a >= rank and in_dim_a >= rank:
                    return factor
        
        for factor in range(min(in_dim, out_dim), 0, -1):
            if out_dim % factor == 0 and in_dim % factor == 0:
                out_dim_a = out_dim // factor
                in_dim_a = in_dim // factor
                if out_dim_a >= rank and in_dim_a >= rank:
                    return factor
        
        return 1

    def _kronecker_product(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Compute Kronecker product of two matrices."""
        m, n = A.shape
        p, q = B.shape
        
        A_expanded = A.unsqueeze(1).unsqueeze(3)  # (m, 1, n, 1)
        B_expanded = B.unsqueeze(0).unsqueeze(2)  # (1, p, 1, q)
        
        kron = (A_expanded * B_expanded).reshape(m * p, n * q)
        return kron

    def forward(self, x: torch.Tensor, original_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with LoKr adaptation."""
        if self.use_fp8:
            if self.training:
                A = self.lokr_A
                B = self.lokr_B
            else:
                A = _dequantize_from_fp8(self.lokr_A_int8, self.scale_A)
                B = _dequantize_from_fp8(self.lokr_B_int8, self.scale_B)
        else:
            A = self.lokr_A
            B = self.lokr_B

        # Ensure all tensors are on the same device and dtype as original_weight
        device = x.device if original_weight is None else original_weight.device
        dtype = original_weight.dtype if original_weight is not None else x.dtype
        
        A = A.to(device=device, dtype=dtype)
        B = B.to(device=device, dtype=dtype)

        delta_W = self._kronecker_product(A, B)
        delta_W = delta_W * self.scaling.to(device=device, dtype=dtype)

        if self.bypass_mode and original_weight is not None:
            scalar = self.scalar.to(device=device, dtype=dtype)
            W_effective = original_weight + scalar * delta_W
        else:
            W_effective = delta_W

        output = F.linear(x, W_effective)
        return self.dropout(output)

    def get_effective_weight(self, original_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get the effective weight matrix (original + LoKr update)."""
        if self.use_fp8:
            A = _dequantize_from_fp8(self.lokr_A_int8, self.scale_A)
            B = _dequantize_from_fp8(self.lokr_B_int8, self.scale_B)
        else:
            A = self.lokr_A
            B = self.lokr_B

        # Ensure all tensors are on the same device and dtype
        device = original_weight.device if original_weight is not None else A.device
        dtype = original_weight.dtype if original_weight is not None else A.dtype
        
        A = A.to(device=device, dtype=dtype)
        B = B.to(device=device, dtype=dtype)

        delta_W = self._kronecker_product(A, B) * self.scaling.to(device=device, dtype=dtype)

        if self.bypass_mode and original_weight is not None:
            scalar = self.scalar.to(device=device, dtype=dtype)
            return original_weight + scalar * delta_W
        else:
            return delta_W


class LoKrLinear(nn.Module):
    """
    LoKr-adapted linear layer that wraps an existing linear layer.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
        factor: int = -1,
        decompose_both: bool = False,
        bypass_mode: bool = True,
        use_fp8: bool = False,
    ):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.has_bias = original_linear.bias is not None
        self.bypass_mode = bypass_mode
        self.use_fp8 = use_fp8

        # Store original weight (frozen)
        self.register_buffer('original_weight', original_linear.weight.data.clone())
        if self.has_bias:
            self.register_buffer('bias', original_linear.bias.data.clone())
        else:
            self.register_buffer('bias', None)

        # Create LoKr layer
        self.lokr = LoKrLayer(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            factor=factor,
            decompose_both=decompose_both,
            bypass_mode=bypass_mode,
            use_fp8=use_fp8,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoKr adaptation."""
        # Ensure original_weight is on the same device as input
        if self.original_weight.device != x.device:
            self.original_weight = self.original_weight.to(x.device)
            if self.bias is not None:
                self.bias = self.bias.to(x.device)
        
        output = self.lokr(x, self.original_weight)
        if self.bias is not None:
            output = output + self.bias
        return output

    def get_effective_weight(self) -> torch.Tensor:
        """Get effective weight after LoKr adaptation."""
        return self.lokr.get_effective_weight(self.original_weight)


def replace_linear_with_lokr(
    module: nn.Module,
    target_module_names: list,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    factor: int = -1,
    decompose_both: bool = False,
    bypass_mode: bool = True,
    use_fp8: bool = False,
) -> dict:
    """
    Replace specified linear layers with LoKr-adapted versions.
    
    Args:
        module: The model to modify
        target_module_names: List of module paths. Can be:
                            - Full paths to Linear layers (e.g., "layers.0.self_attn.q_proj")
                            - Parent module paths (e.g., "layers.0.self_attn") - will replace all Linear children
                            - Short names (e.g., "q_proj") - will match all modules ending with this name
    """
    replaced = {}

    for name, submodule in module.named_modules():
        # Check if full path matches exactly
        full_path_match = name in target_module_names
        
        # Check if short name matches (for backward compatibility)
        short_name = name.split('.')[-1]
        short_name_match = short_name in target_module_names
        
        # Check if this module is a child of any target parent module
        parent_match = False
        for target in target_module_names:
            if name.startswith(target + '.'):
                parent_match = True
                break
        
        if (full_path_match or short_name_match or parent_match) and isinstance(submodule, nn.Linear):
            parts = name.split('.')
            parent = module
            for part in parts[:-1]:
                parent = getattr(parent, part)

            replaced[name] = submodule

            lokr_linear = LoKrLinear(
                submodule,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                factor=factor,
                decompose_both=decompose_both,
                bypass_mode=bypass_mode,
                use_fp8=use_fp8,
            )

            setattr(parent, parts[-1], lokr_linear)

    return replaced


def save_lokr_weights(module: nn.Module, save_path: str):
    """Save only LoKr weights to a file."""
    lokr_state = {}
    for name, submodule in module.named_modules():
        if isinstance(submodule, (LoKrLayer, LoKrLinear)):
            state = submodule.state_dict()
            if isinstance(submodule, LoKrLinear) and submodule.use_fp8:
                state['lokr_A_int8'] = submodule.lokr.lokr_A_int8.clone()
                state['lokr_B_int8'] = submodule.lokr.lokr_B_int8.clone()
                state['scale_A'] = submodule.lokr.scale_A.clone()
                state['scale_B'] = submodule.lokr.scale_B.clone()
            elif isinstance(submodule, LoKrLayer) and submodule.use_fp8:
                state['lokr_A_int8'] = submodule.lokr_A_int8.clone()
                state['lokr_B_int8'] = submodule.lokr_B_int8.clone()
                state['scale_A'] = submodule.scale_A.clone()
                state['scale_B'] = submodule.scale_B.clone()
            lokr_state[name] = state

    torch.save(lokr_state, save_path)
    return lokr_state


def load_lokr_weights(module: nn.Module, load_path: str):
    """Load LoKr weights from a file."""
    lokr_state = torch.load(load_path, map_location='cpu', weights_only=True)

    for name, state_dict in lokr_state.items():
        submodule = module.get_submodule(name)
        if isinstance(submodule, (LoKrLayer, LoKrLinear)):
            if isinstance(submodule, LoKrLinear) and submodule.use_fp8:
                if 'lokr_A_int8' in state_dict:
                    submodule.lokr.lokr_A_int8.copy_(state_dict.pop('lokr_A_int8'))
                if 'lokr_B_int8' in state_dict:
                    submodule.lokr.lokr_B_int8.copy_(state_dict.pop('lokr_B_int8'))
                if 'scale_A' in state_dict:
                    submodule.lokr.scale_A.copy_(state_dict.pop('scale_A'))
                if 'scale_B' in state_dict:
                    submodule.lokr.scale_B.copy_(state_dict.pop('scale_B'))
            elif isinstance(submodule, LoKrLayer) and submodule.use_fp8:
                if 'lokr_A_int8' in state_dict:
                    submodule.lokr_A_int8.copy_(state_dict.pop('lokr_A_int8'))
                if 'lokr_B_int8' in state_dict:
                    submodule.lokr_B_int8.copy_(state_dict.pop('lokr_B_int8'))
                if 'scale_A' in state_dict:
                    submodule.scale_A.copy_(state_dict.pop('scale_A'))
                if 'scale_B' in state_dict:
                    submodule.scale_B.copy_(state_dict.pop('scale_B'))
            submodule.load_state_dict(state_dict, strict=False)

    return lokr_state


def count_lokr_parameters(module: nn.Module) -> Tuple[int, int]:
    """Count LoKr parameters vs total parameters."""
    lokr_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in module.parameters())
    return lokr_params, total_params
