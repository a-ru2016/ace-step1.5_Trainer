"""
Custom LoKr (Low-Rank Kronecker Product) Implementation for LLM

LoKr decomposes weight updates using Kronecker products:
ΔW = (A ⊗ B) where A and B are low-rank matrices

This implementation supports:
- Standard LoKr with Kronecker product decomposition
- Tucker decomposition mode
- Bypass mode for stable training
- Pseudo FP8 quantization for VRAM reduction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


def _quantize_to_fp8(x: torch.Tensor, fp8_scale: float = 448.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to pseudo FP8 format.
    
    FP8 E4M3 format:
    - 1 sign bit, 4 exponent bits, 3 mantissa bits
    - Range: [-448, 448]
    - Uses int8 storage for VRAM efficiency
    
    Args:
        x: Input tensor
        fp8_scale: Maximum representable value (448 for E4M3)
    
    Returns:
        Tuple of (quantized tensor as int8, scale factor)
    """
    # Clamp to FP8 range
    x_clamped = torch.clamp(x, -fp8_scale, fp8_scale)
    
    # Normalize to [-127, 127] range for int8
    x_normalized = x_clamped / fp8_scale * 127.0
    
    # Quantize to int8
    x_int8 = x_normalized.round().to(torch.int8)
    
    return x_int8


def _dequantize_from_fp8(x_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Dequantize int8 tensor back to FP8 range.
    
    Args:
        x_int8: Quantized int8 tensor
        scale: Scale factor for dequantization
    
    Returns:
        Dequantized tensor in original range
    """
    return x_int8.to(torch.float32) * scale


class LoKrLayer(nn.Module):
    """
    LoKr (Low-Rank Kronecker) layer for weight decomposition.

    For a weight matrix W of shape (out_features, in_features):
    - Decompose into two smaller matrices A and B
    - W ≈ A ⊗ B (Kronecker product)
    - Parameters reduced from O(mn) to O(m/r * n/r) where r is the factor

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        rank: Rank of the decomposition (default: 8)
        alpha: Scaling factor (default: 16)
        dropout: Dropout probability (default: 0.1)
        factor: Kronecker factor (-1 for automatic)
        decompose_both: Decompose both input and output dimensions
        bypass_mode: Use bypass connection for stability
        use_fp8: Use pseudo FP8 quantization for VRAM reduction
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
        self.fp8_scale = 448.0  # FP8 E4M3 max range

        # Scaling factor
        self.scaling = alpha / rank if rank > 0 else 1.0

        # Determine factor automatically if not specified
        if factor == -1:
            # Find a factor that divides both dimensions
            factor = self._find_optimal_factor(in_features, out_features, rank)
        self.factor = factor

        # Calculate decomposed dimensions
        # For Kronecker product: (A ⊗ B) where A:(m,n), B:(p,q) -> output:(m*p, n*q)
        # We want output shape (out_features, in_features)
        if decompose_both:
            # Decompose both input and output dimensions
            # A: (out_dim_a, in_dim_a), B: (out_dim_b, in_dim_b)
            # Result: (out_dim_a * out_dim_b, in_dim_a * in_dim_b) = (out_features, in_features)
            self.out_dim_a = out_features // factor
            self.out_dim_b = factor
            self.in_dim_a = in_features // factor
            self.in_dim_b = factor
        else:
            # Only decompose output dimension
            # A: (out_dim_a, in_features), B: (out_dim_b, 1)
            # Result: (out_dim_a * out_dim_b, in_features * 1) = (out_features, in_features)
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
        # A: (out_dim_a, in_dim_a) -> low rank decomposition
        # B: (out_dim_b, in_dim_b) -> low rank decomposition

        # Matrix A (larger matrix, low-rank decomposed)
        if use_fp8:
            # FP8 mode: store quantized values as int8 buffers, but keep float parameters for gradients
            # We use float parameters for training and quantize during forward pass
            self.lokr_A = nn.Parameter(torch.zeros(self.out_dim_a, self.in_dim_a))
            self.lokr_B = nn.Parameter(torch.zeros(self.out_dim_b, self.in_dim_b))
            # Per-tensor scale factors for FP8
            self.register_buffer('scale_A', torch.tensor(1.0))
            self.register_buffer('scale_B', torch.tensor(1.0))
            # Quantized storage (for memory efficiency when not training)
            self.register_buffer('lokr_A_int8', torch.zeros(self.out_dim_a, self.in_dim_a, dtype=torch.int8))
            self.register_buffer('lokr_B_int8', torch.zeros(self.out_dim_b, self.in_dim_b, dtype=torch.int8))
        else:
            self.lokr_A = nn.Parameter(torch.Tensor(self.out_dim_a, self.in_dim_a))
            self.lokr_B = nn.Parameter(torch.Tensor(self.out_dim_b, self.in_dim_b))
            self.register_buffer('scale_A', torch.tensor(1.0))
            self.register_buffer('scale_B', torch.tensor(1.0))

        # Initialize with small values (FP8 compatible)
        if not use_fp8:
            nn.init.kaiming_uniform_(self.lokr_A, a=math.sqrt(5))
        else:
            # FP8 mode: initialize with small values
            nn.init.kaiming_uniform_(self.lokr_A, a=math.sqrt(5))
            # Update quantized storage
            self.lokr_A_int8 = _quantize_to_fp8(self.lokr_A.data, self.fp8_scale)
            self.scale_A.fill_(1.0)
        
        # Initialize B to zeros so LoKr update is zero at start (preserves original behavior)
        nn.init.zeros_(self.lokr_B)
        if use_fp8:
            self.lokr_B_int8 = _quantize_to_fp8(self.lokr_B.data, self.fp8_scale)
            self.scale_B.fill_(1.0)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Learnable scalar (optional, for bypass mode)
        if bypass_mode:
            # Initialize scalar to 0 to preserve original behavior at start
            self.scalar = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('scalar', torch.tensor(1.0))
    
    def _find_optimal_factor(self, in_dim: int, out_dim: int, rank: int) -> int:
        """Find optimal Kronecker factor.
        
        For Qwen3 5Hz-LM:
        - q_proj: 2048→2048, head_dim=128
        - k_proj: 2048→1024, head_dim=128
        - v_proj: 2048→1024, head_dim=128
        - o_proj: 2048→2048, head_dim=128
        
        We want factor to preserve head_dim structure (128).
        """
        # For 2048→2048: factor=16 gives out_dim_a=128, out_dim_b=16
        # For 2048→1024: factor=8 gives out_dim_a=128, out_dim_b=8
        # We want out_dim_a to be a multiple of head_dim (128)
        
        head_dim = 128  # Qwen3 5Hz-LM head dimension
        
        # Try factors that result in out_dim_a being multiple of head_dim
        for factor in [16, 8, 4, 2, 1]:
            if out_dim % factor == 0 and in_dim % factor == 0:
                out_dim_a = out_dim // factor
                in_dim_a = in_dim // factor
                # Check if out_dim_a is reasonable (at least rank, preferably multiple of head_dim)
                if out_dim_a >= rank and in_dim_a >= rank:
                    return factor
        
        # Fallback: find any valid factor
        for factor in range(min(in_dim, out_dim), 0, -1):
            if out_dim % factor == 0 and in_dim % factor == 0:
                out_dim_a = out_dim // factor
                in_dim_a = in_dim // factor
                if out_dim_a >= rank and in_dim_a >= rank:
                    return factor
        
        return 1
    
    def _kronecker_product(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Compute Kronecker product of two matrices."""
        # A: (m, n), B: (p, q) -> output: (m*p, n*q)
        m, n = A.shape
        p, q = B.shape
        
        # Reshape and permute for efficient Kronecker product
        A_expanded = A.unsqueeze(1).unsqueeze(3)  # (m, 1, n, 1)
        B_expanded = B.unsqueeze(0).unsqueeze(2)  # (1, p, 1, q)
        
        kron = (A_expanded * B_expanded).reshape(m * p, n * q)
        return kron
    
    def forward(self, x: torch.Tensor, original_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with LoKr adaptation.

        Args:
            x: Input tensor of shape (batch, seq_len, in_features) or (batch*seq_len, in_features)
            original_weight: Original frozen weight matrix (for bypass mode)

        Returns:
            Output tensor of shape (batch, seq_len, out_features) or (batch*seq_len, out_features)
        """
        # F.linear handles both 2D and 3D inputs correctly
        # 3D: (batch, seq_len, in_features) -> (batch, seq_len, out_features)
        # 2D: (batch*seq_len, in_features) -> (batch*seq_len, out_features)

        # Compute LoKr weight update: ΔW = A ⊗ B
        if self.use_fp8:
            # FP8 mode: use quantized storage for inference, float params for training
            if self.training:
                # During training, use float parameters
                A = self.lokr_A
                B = self.lokr_B
            else:
                # During inference, use quantized storage
                A = _dequantize_from_fp8(self.lokr_A_int8, self.scale_A)
                B = _dequantize_from_fp8(self.lokr_B_int8, self.scale_B)
        else:
            A = self.lokr_A
            B = self.lokr_B
        
        delta_W = self._kronecker_product(A, B)

        # Scale the update
        delta_W = delta_W * self.scaling

        if self.bypass_mode and original_weight is not None:
            # Bypass mode: use original weight + scaled update
            # This preserves the original output distribution
            # Ensure scalar is on the same device as original_weight
            scalar = self.scalar.to(original_weight.device)
            W_effective = original_weight + scalar * delta_W
        else:
            # Standard mode: only use LoKr update
            # (Assumes original weight is zero or handled elsewhere)
            W_effective = delta_W

        # Apply linear transformation
        # F.linear handles both 2D and 3D inputs
        # 3D: (batch, seq_len, in_features) @ (out_features, in_features).T -> (batch, seq_len, out_features)
        # 2D: (batch*seq_len, in_features) @ (out_features, in_features).T -> (batch*seq_len, out_features)
        output = F.linear(x, W_effective)

        return self.dropout(output)
    
    def get_effective_weight(self, original_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get the effective weight matrix (original + LoKr update)."""
        if self.use_fp8:
            # FP8 mode: use quantized storage
            A = _dequantize_from_fp8(self.lokr_A_int8, self.scale_A)
            B = _dequantize_from_fp8(self.lokr_B_int8, self.scale_B)
        else:
            A = self.lokr_A
            B = self.lokr_B
        
        delta_W = self._kronecker_product(A, B) * self.scaling

        if self.bypass_mode and original_weight is not None:
            scalar = self.scalar.to(original_weight.device)
            return original_weight + scalar * delta_W
        else:
            return delta_W
    
    def get_param_count(self) -> int:
        """Get number of trainable parameters."""
        if self.use_fp8:
            # FP8 mode: parameters are stored as int8 (1 byte vs 4 bytes for float32)
            # Return equivalent parameter count for VRAM estimation
            return (self.lokr_A.numel() + self.lokr_B.numel()) // 4
        return self.lokr_A.numel() + self.lokr_B.numel()


class LoKrLinear(nn.Module):
    """
    LoKr-adapted linear layer that wraps an existing linear layer.

    This wrapper keeps the original weight frozen and applies LoKr adaptation.
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
        # Qwen3Attention always passes 3D input: (batch, seq_len, features)
        # No reshaping needed - just pass through LoKr layer
        
        # LoKr layer computes: (original_weight + delta) @ x
        output = self.lokr(x, self.original_weight)

        # Add bias if exists
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def get_effective_weight(self) -> torch.Tensor:
        """Get effective weight after LoKr adaptation."""
        return self.lokr.get_effective_weight(self.original_weight)
    
    def merge_weights(self) -> nn.Linear:
        """Merge LoKr weights into a single linear layer."""
        merged = nn.Linear(self.in_features, self.out_features, bias=self.has_bias)
        merged.weight.data = self.get_effective_weight()
        if self.has_bias:
            merged.bias.data = self.bias
        return merged


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
        module: The parent module containing linear layers
        target_module_names: List of module names to replace (e.g., ['q_proj', 'k_proj'])
        rank: LoKr rank
        alpha: LoKr alpha
        dropout: Dropout probability
        factor: Kronecker factor
        decompose_both: Decompose both dimensions
        bypass_mode: Use bypass mode
        use_fp8: Use pseudo FP8 quantization for VRAM reduction

    Returns:
        Dictionary mapping module names to their original linear layers
    """
    replaced = {}

    for name, submodule in module.named_modules():
        # Check if this module should be replaced
        short_name = name.split('.')[-1]
        if short_name in target_module_names and isinstance(submodule, nn.Linear):
            # Get parent module
            parts = name.split('.')
            parent = module
            for part in parts[:-1]:
                parent = getattr(parent, part)

            # Store original
            replaced[name] = submodule

            # Create LoKr wrapper
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

            # Replace in parent
            setattr(parent, parts[-1], lokr_linear)

    return replaced


def save_lokr_weights(module: nn.Module, save_path: str):
    """Save only LoKr weights to a file."""
    lokr_state = {}
    for name, submodule in module.named_modules():
        if isinstance(submodule, (LoKrLayer, LoKrLinear)):
            state = submodule.state_dict()
            # For FP8 mode, also save the quantized storage
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
            # Handle FP8 mode
            if isinstance(submodule, LoKrLinear) and submodule.use_fp8:
                # Load quantized storage first
                if 'lokr_A_int8' in state_dict:
                    submodule.lokr.lokr_A_int8.copy_(state_dict.pop('lokr_A_int8'))
                if 'lokr_B_int8' in state_dict:
                    submodule.lokr.lokr_B_int8.copy_(state_dict.pop('lokr_B_int8'))
                if 'scale_A' in state_dict:
                    submodule.lokr.scale_A.copy_(state_dict.pop('scale_A'))
                if 'scale_B' in state_dict:
                    submodule.lokr.scale_B.copy_(state_dict.pop('scale_B'))
            elif isinstance(submodule, LoKrLayer) and submodule.use_fp8:
                # Load quantized storage for standalone LoKrLayer
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
    lokr_params = 0
    total_params = 0

    for param in module.parameters():
        total_params += param.numel()
        if param.requires_grad:
            lokr_params += param.numel()

    return lokr_params, total_params


def get_fp8_storage_size(module: nn.Module) -> int:
    """Get the size of FP8 quantized storage in bytes."""
    fp8_bytes = 0
    for name, submodule in module.named_modules():
        if isinstance(submodule, LoKrLinear) and submodule.use_fp8:
            fp8_bytes += submodule.lokr.lokr_A_int8.numel()  # 1 byte per int8
            fp8_bytes += submodule.lokr.lokr_B_int8.numel()
    return fp8_bytes
