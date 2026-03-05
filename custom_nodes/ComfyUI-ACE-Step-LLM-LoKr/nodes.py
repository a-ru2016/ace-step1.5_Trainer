"""
ACE-Step 1.5 LLM LoKr Nodes for ComfyUI

Custom nodes for loading and saving LoKr adapters for the ACE-Step 1.5 LLM (5Hz-LM/Qwen3-based).

Compatible with LoKr weights trained by: acestep/training/train_llm_lokr.py

Note: This node uses CLIP input/output to connect with TextEncodeAceStepAudio1.5
"""

import os
import json
import torch
import folder_paths
import logging
from typing import Tuple, Dict, Any

from .llm_lokr import (
    LoKrLayer,
    LoKrLinear,
    load_lokr_weights,
    save_lokr_weights,
    replace_linear_with_lokr,
    count_lokr_parameters,
)

logger = logging.getLogger(__name__)


class AceStepLLMLoKrLoader:
    """
    Load LoKr adapter for ACE-Step 1.5 LLM (5Hz-LM/Qwen3-based).
    
    This node applies LoKr adapters trained with acestep/training/train_llm_lokr.py
    to the LLM model used in ACE-Step 1.5.
    
    Connect between Load CLIP and TextEncodeAceStepAudio1.5 nodes.
    """

    CATEGORY = "ACE-Step/LLM"
    FUNCTION = "load_lokr"
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "ACE-Step 1.5 CLIP (from Load CLIP node)"
                }),
                "lokr_path": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "LoKr weights file from loras directory"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Adapter strength multiplier"
                }),
            }
        }

    def load_lokr(self, clip, lokr_path: str, strength: float = 1.0):
        """
        Apply LoKr adapter to LLM model in CLIP.

        Args:
            clip: ACE-Step 1.5 CLIP (contains LLM model)
            lokr_path: Path to LoKr weights file from loras directory
            strength: Adapter strength multiplier (0.0-2.0)

        Returns:
            Tuple of (CLIP with LoKr adapter applied,)
        """
        if clip is None:
            raise ValueError("CLIP is required")

        if not lokr_path or not lokr_path.strip():
            logger.warning("LoKR path is empty, skipping adapter loading")
            return (clip,)

        # Resolve path from loras directory
        weights_path = folder_paths.get_full_path("loras", lokr_path)
        if not weights_path:
            raise FileNotFoundError(f"LoKr weights not found in loras directory: {lokr_path}")

        logger.info(f"Loading LoKr adapter from: {weights_path}")
        logger.info(f"Adapter strength: {strength}")

        # Get the underlying CLIP model and LLM
        clip_model = self._get_clip_model(clip)
        transformer_model = self._get_llm_from_clip(clip_model)

        # Load LoKr weights to get the module names
        try:
            lokr_data = self._load_weights(weights_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load LoKr weights: {e}")

        # Auto-detect and load config if available (same directory as weights)
        config_path = os.path.join(os.path.dirname(weights_path), "config.json")
        config = None
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Auto-detected config from: {config_path}")
            logger.debug(f"Config: {config}")

        # Step 1: Replace Linear layers with LoKr layers based on saved weights
        # Extract target module names from saved weights (e.g., "model.layers.0.self_attn.q_proj")
        target_modules = set()
        for module_name in lokr_data.keys():
            # Remove ".lokr" suffix if present
            clean_name = module_name.replace('.lokr', '')
            # Get the base module path (e.g., "model.layers.0.self_attn")
            parts = clean_name.split('.')
            if len(parts) >= 3:
                base_path = '.'.join(parts[:-1])  # e.g., "model.layers.0.self_attn"
                
                # Remove "model." prefix if present (ComfyUI model structure difference)
                if base_path.startswith('model.'):
                    base_path = base_path[6:]
                
                target_modules.add(base_path)
        
        target_modules = sorted(list(target_modules))
        logger.info(f"Target modules from weights: {target_modules[:4]}... ({len(target_modules)} total)")

        # Debug: Check what modules exist in the model
        logger.info("Checking model structure...")
        sample_modules = []
        for name, _ in transformer_model.named_modules():
            if 'layers.0.self_attn' in name:
                sample_modules.append(name)
        logger.info(f"Sample modules in model: {sample_modules[:8]}")

        # Step 2: Replace Linear layers with LoKr layers
        logger.info("Replacing Linear layers with LoKr layers...")
        replaced = replace_linear_with_lokr(
            transformer_model,
            target_module_names=target_modules,
            rank=8,  # Default, will be overwritten by loaded weights
            alpha=16,  # Default, will be overwritten by loaded weights
        )
        logger.info(f"Replaced {len(replaced)} layers with LoKr")
        if replaced:
            logger.info(f"Replaced module names: {list(replaced.keys())[:4]}...")
            
            # Record the device of each replaced layer
            device_map = {}
            for name, orig_linear in replaced.items():
                device_map[name] = orig_linear.weight.device
            logger.info(f"Recorded device map for {len(device_map)} layers")

        # Step 3: Load LoKr weights into the new LoKr layers
        loaded_count = self._apply_lokr_weights(transformer_model, lokr_data, strength, device_map if 'device_map' in locals() else None)

        logger.info(f"Loaded {loaded_count} LoKr modules")
        logger.info("LoKr adapter applied successfully")

        return (clip,)

    def _get_device_from_clip(self, clip) -> str:
        """Get the device where CLIP model is loaded."""
        if hasattr(clip, 'patcher'):
            if hasattr(clip.patcher, 'model'):
                return str(clip.patcher.model.device)
        if hasattr(clip, 'cond_stage_model'):
            if hasattr(clip.cond_stage_model, 'device'):
                return str(clip.cond_stage_model.device)
        # Default to cuda
        return 'cuda'

    def _get_clip_model(self, clip) -> torch.nn.Module:
        """
        Extract CLIP model from ComfyUI CLIP wrapper.
        
        For ACE-Step 1.5, the clip object is ACE15TEModel which contains:
        - qwen3_06b: base encoder
        - qwen3_2b or qwen3_4b: LLM model
        """
        # ComfyUI CLIP wrapper
        if hasattr(clip, 'cond_stage_model'):
            return clip.cond_stage_model
        elif hasattr(clip, 'model'):
            return clip.model
        # Direct ACE15TEModel
        return clip

    def _get_llm_from_clip(self, clip_model) -> torch.nn.Module:
        """
        Extract LLM model from ACE-Step 1.5 CLIP.
        
        ACE15TEModel structure:
        - ACE15TEModel (Qwen3_4B_ACE15)
          └── qwen3_4b: Qwen3_4B_ACE15_lm (BaseLlama)
               └── transformer: Qwen3_4B_ACE15_lm
                    └── model: Llama2_
                         └── layers[0].self_attn.q_proj
        
        We need to return Llama2_ (the actual transformer with layers).
        """
        llm_model = None
        
        # Step 1: Get the LLM model from ACE15TEModel
        if hasattr(clip_model, 'qwen3_2b'):
            logger.info("Found qwen3_2b attribute")
            llm_model = clip_model.qwen3_2b
        elif hasattr(clip_model, 'qwen3_4b'):
            logger.info("Found qwen3_4b attribute")
            llm_model = clip_model.qwen3_4b
        elif hasattr(clip_model, 'lm_model'):
            lm_name = clip_model.lm_model
            if lm_name and hasattr(clip_model, lm_name):
                logger.info(f"Found LLM model by name: {lm_name}")
                llm_model = getattr(clip_model, lm_name)
        
        if llm_model is None:
            logger.warning("No LLM model found in clip_model")
            return clip_model
        
        logger.info(f"LLM model type: {type(llm_model).__name__}")
        
        # Step 2: Navigate through the wrapper layers to find Llama2_
        # Try .transformer first (BaseLlama)
        current_model = llm_model
        depth = 0
        max_depth = 5
        
        while depth < max_depth:
            # Check if current model has .layers (Llama2_)
            if hasattr(current_model, 'layers'):
                logger.info(f"Found .layers attribute at depth {depth}, type: {type(current_model).__name__}")
                logger.info(f"Number of layers: {len(current_model.layers)}")
                return current_model
            
            # Try to go deeper
            next_model = None
            if hasattr(current_model, 'transformer'):
                next_model = current_model.transformer
                logger.info(f"Found .transformer at depth {depth}, type: {type(next_model).__name__}")
            elif hasattr(current_model, 'model'):
                next_model = current_model.model
                logger.info(f"Found .model at depth {depth}, type: {type(next_model).__name__}")
            
            if next_model is None or next_model is current_model:
                break
            
            current_model = next_model
            depth += 1
        
        # Fallback: return what we found
        logger.warning(f"Using model at depth {depth}: {type(current_model).__name__}")
        return current_model

    def _load_weights(self, weights_path: str) -> Dict:
        """Load LoKr weights from file."""
        if weights_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            return load_file(weights_path)
        else:
            return torch.load(weights_path, map_location='cpu', weights_only=True)

    def _apply_lokr_weights(self, torch_model: torch.nn.Module, lokr_data: Dict, strength: float, device_map: Dict = None) -> int:
        """Apply LoKr weights to model."""
        loaded_count = 0
        
        # Debug: print model structure
        logger.info(f"Target model type: {type(torch_model).__name__}")
        logger.info(f"Target model has 'layers' attr: {hasattr(torch_model, 'layers')}")
        if hasattr(torch_model, 'layers'):
            logger.info(f"Number of layers: {len(torch_model.layers)}")
        
        for module_name, state_dict in lokr_data.items():
            try:
                # Remove ".lokr" suffix for matching
                clean_name = module_name.replace('.lokr', '')
                
                # Remove "model." prefix if present (ComfyUI model structure difference)
                if clean_name.startswith('model.'):
                    clean_name = clean_name[6:]
                    logger.debug(f"Removed 'model.' prefix: {clean_name}")
                
                submodule = torch_model.get_submodule(clean_name)
                
                if isinstance(submodule, (LoKrLayer, LoKrLinear)) and hasattr(submodule, 'load_state_dict'):
                    # Determine the target device
                    # Use device_map if available, otherwise use model's device
                    target_device = None
                    if device_map and module_name in device_map:
                        target_device = device_map[module_name]
                    else:
                        # Fallback: use the device of the submodule's parameters
                        try:
                            target_device = next(submodule.parameters()).device
                        except StopIteration:
                            target_device = 'cpu'
                    
                    logger.debug(f"Loading weights for {module_name} on device: {target_device}")
                    
                    # Move state dict tensors to the correct device
                    moved_state_dict = {}
                    for k, v in state_dict.items():
                        if hasattr(v, 'to'):
                            moved_state_dict[k] = v.to(target_device)
                        else:
                            moved_state_dict[k] = v
                    
                    # Adjust scalar if strength != 1.0
                    if strength != 1.0 and 'lokr.scalar' in moved_state_dict:
                        moved_state_dict['lokr.scalar'] = moved_state_dict['lokr.scalar'] * strength
                        logger.debug(f"Adjusted scalar for {module_name} by strength {strength}")
                    
                    submodule.load_state_dict(moved_state_dict, strict=False)
                    loaded_count += 1
                    logger.debug(f"Loaded weights for {module_name}")
            except Exception as e:
                logger.warning(f"Failed to load weights for {module_name}: {e}")
        
        return loaded_count


class AceStepLLMLoKrSave:
    """
    Save LoKr adapter from ACE-Step 1.5 LLM.
    """

    CATEGORY = "ACE-Step/LLM"
    FUNCTION = "save_lokr"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "CLIP with LoKr adapter"
                }),
                "output_path": ("STRING", {
                    "default": "llm_lokr/lokr_weights.pt",
                    "multiline": False,
                    "dynamicPrompts": False,
                    "tooltip": "Output path for LoKr weights"
                }),
            },
            "optional": {
                "save_config": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Save config.json alongside weights"
                }),
            }
        }

    def save_lokr(self, clip, output_path: str, save_config: bool = True):
        """
        Save LoKr adapter from LLM model in CLIP.

        Args:
            clip: CLIP containing LLM model with LoKr adapter
            output_path: Output path for weights file
            save_config: Whether to save config.json

        Returns:
            Empty tuple (output node)
        """
        if clip is None:
            raise ValueError("CLIP is required")

        if not output_path:
            raise ValueError("Output path is required")

        # Ensure .pt extension (custom LoKr format)
        if not output_path.endswith('.pt') and not output_path.endswith('.safetensors'):
            output_path += ".pt"

        # Make path absolute relative to ComfyUI output directory
        if not os.path.isabs(output_path):
            output_dir = folder_paths.get_output_directory()
            output_path = os.path.join(output_dir, output_path)

        logger.info(f"Saving LoKr adapter to: {output_path}")

        # Get the CLIP model and LLM
        clip_model = self._get_clip_model(clip)
        llm_model = self._get_llm_from_clip(clip_model)

        # Extract and save LoKr weights
        try:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            
            if output_path.endswith('.safetensors'):
                self._save_safetensors(llm_model, output_path)
            else:
                save_lokr_weights(llm_model, output_path)
            
            logger.info(f"LoKr adapter saved successfully to {output_path}")
            
            # Save config if requested
            if save_config:
                config_path = os.path.join(os.path.dirname(output_path), "config.json")
                self._save_config(llm_model, config_path)
                
        except Exception as e:
            raise RuntimeError(f"Failed to save LoKr weights: {e}")

        return ()

    def _get_clip_model(self, clip) -> torch.nn.Module:
        """
        Extract CLIP model from ComfyUI CLIP wrapper.
        
        For ACE-Step 1.5, the clip object is ACE15TEModel which contains:
        - qwen3_06b: base encoder
        - qwen3_2b or qwen3_4b: LLM model
        """
        # ComfyUI CLIP wrapper
        if hasattr(clip, 'cond_stage_model'):
            return clip.cond_stage_model
        elif hasattr(clip, 'model'):
            return clip.model
        # Direct ACE15TEModel
        return clip

    def _get_llm_from_clip(self, clip_model) -> torch.nn.Module:
        """
        Extract LLM model from ACE-Step 1.5 CLIP.
        
        ACE15TEModel structure:
        - ACE15TEModel (Qwen3_4B_ACE15)
          └── qwen3_4b: Qwen3_4B_ACE15_lm (BaseLlama)
               └── transformer: Qwen3_4B_ACE15_lm
                    └── model: Llama2_
                         └── layers[0].self_attn.q_proj
        
        We need to return Llama2_ (the actual transformer with layers).
        """
        llm_model = None
        
        # Step 1: Get the LLM model from ACE15TEModel
        if hasattr(clip_model, 'qwen3_2b'):
            logger.info("Found qwen3_2b attribute")
            llm_model = clip_model.qwen3_2b
        elif hasattr(clip_model, 'qwen3_4b'):
            logger.info("Found qwen3_4b attribute")
            llm_model = clip_model.qwen3_4b
        elif hasattr(clip_model, 'lm_model'):
            lm_name = clip_model.lm_model
            if lm_name and hasattr(clip_model, lm_name):
                logger.info(f"Found LLM model by name: {lm_name}")
                llm_model = getattr(clip_model, lm_name)
        
        if llm_model is None:
            logger.warning("No LLM model found in clip_model")
            return clip_model
        
        logger.info(f"LLM model type: {type(llm_model).__name__}")
        
        # Step 2: Navigate through the wrapper layers to find Llama2_
        # Try .transformer first (BaseLlama)
        current_model = llm_model
        depth = 0
        max_depth = 5
        
        while depth < max_depth:
            # Check if current model has .layers (Llama2_)
            if hasattr(current_model, 'layers'):
                logger.info(f"Found .layers attribute at depth {depth}, type: {type(current_model).__name__}")
                logger.info(f"Number of layers: {len(current_model.layers)}")
                return current_model
            
            # Try to go deeper
            next_model = None
            if hasattr(current_model, 'transformer'):
                next_model = current_model.transformer
                logger.info(f"Found .transformer at depth {depth}, type: {type(next_model).__name__}")
            elif hasattr(current_model, 'model'):
                next_model = current_model.model
                logger.info(f"Found .model at depth {depth}, type: {type(next_model).__name__}")
            
            if next_model is None or next_model is current_model:
                break
            
            current_model = next_model
            depth += 1
        
        # Fallback: return what we found
        logger.warning(f"Using model at depth {depth}: {type(current_model).__name__}")
        return current_model

    def _save_safetensors(self, torch_model: torch.nn.Module, output_path: str):
        """Save LoKr weights in safetensors format."""
        from safetensors.torch import save_file
        
        lokr_weights = {}
        for name, module in torch_model.named_modules():
            if isinstance(module, (LoKrLayer, LoKrLinear)):
                if hasattr(module, 'state_dict'):
                    state_dict = module.state_dict()
                    for param_name, tensor in state_dict.items():
                        lokr_weights[f"{name}.{param_name}"] = tensor
        
        save_file(lokr_weights, output_path)

    def _save_config(self, torch_model: torch.nn.Module, config_path: str):
        """Save LoKr configuration."""
        lokr_params, total_params = count_lokr_parameters(torch_model)
        
        config = {
            "lokr_params": lokr_params,
            "total_params": total_params,
            "trainable_ratio": lokr_params / total_params if total_params > 0 else 0.0,
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Config saved to {config_path}")


class AceStepLLMLoKrApply:
    """
    Apply LoKr adapter to ACE-Step 1.5 LLM with advanced options.

    This node provides more control over LoKr application, including
    the ability to specify target modules and LoKr parameters.
    
    Connect between Load CLIP and TextEncodeAceStepAudio1.5 nodes.
    """

    CATEGORY = "ACE-Step/LLM"
    FUNCTION = "apply_lokr"
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "ACE-Step 1.5 CLIP"
                }),
                "lokr_path": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "LoKr weights file from loras directory"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                }),
            },
            "optional": {
                "target_modules": ("STRING", {
                    "default": "q_proj,k_proj,v_proj,o_proj",
                    "multiline": True,
                    "placeholder": "Comma-separated module names (e.g., q_proj,k_proj,v_proj,o_proj)"
                }),
                "rank": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 256,
                    "tooltip": "LoKr rank (only needed for new adapters)"
                }),
                "alpha": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 256,
                    "tooltip": "LoKr alpha (only needed for new adapters)"
                }),
            }
        }

    def apply_lokr(
        self,
        clip,
        lokr_path: str,
        strength: float = 1.0,
        target_modules: str = None,
        rank: int = 8,
        alpha: int = 16,
    ):
        """Apply LoKr adapter with advanced options."""
        if clip is None:
            raise ValueError("CLIP is required")

        clip_model = self._get_clip_model(clip)
        llm_model = self._get_llm_from_clip(clip_model)

        # Load existing weights if path provided
        if lokr_path:
            weights_path = folder_paths.get_full_path("loras", lokr_path)
            if weights_path and os.path.exists(weights_path):
                loader = AceStepLLMLoKrLoader()
                return loader.load_lokr(clip, lokr_path, strength)

        # If no path or file not found, check if model already has LoKr layers
        has_lokr = any(
            isinstance(m, (LoKrLayer, LoKrLinear))
            for m in llm_model.modules()
        )

        if not has_lokr and target_modules:
            # Apply new LoKr layers
            modules = [m.strip() for m in target_modules.split(',')]
            replace_linear_with_lokr(
                llm_model,
                target_module_names=modules,
                rank=rank,
                alpha=alpha,
            )
            logger.info(f"Applied new LoKr layers to: {modules}")

        return (clip,)

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AceStepLLMLoKrLoader": AceStepLLMLoKrLoader,
    "AceStepLLMLoKrSave": AceStepLLMLoKrSave,
    "AceStepLLMLoKrApply": AceStepLLMLoKrApply,
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepLLMLoKrLoader": "ACE-Step LLM LoKr Loader",
    "AceStepLLMLoKrSave": "ACE-Step LLM LoKr Save",
    "AceStepLLMLoKrApply": "ACE-Step LLM LoKr Apply (Advanced)",
}

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
