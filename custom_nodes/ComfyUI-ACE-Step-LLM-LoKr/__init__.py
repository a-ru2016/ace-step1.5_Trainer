"""
ComfyUI ACE-Step LLM LoKr Node

Custom node for loading and saving LoKr adapters for ACE-Step 1.5 LLM.

Compatible with LoKr weights trained by: acestep/training/train_llm_lokr.py
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__version__ = "1.0.0"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# WEB_DIRECTORY is optional - uncomment if you add JavaScript for the frontend
# WEB_DIRECTORY = "./web"
