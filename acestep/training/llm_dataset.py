"""
LLM Dataset for ACE-Step 1.5 LoKr Training

Dataset classes for LLM causal language modeling training.
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from loguru import logger

import torch
from torch.utils.data import Dataset, DataLoader


class LLMTrainingDataset(Dataset):
    """
    Dataset for LLM causal language modeling.

    Loads preprocessed tensor files containing:
    - input_ids: Tokenized prompt + response
    - labels: Same as input_ids (for causal LM)
    - attention_mask: Attention mask
    """

    def __init__(
        self,
        tensor_dir: str,
        max_length: int = 4096,
        pad_token_id: int = 151643,
    ):
        """
        Initialize LLM training dataset.

        Args:
            tensor_dir: Directory containing preprocessed .pt files
            max_length: Maximum sequence length
            pad_token_id: Padding token ID
        """
        self.tensor_dir = tensor_dir
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.sample_paths: List[str] = []

        # Load manifest
        manifest_path = os.path.join(tensor_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            raw_paths = manifest.get("samples", [])
            for raw in raw_paths:
                resolved = os.path.join(tensor_dir, raw) if not os.path.isabs(raw) else raw
                if os.path.exists(resolved):
                    self.sample_paths.append(resolved)
        else:
            # Fallback: scan for .pt files
            for f in os.listdir(tensor_dir):
                if f.endswith('.pt'):
                    self.sample_paths.append(os.path.join(tensor_dir, f))

        logger.info(f"LLM dataset loaded: {len(self.sample_paths)} samples from {tensor_dir}")

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a preprocessed tensor file."""
        tensor_path = self.sample_paths[idx]
        data = torch.load(tensor_path, map_location='cpu', weights_only=True)

        # Extract required fields
        input_ids = data.get("input_ids", data.get("tokens", torch.tensor([])))
        labels = data.get("labels", input_ids.clone())
        attention_mask = data.get("attention_mask", torch.ones_like(input_ids))

        # Truncate if needed
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "metadata": data.get("metadata", {}),
        }


class LLMTextDataset(Dataset):
    """
    Simple text-only dataset for LLM training.

    Loads text files and tokenizes on-the-fly.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 4096,
        template: str = "{text}",
    ):
        """
        Initialize LLM text dataset.

        Args:
            data_path: Path to text file or directory containing .txt files
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            template: Text template for formatting (default: "{text}")
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template
        self.texts: List[str] = []

        # Load text files
        if os.path.isfile(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                self.texts = [line.strip() for line in f if line.strip()]
        elif os.path.isdir(data_path):
            for fname in os.listdir(data_path):
                if fname.endswith('.txt'):
                    fpath = os.path.join(data_path, fname)
                    with open(fpath, 'r', encoding='utf-8') as f:
                        self.texts.extend([line.strip() for line in f if line.strip()])
        else:
            raise ValueError(f"data_path must be a file or directory: {data_path}")

        logger.info(f"LLM text dataset loaded: {len(self.texts)} samples from {data_path}")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Tokenize a text sample."""
        text = self.texts[idx]
        formatted = self.template.format(text=text)

        # Tokenize
        encoded = self.tokenizer.encode_plus(
            formatted,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "metadata": {"text": text},
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for LLM training.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=lambda x: x[0] if len(x) == 1 else torch.utils.data.default_collate(x),
    )
