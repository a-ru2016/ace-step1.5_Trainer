"""
Preprocessing Script for LLM LoKr Training

Prepares text datasets for LLM causal language modeling training.
Tokenizes text data and saves as tensor files for efficient training.
"""

import os
import json
import argparse
from typing import Optional, Dict, Any, List
from loguru import logger

import torch
from tqdm import tqdm


def load_tokenizer(model_path: str):
    """
    Load tokenizer from a model path.

    Args:
        model_path: Path to the model or tokenizer directory

    Returns:
        Tokenizer instance
    """
    from transformers import AutoTokenizer

    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}, pad_token_id={tokenizer.pad_token_id}")
    return tokenizer


def preprocess_text_file(
    input_path: str,
    output_dir: str,
    tokenizer,
    max_length: int = 4096,
    template: str = "{text}",
) -> Dict[str, Any]:
    """
    Preprocess a text file into tensor format.

    Args:
        input_path: Path to input text file
        output_dir: Output directory for tensor files
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        template: Text template for formatting

    Returns:
        Metadata dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read text file
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Process each line as a sample
    sample_paths = []
    total_tokens = 0

    for idx, line in enumerate(tqdm(lines, desc="Preprocessing")):
        text = line.strip()
        if not text:
            continue

        # Apply template
        formatted = template.format(text=text)

        # Tokenize
        encoded = tokenizer.encode_plus(
            formatted,
            max_length=max_length,
            padding='max_length' if len(formatted) < max_length else False,
            truncation=True,
            return_tensors='pt',
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        labels = input_ids.clone()

        # Set padding tokens in labels to -100 (ignored in loss)
        if tokenizer.pad_token_id is not None:
            labels[input_ids == tokenizer.pad_token_id] = -100

        # Save tensor
        sample_filename = f"sample_{idx:06d}.pt"
        sample_path = os.path.join(output_dir, sample_filename)

        tensor_data = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "metadata": {
                "text": text,
                "formatted_length": len(formatted),
                "num_tokens": (input_ids != tokenizer.pad_token_id).sum().item(),
            },
        }

        torch.save(tensor_data, sample_path)
        sample_paths.append(sample_filename)
        total_tokens += tensor_data["metadata"]["num_tokens"]

    # Save manifest
    manifest = {
        "source_file": os.path.basename(input_path),
        "num_samples": len(sample_paths),
        "total_tokens": total_tokens,
        "max_length": max_length,
        "template": template,
        "pad_token_id": tokenizer.pad_token_id,
        "samples": sample_paths,
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(f"Preprocessed {len(sample_paths)} samples to {output_dir}")
    logger.info(f"Total tokens: {total_tokens:,}")

    return manifest


def preprocess_text_directory(
    input_dir: str,
    output_dir: str,
    tokenizer,
    max_length: int = 4096,
    template: str = "{text}",
) -> Dict[str, Any]:
    """
    Preprocess all text files in a directory.

    Args:
        input_dir: Input directory containing .txt files
        output_dir: Output directory for tensor files
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        template: Text template for formatting

    Returns:
        Metadata dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    all_sample_paths = []
    total_tokens = 0
    file_count = 0

    # Find all text files
    text_files = []
    for fname in os.listdir(input_dir):
        if fname.endswith('.txt'):
            text_files.append(os.path.join(input_dir, fname))

    logger.info(f"Found {len(text_files)} text files in {input_dir}")

    for fpath in text_files:
        logger.info(f"Processing {os.path.basename(fpath)}...")

        with open(fpath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for idx, line in enumerate(tqdm(lines, desc=f"  {os.path.basename(fpath)}", leave=False)):
            text = line.strip()
            if not text:
                continue

            formatted = template.format(text=text)

            encoded = tokenizer.encode_plus(
                formatted,
                max_length=max_length,
                padding='max_length' if len(formatted) < max_length else False,
                truncation=True,
                return_tensors='pt',
            )

            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
            labels = input_ids.clone()

            if tokenizer.pad_token_id is not None:
                labels[input_ids == tokenizer.pad_token_id] = -100

            sample_idx = file_count * 10000 + idx
            sample_filename = f"sample_{sample_idx:06d}.pt"
            sample_path = os.path.join(output_dir, sample_filename)

            num_tokens = (input_ids != tokenizer.pad_token_id).sum().item()

            tensor_data = {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
                "metadata": {
                    "text": text,
                    "source_file": os.path.basename(fpath),
                    "formatted_length": len(formatted),
                    "num_tokens": num_tokens,
                },
            }

            torch.save(tensor_data, sample_path)
            all_sample_paths.append(sample_filename)
            total_tokens += num_tokens

        file_count += 1

    # Save manifest
    manifest = {
        "source_directory": input_dir,
        "num_files_processed": file_count,
        "num_samples": len(all_sample_paths),
        "total_tokens": total_tokens,
        "max_length": max_length,
        "template": template,
        "pad_token_id": tokenizer.pad_token_id,
        "samples": all_sample_paths,
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(f"Preprocessed {len(all_sample_paths)} samples from {file_count} files to {output_dir}")
    logger.info(f"Total tokens: {total_tokens:,}")

    return manifest


def main():
    """Main entry point for preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess text data for LLM LoKr training")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input text file or directory containing .txt files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for preprocessed tensor files"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the LLM model or tokenizer"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096)"
    )
    parser.add_argument(
        "--template",
        type=str,
        default="{text}",
        help="Text template for formatting (default: {text})"
    )

    args = parser.parse_args()

    # Load tokenizer
    tokenizer = load_tokenizer(args.model_path)

    # Preprocess
    if os.path.isfile(args.input_path):
        if not args.input_path.endswith('.txt'):
            logger.warning(f"Input file does not have .txt extension: {args.input_path}")
        preprocess_text_file(
            input_path=args.input_path,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            max_length=args.max_length,
            template=args.template,
        )
    elif os.path.isdir(args.input_path):
        preprocess_text_directory(
            input_dir=args.input_path,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            max_length=args.max_length,
            template=args.template,
        )
    else:
        logger.error(f"Input path does not exist: {args.input_path}")
        return 1

    logger.info("Preprocessing completed successfully!")
    return 0


if __name__ == "__main__":
    main()
