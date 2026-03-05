"""
Preprocessing Script for LLM LoKr Training

Creates training data for 5Hz-LM (Qwen2.5-based) by:
1. Converting audio files to audio codes using DiT tokenizer
2. Generating metadata and lyrics from audio codes using LLM
3. Saving formatted training data for LoKr fine-tuning

The 5Hz-LM learns to generate structured output:
 <think>
  bpm: 120
  caption: A calm piano melody
  duration: 180
  keyscale: C major
  language: en
  timesignature: 4
  </think>
  <|audio_code_12345|><|audio_code_67890|>...
"""

import os
import sys
import json
import argparse
import time
import gc
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import torch
import torchaudio
from loguru import logger
import yaml
from tqdm import tqdm


def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    logger.remove()
    level = "DEBUG" if debug else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )


def load_acestep_handler(lm_model_name: Optional[str] = None):
    """
    Load ACE-Step handler for audio encoding and LLM inference.

    Args:
        lm_model_name: Optional LLM model name to use. 
                       If None, auto-select from available models.

    Returns:
        Tuple of (handler, llm_handler)
    """
    logger.info("Loading ACE-Step handlers...")

    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler

    # Get checkpoint directory - try multiple paths
    checkpoint_dir = None

    # Try 1: Relative to this file (acestep/training -> checkpoints)
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    candidate = os.path.join(project_root, "checkpoints")
    if os.path.exists(candidate):
        checkpoint_dir = candidate
        logger.info(f"Found checkpoints directory: {checkpoint_dir}")

    # Try 2: Current working directory
    if checkpoint_dir is None:
        candidate = os.path.join(os.getcwd(), "checkpoints")
        if os.path.exists(candidate):
            checkpoint_dir = candidate
            logger.info(f"Found checkpoints directory: {checkpoint_dir}")

    # Try 3: Default path
    if checkpoint_dir is None:
        checkpoint_dir = r"C:\Users\newuser\Desktop\ACE-Step-1.5\checkpoints"
        if not os.path.exists(checkpoint_dir):
            checkpoint_dir = os.path.join(project_root, "acestep", "checkpoints")

    if not os.path.exists(checkpoint_dir):
        raise RuntimeError(f"Checkpoints directory not found: {checkpoint_dir}")

    # Find 5Hz-LM model
    lm_model_path = None
    
    if lm_model_name:
        # Use specified model name
        candidate = os.path.join(checkpoint_dir, lm_model_name)
        if os.path.exists(candidate):
            lm_model_path = candidate
            logger.info(f"Using specified LLM model: {lm_model_name}")
        else:
            raise RuntimeError(
                f"Specified model '{lm_model_name}' not found in {checkpoint_dir}. "
                f"Available models: {os.listdir(checkpoint_dir)}"
            )
    else:
        # Auto-select from available models (priority order)
        for model_name in ["acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-4B"]:
            candidate = os.path.join(checkpoint_dir, model_name)
            if os.path.exists(candidate):
                lm_model_path = candidate
                break

    if lm_model_path is None:
        available_models = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
        raise RuntimeError(
            f"5Hz-LM model not found in {checkpoint_dir}. "
            f"Available: {available_models}. "
            "Please download the model first or specify --lm_model."
        )

    logger.info(f"Using LLM model: {lm_model_path}")

    # Initialize handlers
    handler = AceStepHandler()
    llm_handler = LLMHandler()

    # Initialize DiT handler with CPU offload for memory efficiency
    logger.info("Initializing DiT handler (with CPU offload for memory efficiency)...")
    # Get project root (parent of checkpoints directory)
    project_root = os.path.dirname(checkpoint_dir)
    handler.initialize_service(
        project_root=project_root,
        config_path="acestep-v15-sft",  # Use turbo model
        device="cuda",  # Explicitly use CUDA
        offload_to_cpu=True,  # Enable CPU offload to reduce VRAM usage
    )

    # Initialize LLM handler with CPU offload for memory efficiency
    logger.info("Initializing LLM handler (with CPU offload for memory efficiency)...")
    llm_handler.initialize(
        checkpoint_dir=checkpoint_dir,
        lm_model_path=lm_model_path,
        backend="pt",  # Use PyTorch backend for preprocessing
        device="cuda",  # Explicitly use CUDA
        offload_to_cpu=True,  # Enable CPU offload for 4B model support on 16GB VRAM
    )

    logger.info(f"Handlers initialized. LLM: {lm_model_path}")
    return handler, llm_handler


def convert_audio_to_codes(
    dit_handler,
    audio_path: str,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Convert audio file to audio codes using DiT tokenizer.
    
    Args:
        dit_handler: DiT handler instance
        audio_path: Path to audio file
        
    Returns:
        Tuple of (audio_codes_string, metadata)
    """
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return None, None
    
    try:
        # Convert audio to codes
        logger.info(f"Converting audio to codes: {os.path.basename(audio_path)}")
        audio_codes = dit_handler.convert_src_audio_to_codes(audio_path)
        
        if not audio_codes or audio_codes.startswith("❌"):
            logger.error(f"Failed to convert audio: {audio_codes}")
            return None, None
        
        # Count codes
        code_count = audio_codes.count("<|audio_code_")
        logger.info(f"Generated {code_count} audio codes")
        
        # Extract duration from codes (5 codes = 1 second)
        duration_seconds = code_count / 5
        
        metadata = {
            "audio_path": audio_path,
            "num_codes": code_count,
            "duration_seconds": duration_seconds,
        }
        
        return audio_codes, metadata
        
    except Exception as e:
        logger.exception(f"Error converting audio: {e}")
        return None, None


def generate_metadata_from_codes(
    llm_handler,
    audio_codes: str,
    temperature: float = 0.3,
    use_constrained_decoding: bool = True,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Generate metadata and lyrics from audio codes using LLM.
    
    This is the "understanding" task: audio codes → metadata + lyrics
    
    Args:
        llm_handler: LLM handler instance
        audio_codes: Audio codes string
        temperature: Sampling temperature
        use_constrained_decoding: Use FSM-based constrained decoding
        
    Returns:
        Tuple of (metadata_dict, status_message)
    """
    if not audio_codes or not audio_codes.strip():
        return None, "No audio codes provided"
    
    try:
        logger.info("Generating metadata from audio codes...")
        
        metadata, status = llm_handler.understand_audio_from_codes(
            audio_codes=audio_codes,
            temperature=temperature,
            use_constrained_decoding=use_constrained_decoding,
        )
        
        if not metadata:
            logger.error(f"LLM failed to generate metadata: {status}")
            return None, status
        
        logger.info(f"Metadata generated: bpm={metadata.get('bpm', 'N/A')}, "
                   f"caption={len(metadata.get('caption', ''))} chars, "
                   f"language={metadata.get('language', 'N/A')}")
        
        return metadata, status
        
    except Exception as e:
        logger.exception(f"Error generating metadata: {e}")
        return None, f"Error: {str(e)}"


def format_training_example(
    metadata: Dict[str, Any],
    audio_codes: str,
    lyrics: Optional[str] = None,
) -> str:
    """
    Format a training example for 5Hz-LM.
    
    Creates the expected output format:
    <think>
    bpm: 120
    caption: A calm piano melody
    duration: 180
    keyscale: C major
    language: en
    timesignature: 4
    </think>
    [lyrics if available]
    <|audio_code_12345|><|audio_code_67890|>...
    
    Args:
        metadata: Metadata dictionary from LLM
        audio_codes: Audio codes string
        lyrics: Optional lyrics
        
    Returns:
        Formatted training text
    """
    # Build reasoning section
    reasoning_lines = ["<think>"]
    
    # Add metadata fields
    if "bpm" in metadata:
        reasoning_lines.append(f"bpm: {metadata['bpm']}")
    if "caption" in metadata:
        reasoning_lines.append(f"caption: {metadata['caption']}")
    if "duration" in metadata:
        reasoning_lines.append(f"duration: {metadata['duration']}")
    if "keyscale" in metadata:
        reasoning_lines.append(f"keyscale: {metadata['keyscale']}")
    if "language" in metadata:
        reasoning_lines.append(f"language: {metadata['language']}")
    if "timesignature" in metadata:
        reasoning_lines.append(f"timesignature: {metadata['timesignature']}")
    if "genres" in metadata:
        reasoning_lines.append(f"genres: {metadata['genres']}")
    
    reasoning_lines.append("</think>")
    
    # Add lyrics if available
    if lyrics:
        reasoning_lines.append(lyrics)
    
    # Add audio codes
    reasoning_lines.append(audio_codes)
    
    return "\n".join(reasoning_lines)


def create_training_prompt(
    tags: str,
    lyrics: str = "",
    bpm: int = 120,
    duration: float = 120.0,
    timesignature: int = 4,
    language: str = "en",
    keyscale: str = "C major",
    seed: int = 0,
    generate_audio_codes: bool = True,
    cfg_scale: float = 2.0,
    temperature: float = 0.85,
    top_p: float = 0.9,
) -> str:
    """
    Create a training prompt matching the ACE-Step 1.5 format.
    
    This creates the user input that would generate the expected output.
    """
    # The prompt format depends on the model's expected input
    # For 5Hz-LM, the input is typically a chat-formatted prompt
    prompt_parts = []
    
    if tags:
        prompt_parts.append(f"Tags: {tags}")
    if bpm:
        prompt_parts.append(f"BPM: {bpm}")
    if duration:
        prompt_parts.append(f"Duration: {duration}s")
    if timesignature:
        prompt_parts.append(f"Time Signature: {timesignature}/4")
    if language:
        prompt_parts.append(f"Language: {language}")
    if keyscale:
        prompt_parts.append(f"Keyscale: {keyscale}")
    if lyrics:
        prompt_parts.append(f"Lyrics: {lyrics}")
    
    return "\n".join(prompt_parts)


def load_existing_metadata(audio_path: str) -> Optional[Dict[str, Any]]:
    """
    Load existing metadata files (.caption.txt, .lyrics.txt, .json) if available.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Metadata dictionary or None if files not found
    """
    base_path = Path(audio_path)
    base_name = base_path.stem
    # Remove common extensions to get base name
    for ext in ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.webm']:
        if base_name.endswith(ext):
            base_name = base_name[:-len(ext)]
    
    parent_dir = base_path.parent
    caption_file = parent_dir / f"{base_name}.caption.txt"
    lyrics_file = parent_dir / f"{base_name}.lyrics.txt"
    json_file = parent_dir / f"{base_name}.json"
    
    metadata = {}
    found_files = []
    
    # Load caption
    if caption_file.exists():
        with open(caption_file, 'r', encoding='utf-8') as f:
            metadata['caption'] = f.read().strip()
        found_files.append('caption')
    
    # Load lyrics
    if lyrics_file.exists():
        with open(lyrics_file, 'r', encoding='utf-8') as f:
            metadata['lyrics'] = f.read().strip()
        found_files.append('lyrics')
    
    # Load JSON metadata
    if json_file.exists():
        with open(json_file, 'r', encoding='utf-8') as f:
            json_meta = json.load(f)
        # Map common fields
        field_mapping = {
            'bpm': ['bpm', 'BPM'],
            'duration': ['duration', 'duration_seconds', 'length'],
            'keyscale': ['keyscale', 'key', 'scale', 'key_scale'],
            'language': ['language', 'lang'],
            'timesignature': ['timesignature', 'time_signature', 'time_sig'],
            'genres': ['genres', 'genre', 'tags'],
        }
        for target_key, source_keys in field_mapping.items():
            for source_key in source_keys:
                if source_key in json_meta:
                    metadata[target_key] = json_meta[source_key]
                    break
        found_files.append('json')
    
    if found_files:
        logger.debug(f"Loaded existing metadata for {base_name}: {', '.join(found_files)}")
        return metadata
    return None


def preprocess_audio_file(
    dit_handler,
    llm_handler,
    audio_path: str,
    output_dir: str,
    skip_if_exists: bool = True,
    use_existing_metadata: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Preprocess a single audio file for LLM training.

    Pipeline:
    1. Convert audio to audio codes
    2. Load existing metadata (if available) or generate from codes
    3. Save training example

    Args:
        dit_handler: DiT handler
        llm_handler: LLM handler
        audio_path: Path to audio file
        output_dir: Output directory
        skip_if_exists: Skip if output already exists
        use_existing_metadata: Use existing .caption.txt/.lyrics.txt/.json files

    Returns:
        Metadata dictionary or None on failure
    """
    audio_name = Path(audio_path).stem
    output_file = os.path.join(output_dir, f"{audio_name}.json")

    # Check if already processed
    if skip_if_exists and os.path.exists(output_file):
        logger.info(f"Skipping {audio_name} - already processed")
        with open(output_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    logger.info(f"Processing: {audio_name}")

    # Step 1: Convert audio to codes
    audio_codes, audio_meta = convert_audio_to_codes(dit_handler, audio_path)
    if audio_codes is None:
        return None

    # Step 2: Load or generate metadata
    metadata = None
    if use_existing_metadata:
        metadata = load_existing_metadata(audio_path)
        if metadata:
            logger.info(f"  ✓ Loaded existing metadata for {audio_name}")

    if metadata is None:
        # Generate metadata from codes using LLM
        metadata, status = generate_metadata_from_codes(llm_handler, audio_codes)
        if metadata is None:
            logger.warning(f"  ⚠ LLM metadata generation failed: {status}")
            # Create minimal metadata
            metadata = {
                "caption": f"Audio track {audio_name}",
                "language": "unknown",
            }
        else:
            logger.info(f"  ✓ Generated metadata via LLM")

    # Step 3: Format training example
    lyrics = metadata.get('lyrics', '')
    training_text = format_training_example(metadata, audio_codes, lyrics)

    # Create prompt (user input)
    prompt = create_training_prompt(
        tags=metadata.get('caption', ''),
        lyrics=lyrics,
        bpm=metadata.get('bpm', 120),
        duration=metadata.get('duration', 120),
        timesignature=int(metadata.get('timesignature', 4)),
        language=metadata.get('language', 'en'),
        keyscale=metadata.get('keyscale', 'C major'),
    )

    # Save result
    result = {
        "audio_path": audio_path,
        "audio_name": audio_name,
        "prompt": prompt,
        "completion": training_text,
        "metadata": metadata,
        "audio_codes": audio_codes,
        "num_codes": audio_meta.get("num_codes", 0),
        "duration_seconds": audio_meta.get("duration_seconds", 0),
        "used_existing_metadata": metadata is not None and use_existing_metadata,
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved: {output_file}")
    return result


def convert_all_audio_to_codes(
    dit_handler,
    audio_files: List[str],
    output_dir: str,
    skip_if_exists: bool = True,
) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    """
    Convert all audio files to codes in a batch.
    
    This function processes all audio files and saves intermediate results,
    then releases VRAM before LLM processing.
    
    Args:
        dit_handler: DiT handler
        audio_files: List of audio file paths
        output_dir: Output directory for intermediate results
        skip_if_exists: Skip if output already exists
        
    Returns:
        Dictionary mapping audio_name to (audio_codes, audio_meta)
    """
    logger.info("Phase 1: Converting all audio files to codes...")
    audio_codes_dict = {}
    
    for i, audio_path in enumerate(tqdm(audio_files, desc="Converting audio to codes")):
        audio_name = Path(audio_path).stem
        intermediate_file = os.path.join(output_dir, f"{audio_name}.codes.json")
        final_file = os.path.join(output_dir, f"{audio_name}.json")

        # Skip if final output already exists (no need for intermediate file)
        if skip_if_exists and os.path.exists(final_file):
            logger.debug(f"Skipping {audio_name} - final output already exists")
            # Clean up intermediate file if it exists
            if os.path.exists(intermediate_file):
                os.remove(intermediate_file)
                logger.debug(f"Removed intermediate file: {intermediate_file}")
            continue

        # Check if intermediate file exists
        if skip_if_exists and os.path.exists(intermediate_file):
            logger.debug(f"Skipping {audio_name} - already converted")
            with open(intermediate_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            audio_codes_dict[audio_name] = (data["audio_codes"], data["audio_meta"])
            continue
        
        # Convert audio to codes
        audio_codes, audio_meta = convert_audio_to_codes(dit_handler, audio_path)
        if audio_codes is None:
            logger.warning(f"Failed to convert {audio_name}, skipping...")
            continue
        
        # Save intermediate result
        audio_codes_dict[audio_name] = (audio_codes, audio_meta)
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump({
                "audio_path": audio_path,
                "audio_codes": audio_codes,
                "audio_meta": audio_meta,
            }, f, ensure_ascii=False, indent=2)

        # Periodic cleanup every 10 files (reduced from 3 for faster processing)
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    logger.info(f"Converted {len(audio_codes_dict)}/{len(audio_files)} files to codes")
    return audio_codes_dict


def generate_all_metadata_and_save(
    llm_handler,
    audio_files: List[str],
    audio_codes_dict: Dict[str, Tuple[str, Dict[str, Any]]],
    output_dir: str,
    use_existing_metadata: bool = True,
    temperature: float = 0.3,
):
    """
    Generate metadata for all audio codes and save training examples.
    
    This function processes all audio codes and generates metadata using LLM,
    then saves the final training examples.
    
    Args:
        llm_handler: LLM handler
        audio_files: List of audio file paths
        audio_codes_dict: Dictionary mapping audio_name to (audio_codes, audio_meta)
        output_dir: Output directory
        use_existing_metadata: Use existing metadata files if available
        temperature: LLM temperature for metadata generation
    """
    logger.info("Phase 2: Generating metadata and saving training examples...")
    
    samples = []
    for audio_path in tqdm(audio_files, desc="Generating metadata"):
        audio_name = Path(audio_path).stem
        output_file = os.path.join(output_dir, f"{audio_name}.json")
        
        # Skip if already fully processed
        if os.path.exists(output_file):
            logger.debug(f"Skipping {audio_name} - already processed")
            with open(output_file, 'r', encoding='utf-8') as f:
                samples.append(json.load(f))
            continue
        
        # Get audio codes from intermediate result
        if audio_name not in audio_codes_dict:
            logger.warning(f"No audio codes found for {audio_name}, skipping...")
            continue
        
        audio_codes, audio_meta = audio_codes_dict[audio_name]
        
        # Load or generate metadata
        metadata = None
        if use_existing_metadata:
            metadata = load_existing_metadata(audio_path)
            if metadata:
                logger.debug(f"  ✓ Loaded existing metadata for {audio_name}")
        
        if metadata is None:
            # Generate metadata from codes using LLM
            metadata, status = generate_metadata_from_codes(
                llm_handler, 
                audio_codes,
                temperature=temperature,
            )
            if metadata is None:
                logger.warning(f"  ⚠ LLM metadata generation failed for {audio_name}")
                # Create minimal metadata
                metadata = {
                    "caption": f"Audio track {audio_name}",
                    "language": "unknown",
                }
            else:
                logger.debug(f"  ✓ Generated metadata via LLM")
        
        # Format training example
        lyrics = metadata.get('lyrics', '')
        training_text = format_training_example(metadata, audio_codes, lyrics)
        
        # Create prompt (user input)
        prompt = create_training_prompt(
            tags=metadata.get('caption', ''),
            lyrics=lyrics,
            bpm=metadata.get('bpm', 120),
            duration=metadata.get('duration', 120),
            timesignature=int(metadata.get('timesignature', 4)),
            language=metadata.get('language', 'en'),
            keyscale=metadata.get('keyscale', 'C major'),
        )
        
        # Save result
        result = {
            "audio_path": audio_path,
            "audio_name": audio_name,
            "prompt": prompt,
            "completion": training_text,
            "metadata": metadata,
            "audio_codes": audio_codes,
            "num_codes": audio_meta.get("num_codes", 0),
            "duration_seconds": audio_meta.get("duration_seconds", 0),
            "used_existing_metadata": metadata is not None and use_existing_metadata,
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        samples.append(result)

        # Periodic cleanup every 10 files (reduced from 5 for faster processing)
        if len(samples) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    logger.info(f"Generated metadata for {len(samples)} files")
    return samples


def create_training_manifest(
    output_dir: str,
    samples: List[Dict[str, Any]],
) -> str:
    """
    Create a manifest file for training data.
    
    Args:
        output_dir: Output directory
        samples: List of sample metadata
        
    Returns:
        Path to manifest file
    """
    manifest = {
        "num_samples": len(samples),
        "total_codes": sum(s.get("num_codes", 0) for s in samples),
        "total_duration_seconds": sum(s.get("duration_seconds", 0) for s in samples),
        "samples": [
            {
                "audio_name": s["audio_name"],
                "num_codes": s["num_codes"],
                "duration_seconds": s["duration_seconds"],
                "metadata": {
                    "bpm": s["metadata"].get("bpm"),
                    "language": s["metadata"].get("language"),
                    "keyscale": s["metadata"].get("keyscale"),
                },
            }
            for s in samples
        ],
    }
    
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Manifest saved: {manifest_path}")
    return manifest_path


def convert_to_huggingface_format(
    input_dir: str,
    output_path: str,
):
    """
    Convert preprocessed data to HuggingFace dataset format.
    
    Creates a JSONL file with "text" field for each training example.
    This format is compatible with HuggingFace datasets and TRL.
    
    Args:
        input_dir: Directory containing preprocessed JSON files
        output_path: Output JSONL file path
    """
    logger.info("Converting to HuggingFace format...")

    samples = []
    for fname in os.listdir(input_dir):
        if fname.endswith('.json') and fname != 'manifest.json':
            fpath = os.path.join(input_dir, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Skip if required keys are missing (incomplete processing)
            if "prompt" not in data or "completion" not in data:
                logger.debug(f"Skipping {fname} - missing required keys")
                continue

            # Create chat format for Qwen2.5
            conversation = [
                {
                    "role": "user",
                    "content": data["prompt"]
                },
                {
                    "role": "assistant",
                    "content": data["completion"]
                }
            ]

            samples.append({"messages": conversation})
    
    # Save as JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"HuggingFace format saved: {output_path} ({len(samples)} samples)")
    return output_path


def main():
    """Main entry point for preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess audio files for LLM LoKr training"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing audio files (wav, mp3, flac)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for preprocessed data"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of samples to process (-1 = all)"
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        default=True,
        help="Skip already processed files"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM temperature for metadata generation"
    )
    parser.add_argument(
        "--use_existing_metadata",
        action="store_true",
        default=True,
        help="Use existing .caption.txt/.lyrics.txt/.json files if available"
    )
    parser.add_argument(
        "--no_existing_metadata",
        action="store_false",
        dest="use_existing_metadata",
        help="Ignore existing metadata files and generate all via LLM"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--lm_model",
        type=str,
        default=None,
        help="LLM model name to use (e.g., acestep-5Hz-lm-4B). "
             "If not specified, auto-selects from available models."
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List available LLM models and exit"
    )
    
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.debug)

    # List available models if requested
    if args.list_models:
        # Find checkpoint directory
        checkpoint_dir = None
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        for candidate in [
            os.path.join(project_root, "checkpoints"),
            os.path.join(os.getcwd(), "checkpoints"),
            r"C:\Users\newuser\Desktop\ACE-Step-1.5\checkpoints",
        ]:
            if os.path.exists(candidate):
                checkpoint_dir = candidate
                break
        
        if checkpoint_dir:
            models = [d for d in os.listdir(checkpoint_dir) 
                     if os.path.isdir(os.path.join(checkpoint_dir, d)) and "5Hz-lm" in d]
            logger.info(f"Available LLM models in {checkpoint_dir}:")
            for model in models:
                logger.info(f"  - {model}")
        else:
            logger.error("Checkpoints directory not found")
        sys.exit(0)

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Find audio files
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    audio_files = []
    for fname in os.listdir(args.input_dir):
        if Path(fname).suffix.lower() in audio_extensions:
            audio_files.append(os.path.join(args.input_dir, fname))

    if not audio_files:
        logger.error(f"No audio files found in {args.input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(audio_files)} audio files")

    # Limit samples if specified
    if args.max_samples > 0:
        audio_files = audio_files[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================
    # Memory-efficient 2-phase pipeline for 16GB VRAM
    # Phase 1: Convert all audio to codes (DiT only)
    # Phase 2: Generate metadata from codes (LLM only)
    # =========================================================
    
    samples = []
    
    try:
        # Load both handlers
        logger.info("Loading handlers...")
        dit_handler, llm_handler = load_acestep_handler(lm_model_name=args.lm_model)
        
        # Phase 1: Convert all audio files to codes using DiT
        audio_codes_dict = convert_all_audio_to_codes(
            dit_handler=dit_handler,
            audio_files=audio_files,
            output_dir=args.output_dir,
            skip_if_exists=args.skip_if_exists,
        )
        
        if not audio_codes_dict:
            logger.error("No audio files were successfully converted")
            sys.exit(1)
        
        # Release DiT from VRAM before loading LLM
        logger.info("Releasing DiT handler to free VRAM...")
        del dit_handler
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(0.5)  # Reduced from 3s for faster processing
        
        # Phase 2: Generate metadata using LLM and save training examples
        samples = generate_all_metadata_and_save(
            llm_handler=llm_handler,
            audio_files=audio_files,
            audio_codes_dict=audio_codes_dict,
            output_dir=args.output_dir,
            use_existing_metadata=args.use_existing_metadata,
            temperature=args.temperature,
        )
        
        # Release LLM from VRAM
        logger.info("Releasing LLM handler to free VRAM...")
        del llm_handler
        torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        logger.exception("Traceback:")
        sys.exit(1)

    # Create manifest and convert to HuggingFace format
    if samples:
        create_training_manifest(args.output_dir, samples)

        # Convert to HuggingFace format
        hf_output = os.path.join(args.output_dir, "train.jsonl")
        convert_to_huggingface_format(args.output_dir, hf_output)

        logger.info(f"Preprocessing completed: {len(samples)}/{len(audio_files)} successful")
    else:
        logger.error("No samples were successfully processed")
        sys.exit(1)

    # Cleanup intermediate files
    logger.info("Cleaning up intermediate files...")
    for fname in os.listdir(args.output_dir):
        if fname.endswith('.codes.json'):
            os.remove(os.path.join(args.output_dir, fname))
    
    torch.cuda.empty_cache()
    logger.info("Done!")


if __name__ == "__main__":
    main()
