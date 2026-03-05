"""LLM LoKr Training for ACE-Step 1.5 - Qwen2.5-based 5Hz-LM"""
import os, sys, argparse, json, random, importlib
from typing import Optional, Dict, Any, Tuple, List
from loguru import logger
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD, Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts, LinearLR, SequentialLR,
    CosineAnnealingLR, ConstantLR
)
from torch.utils.data import Dataset

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from acestep.training.configs import LoKRConfig, TrainingConfig
from acestep.training.llm_lokr_custom import (
    replace_linear_with_lokr, save_lokr_weights, load_lokr_weights,
    count_lokr_parameters,
)
from acestep.training.llm_dataset import create_dataloader

QWEN_PAD_TOKEN_ID = 151643


def _select_compute_dtype(device_type: str) -> torch.dtype:
    if device_type in ("cuda", "xpu"): return torch.bfloat16
    if device_type == "mps": return torch.float16
    return torch.float32


def _count_trainable_params(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def _load_jsonl_dataset(jsonl_path: str) -> List[Dict[str, Any]]:
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): samples.append(json.loads(line))
    return samples


class ChatDataset(Dataset):
    """Dataset for chat-formatted training data with Qwen2.5 template."""
    def __init__(self, samples: List[Dict[str, Any]], tokenizer, max_length: int = 4096):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        messages = sample.get("messages", [])
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoded = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = encoded['input_ids'].squeeze(0)
        # attention_mask should be 1D (seq_len) - the model will handle batching
        attention_mask = encoded['attention_mask'].squeeze(0)  # (1, seq_len) -> (seq_len)
        labels = input_ids.clone()
        # Mask user tokens - only train on assistant response
        assistant_id = self.tokenizer.convert_tokens_to_ids('</think>')
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == assistant_id:
                labels[:i+1] = -100
                break
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class LLMLoKrTrainer:
    """Trainer for LLM LoKr fine-tuning using custom implementation."""
    def __init__(self, llm_model, tokenizer, lokr_config: LoKRConfig, training_config: TrainingConfig, device=None):
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.lokr_config = lokr_config
        self.training_config = training_config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = QWEN_PAD_TOKEN_ID
        self.optimizer, self.scheduler = None, None
        self.is_training, self.should_stop = False, False
        self.replaced_modules = {}  # Track replaced modules
        self.use_fp8 = getattr(self.lokr_config, 'use_fp8', False)
        self.use_gradient_checkpointing = getattr(self.training_config, 'gradient_checkpointing', False)

    def setup(self) -> Dict[str, Any]:
        logger.info("Setting up LLM LoKr training with custom LoKr implementation...")
        compute_dtype = _select_compute_dtype(self.device.type)

        # Move model to device first
        self.llm_model = self.llm_model.to(self.device).to(compute_dtype)

        # Freeze all parameters first
        for param in self.llm_model.parameters():
            param.requires_grad = False

        # Replace target linear layers with LoKr
        target_modules = self.lokr_config.target_modules
        use_fp8 = getattr(self.lokr_config, 'use_fp8', False)
        logger.info(f"LoKr FP8 mode: {use_fp8}")
        self.replaced_modules = replace_linear_with_lokr(
            self.llm_model,
            target_modules,
            rank=self.lokr_config.linear_dim,
            alpha=self.lokr_config.linear_alpha,
            dropout=0.1,
            factor=self.lokr_config.factor if hasattr(self.lokr_config, 'factor') else -1,
            bypass_mode=getattr(self.lokr_config, 'bypass_mode', True),
            use_fp8=use_fp8,
        )
        
        # Move newly created LoKr layers to the device
        logger.info(f"Moving LoKr layers to {self.device}")
        self.llm_model = self.llm_model.to(self.device)

        # Enable gradient checkpointing for VRAM reduction
        if self.use_gradient_checkpointing:
            logger.info("Enabling gradient checkpointing for VRAM reduction")
            self.llm_model.gradient_checkpointing_enable()
            self.llm_model.enable_input_require_grads()

        logger.info(f"Replaced {len(self.replaced_modules)} linear layers with LoKr: {target_modules}")
        
        # Count parameters
        lokr_params, total_params = count_lokr_parameters(self.llm_model)
        logger.info(f"LoKr parameters: {lokr_params:,} / {total_params:,} ({lokr_params/total_params:.2%})")

        # Setup optimizer with only trainable parameters
        trainable_params = [p for p in self.llm_model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found!")

        # Create optimizer based on config
        optimizer_type = self.training_config.optimizer.lower()
        lr = self.training_config.learning_rate
        weight_decay = self.training_config.weight_decay
        
        if optimizer_type == "adamw":
            self.optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        elif optimizer_type == "adamw8bit":
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(trainable_params, lr=lr, weight_decay=weight_decay)
                logger.info("Using 8-bit AdamW optimizer")
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to AdamW")
                self.optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        elif optimizer_type == "sgd":
            self.optimizer = SGD(trainable_params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_type == "adam":
            self.optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        elif optimizer_type == "lion":
            try:
                from lion_pytorch import Lion
                self.optimizer = Lion(trainable_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))
                logger.info("Using Lion optimizer")
            except ImportError:
                logger.warning("lion_pytorch not available, falling back to AdamW")
                self.optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        elif optimizer_type == "prodigy":
            try:
                from prodigyopt import Prodigy
                self.optimizer = Prodigy(trainable_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
                logger.info("Using Prodigy optimizer (auto-tuning learning rate)")
            except ImportError:
                logger.warning("prodigyopt not available, falling back to AdamW")
                self.optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        else:
            self.optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        
        logger.info(f"Optimizer: {self.optimizer.__class__.__name__}")
        
        # Scheduler will be created in train() method after dataloader is available
        self.scheduler = None
        logger.info(f"Scheduler: will be configured in train() method")

        return {"replaced_modules": len(self.replaced_modules), "lokr_params": lokr_params}

    def _setup_scheduler(self, num_batches: int):
        """Setup learning rate scheduler based on config."""
        if self.optimizer is None:
            raise ValueError("Optimizer must be set up before scheduler")
        
        scheduler_type = self.training_config.scheduler.lower()
        total_steps = max(1, num_batches * self.training_config.max_epochs)
        warmup_steps = min(self.training_config.warmup_steps, max(1, total_steps // 10))
        
        if scheduler_type == "cosine":
            warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
            main_scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps,
                                                eta_min=self.training_config.learning_rate * 0.1)
            self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, main_scheduler],
                                           milestones=[warmup_steps])
            logger.info(f"Scheduler: LinearWarmup + CosineAnnealingLR (total_steps={total_steps})")
        elif scheduler_type == "cosine_restarts":
            warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
            main_scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=max(100, total_steps - warmup_steps),
                                                          T_mult=1, eta_min=self.training_config.learning_rate * 0.1)
            self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, main_scheduler],
                                           milestones=[warmup_steps])
            logger.info(f"Scheduler: LinearWarmup + CosineAnnealingWarmRestarts (total_steps={total_steps})")
        elif scheduler_type == "linear":
            warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
            main_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1,
                                       total_iters=total_steps - warmup_steps)
            self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, main_scheduler],
                                           milestones=[warmup_steps])
            logger.info(f"Scheduler: LinearWarmup + LinearDecay (total_steps={total_steps})")
        elif scheduler_type == "constant":
            self.scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=float('inf'))
            logger.info("Scheduler: Constant LR")
        else:
            # Default: cosine
            warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
            main_scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps,
                                                eta_min=self.training_config.learning_rate * 0.1)
            self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, main_scheduler],
                                           milestones=[warmup_steps])
            logger.info(f"Scheduler: LinearWarmup + CosineAnnealingLR (default, total_steps={total_steps})")

    def train_epoch(self, dataloader, epoch: int, global_step: int):
        self.llm_model.train()
        total_loss, num_batches = 0.0, 0
        for batch_idx, batch in enumerate(dataloader):
            if self.should_stop:
                break
            # Ensure 2D input (batch, seq_len)
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                labels = labels.unsqueeze(0)
            # attention_mask is not needed for now - model handles it internally
            attention_mask = None
            labels[input_ids == self.tokenizer.pad_token_id] = -100
            compute_dtype = _select_compute_dtype(self.device.type)
            
            # Forward pass with autocast (FP8 mode uses BF16/FP16 for stability)
            with torch.autocast(device_type=self.device.type, dtype=compute_dtype):
                outputs = self.llm_model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                loss = outputs.loss / self.training_config.gradient_accumulation_steps
            
            loss.backward()
            if (batch_idx + 1) % self.training_config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.llm_model.parameters() if p.requires_grad],
                    self.training_config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1
                total_loss += loss.item() * self.training_config.gradient_accumulation_steps
                num_batches += 1
                if global_step % self.training_config.log_every_n_steps == 0:
                    avg_loss = total_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(f"Epoch {epoch}, Step {global_step}, Loss: {avg_loss:.4f}, LR: {lr:.6f}")
                    yield global_step, avg_loss, f"Epoch {epoch}, Step {global_step}, Loss: {avg_loss:.4f}"
        return global_step, (total_loss / num_batches) if num_batches > 0 else 0.0, f"Epoch {epoch} completed"

    def train(self, tensor_dir: str, resume_from: Optional[str] = None):
        self.is_training = True
        global_step, start_epoch = 0, 0
        try:
            setup_info = self.setup()
            trainable, total = _count_trainable_params(self.llm_model)
            logger.info(f"Trainable params: {trainable:,}/{total:,} ({trainable/total:.2%})")
            if resume_from and os.path.exists(resume_from):
                logger.info(f"Resuming from checkpoint: {resume_from}")
                # Load LoKr weights
                load_lokr_weights(self.llm_model, resume_from)
                start_epoch = 0  # Could be extended to save/load epoch info
                global_step = 0
            
            jsonl_path = os.path.join(tensor_dir, "train.jsonl")
            if not os.path.exists(jsonl_path):
                raise FileNotFoundError(f"Dataset not found: {jsonl_path}")
            samples = _load_jsonl_dataset(jsonl_path)
            dataset = ChatDataset(samples, self.tokenizer,
                                  max_length=getattr(self.training_config, 'max_length', 4096))
            dataloader = create_dataloader(dataset, batch_size=self.training_config.batch_size,
                                           shuffle=True, num_workers=0, pin_memory=self.device.type == "cuda")
            logger.info(f"Dataset loaded: {len(dataset)} samples")
            
            # Setup scheduler now that dataloader is available
            self._setup_scheduler(len(dataloader))
            
            for epoch in range(start_epoch, self.training_config.max_epochs):
                logger.info(f"Epoch {epoch + 1}/{self.training_config.max_epochs}")
                yield epoch, 0.0, f"Starting epoch {epoch + 1}/{self.training_config.max_epochs}"
                global_step, epoch_loss, msg = yield from self.train_epoch(dataloader, epoch, global_step)
                if (epoch + 1) % self.training_config.save_every_n_epochs == 0:
                    checkpoint_dir = os.path.join(self.training_config.output_dir,
                                                   f"checkpoint_epoch_{epoch + 1}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    save_lokr_weights(self.llm_model, os.path.join(checkpoint_dir, "lokr_weights.pt"))
                    # Save config
                    config_path = os.path.join(checkpoint_dir, "config.json")
                    with open(config_path, 'w') as f:
                        json.dump({
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "lokr_config": self.lokr_config.to_dict(),
                        }, f, indent=2)
                    logger.info(f"Checkpoint saved: {checkpoint_dir}")
            final_dir = os.path.join(self.training_config.output_dir, "final")
            os.makedirs(final_dir, exist_ok=True)
            save_lokr_weights(self.llm_model, os.path.join(final_dir, "lokr_weights.pt"))
            # Save final config
            config_path = os.path.join(final_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump({
                    "epoch": self.training_config.max_epochs,
                    "global_step": global_step,
                    "lokr_config": self.lokr_config.to_dict(),
                }, f, indent=2)
            logger.info(f"Training completed! Weights saved to {final_dir}")
            yield global_step, epoch_loss, "Training completed!"
        except Exception as e:
            logger.exception("LLM LoKr training failed")
            yield global_step, 0.0, f"Training failed: {str(e)}"
        finally:
            self.is_training = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def load_llm_model(model_path: str, device: torch.device):
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers required: pip install transformers")
    logger.info(f"Loading LLM model from {model_path}")
    compute_dtype = _select_compute_dtype(device.type)

    # Determine attention implementation
    # flash_attn has compatibility issues on Windows, use sdpa instead
    attn_impl = None
    if device.type == "cuda":
        try:
            # Try flash_attention_2 first
            import flash_attn
            attn_impl = "flash_attention_2"
            logger.info("Using flash Attention 2")
        except ImportError:
            # Fall back to sdpa (scaled dot-product attention)
            attn_impl = "sdpa"
            logger.info("Flash Attention not available, using SDPA instead")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=compute_dtype, trust_remote_code=True,
        attn_implementation=attn_impl)
    logger.info(f"LLM model loaded: {model.__class__.__name__}")
    
    # No patching needed - LoKrLinear now handles 3D inputs correctly
    logger.info("LoKrLinear handles 3D inputs correctly")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train LLM with LoKr adapters")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LLM model")
    parser.add_argument("--tensor_dir", type=str, required=True, help="Directory with preprocessed data")
    parser.add_argument("--output_dir", type=str, default="./llm_lokr_output", help="Output directory")
    parser.add_argument("--lokr_linear_dim", type=int, default=64, help="LoKr linear dimension")
    parser.add_argument("--lokr_linear_alpha", type=int, default=64, help="LoKr linear alpha")
    parser.add_argument("--lokr_factor", type=int, default=-1, help="LoKr factor")
    parser.add_argument("--bypass_mode", action="store_true", default=True, help="Enable bypass mode")
    parser.add_argument("--no_bypass_mode", action="store_false", dest="bypass_mode", help="Disable bypass mode")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum epochs")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--use_fp8", action="store_true", default=False, help="Use pseudo FP8 for VRAM")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Enable gradient checkpointing for VRAM reduction")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adamw8bit", "sgd", "adam", "lion", "prodigy"], help="Optimizer type")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "linear", "constant", "cosine_restarts"], help="Scheduler type")
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    lokr_config = LoKRConfig(
        linear_dim=args.lokr_linear_dim,
        linear_alpha=args.lokr_linear_alpha,
        factor=args.lokr_factor,
        use_fp8=args.use_fp8,
        bypass_mode=args.bypass_mode,
    )
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_epochs=args.max_epochs,
        save_every_n_epochs=1,  # Save every epoch for testing
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        output_dir=args.output_dir,
        max_length=args.max_length,
        gradient_checkpointing=args.gradient_checkpointing,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
    )
    try:
        llm_model = load_llm_model(args.model_path, device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    trainer = LLMLoKrTrainer(llm_model, tokenizer, lokr_config, training_config, device)
    logger.info("Starting LLM LoKr training...")
    for step, loss, msg in trainer.train(args.tensor_dir, args.resume_from):
        logger.info(msg)
    logger.info("Training completed!")


if __name__ == "__main__":
    main()