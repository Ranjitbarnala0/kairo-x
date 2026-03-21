"""
Training Loop with Curriculum
===============================

The main training loop for the KAIRO-X controller. Integrates:
    - Model creation and initialization
    - Data pipeline with curriculum-aware sampling
    - 3-phase curriculum with gating
    - 6-loss computation with weighting
    - Evaluation and metric tracking
    - Checkpointing and logging
    - Export to binary format

Run with:
    python -m training.trainer [--config CONFIG_PATH] [--resume CHECKPOINT]

The trainer processes both single-step (context/budget) and sequential
(action/session) batches, handling recurrent state management for the
liquid blocks.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from training.model.controller import KairoController, ControllerConfig
from training.data_factory.pipeline import (
    TrainingPipeline,
    PipelineConfig,
    TrainingBatch,
    CurriculumPhase,
)
from training.losses import CombinedLoss, LossWeights
from training.curriculum import CurriculumManager, CurriculumConfig
from training.evaluation import Evaluator
from training.export import export_model, verify_export

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Full training configuration."""

    # Model
    model: ControllerConfig = field(default_factory=ControllerConfig)

    # Data
    data: PipelineConfig = field(default_factory=PipelineConfig)

    # Curriculum
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    # Losses
    loss_weights: LossWeights = field(default_factory=LossWeights)

    # Training
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    seed: int = 42
    log_every_steps: int = 50
    checkpoint_every_steps: int = 2000
    eval_batches: int = 100

    # Distributed training (DDP)
    distributed: bool = False
    local_rank: int = 0

    # Paths
    output_dir: str = "outputs/kairo_training"
    checkpoint_dir: str = "outputs/kairo_training/checkpoints"
    export_dir: str = "outputs/kairo_training/export"
    log_file: str = "outputs/kairo_training/training.log"


class Trainer:
    """
    Main training orchestrator for the KAIRO-X controller.

    Manages the full training lifecycle:
        1. Initialize model, optimizer, data pipeline
        2. Run 3-phase curriculum loop
        3. Evaluate, checkpoint, log
        4. Export final model

    Usage:
        config = TrainerConfig()
        trainer = Trainer(config)
        trainer.train()
    """

    def __init__(self, config: Optional[TrainerConfig] = None) -> None:
        if config is None:
            config = TrainerConfig()
        self.config = config

        # Distributed training setup
        self._distributed = config.distributed
        self._local_rank = config.local_rank
        self._rank = 0
        self._world_size = 1

        if self._distributed:
            dist.init_process_group(backend="nccl")
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
            self._local_rank = int(os.environ.get("LOCAL_RANK", config.local_rank))
            torch.cuda.set_device(self._local_rank)
            config.device = f"cuda:{self._local_rank}"

        self._is_main = self._rank == 0

        # Setup directories (rank 0 only to avoid race conditions)
        if self._is_main:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
            Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            Path(config.export_dir).mkdir(parents=True, exist_ok=True)

        # Barrier to ensure directories are created before other ranks proceed
        if self._distributed:
            dist.barrier()

        # Setup logging
        self._setup_logging()

        # Set seed (offset per rank for data diversity)
        torch.manual_seed(config.seed + self._rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed + self._rank)

        # Device
        self.device = torch.device(config.device)
        if self._is_main:
            logger.info("Using device: %s", self.device)
            if self._distributed:
                logger.info(
                    "Distributed training: %d GPUs, rank %d",
                    self._world_size, self._rank,
                )

        # Model
        self.model = KairoController(config.model)
        self.model.to(self.device)

        # Wrap model in DDP if distributed
        self._raw_model = self.model  # Keep reference to unwrapped model
        if self._distributed:
            self.model = DDP(
                self.model,
                device_ids=[self._local_rank],
                output_device=self._local_rank,
            )

        n_params = self._raw_model.count_parameters()
        if self._is_main:
            logger.info("Model parameters: %d (%.2fM)", n_params, n_params / 1e6)

        # Data pipeline
        self.pipeline = TrainingPipeline(config.data)

        # Loss
        self.loss_fn = CombinedLoss(config.loss_weights)

        # Curriculum
        self.curriculum = CurriculumManager(config.curriculum)

        # Optimizer (created per phase)
        self.optimizer: Optional[torch.optim.AdamW] = None
        self.scaler = GradScaler(enabled=config.mixed_precision and self.device.type == "cuda")

        # Evaluator
        self.evaluator = Evaluator()

        # Tracking
        self._train_losses: List[float] = []
        self._best_loss = float("inf")

    def _setup_logging(self) -> None:
        """Configure file and console logging.

        In distributed mode, only rank 0 logs to console and file.
        Non-main ranks set log level to WARNING to reduce noise.
        """
        log_path = Path(self.config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        root = logging.getLogger()

        if not self._is_main:
            # Suppress verbose logging on non-main ranks
            root.setLevel(logging.WARNING)
            return

        root.setLevel(logging.INFO)

        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        root.addHandler(console)

        # File handler
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
        )
        root.addHandler(file_handler)

    def _create_optimizer(self) -> torch.optim.AdamW:
        """Create optimizer for the current curriculum phase.

        The initial learning rate is set to the warmup-appropriate value
        (i.e. near zero at step 0) rather than the full phase LR, so that
        the first training step uses the correct warmup LR even if
        _update_learning_rate is not called beforehand.
        """
        pc = self.curriculum.current_phase_config()

        # Compute the initial LR with warmup taken into account
        initial_lr = self.curriculum.get_learning_rate()

        # Weight decay: don't apply to biases or LayerNorm
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "h0" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": pc.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=initial_lr,
            betas=(0.9, 0.98),
            eps=1e-8,
        )
        logger.info(
            "Created optimizer for phase %d: initial_lr=%.2e (peak=%.2e), wd=%.4f, "
            "%d decay params, %d no-decay params",
            pc.phase,
            initial_lr,
            pc.learning_rate,
            pc.weight_decay,
            len(decay_params),
            len(no_decay_params),
        )
        return optimizer

    def _update_learning_rate(self) -> float:
        """Update the optimizer learning rate based on curriculum schedule."""
        lr = self.curriculum.get_learning_rate()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _process_single_step_batch(
        self,
        batch: TrainingBatch,
    ) -> Dict[str, torch.Tensor]:
        """
        Process a non-sequence batch (context or budget trace) through the model.

        Creates fresh recurrent state for each batch.

        Returns:
            Dict of model outputs.
        """
        B = batch.input_slots.size(0)
        state = self._raw_model.initial_state(B, self.device)

        candidates = batch.context_candidates
        mask = batch.candidate_mask

        outputs, _ = self.model(
            input_slots=batch.input_slots,
            state=state,
            context_candidates=candidates,
            candidate_mask=mask,
        )
        return outputs

    def _process_sequence_batch(
        self,
        batch: TrainingBatch,
    ) -> Dict[str, torch.Tensor]:
        """
        Process a sequence batch (action or session trace) through the model.

        Uses forward_sequence to unroll the recurrent state across time steps.

        Returns:
            Dict of model outputs with seq_len dimension.
        """
        # forward_sequence is on the raw model; DDP wraps the forward() call
        # inside it, so we call it on the unwrapped model.  The underlying
        # forward() calls still go through DDP when distributed.
        model = self._raw_model if not self._distributed else self.model.module
        outputs, _ = model.forward_sequence(
            input_sequence=batch.input_slots,
            context_candidates_seq=batch.context_candidates,
            candidate_mask_seq=batch.candidate_mask,
            itch_active_seq=batch.itch_active,
        )
        return outputs

    def _train_step(self, batch: TrainingBatch) -> Dict[str, float]:
        """
        Execute a single training step.

        Args:
            batch: The training batch (already on device).

        Returns:
            Dict of loss components for logging.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        pc = self.curriculum.current_phase_config()
        use_amp = self.config.mixed_precision and self.device.type == "cuda"

        with torch.autocast(device_type=self.device.type, enabled=use_amp):
            # Forward pass
            if batch.is_sequence:
                outputs = self._process_sequence_batch(batch)
            else:
                outputs = self._process_single_step_batch(batch)

            # Compute loss
            total_loss, components = self.loss_fn(outputs, batch)

        # Backward pass with gradient scaling
        self.scaler.scale(total_loss).backward()

        # Gradient clipping
        if pc.gradient_clip > 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), pc.gradient_clip
            )
            components["grad_norm"] = grad_norm.item()

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return components

    @torch.no_grad()
    def _eval_step(self, batch: TrainingBatch) -> None:
        """
        Execute a single evaluation step (no gradient).

        Args:
            batch: The evaluation batch (already on device).
        """
        self.model.eval()
        use_amp = self.config.mixed_precision and self.device.type == "cuda"

        with torch.autocast(device_type=self.device.type, enabled=use_amp):
            if batch.is_sequence:
                outputs = self._process_sequence_batch(batch)
            else:
                outputs = self._process_single_step_batch(batch)

        self.evaluator.update(outputs, batch)

    def _run_evaluation(self) -> Dict[str, float]:
        """
        Run evaluation over eval_batches from the pipeline.

        Returns:
            Dict of evaluation metrics.
        """
        self.evaluator.reset()
        phase = self.curriculum.current_phase

        eval_iter = self.pipeline.iterate(
            phase=phase, max_batches=self.config.eval_batches
        )

        for batch in eval_iter:
            batch = batch.to(self.device)
            self._eval_step(batch)

        metrics = self.evaluator.compute()

        # Log metrics
        logger.info("=== Evaluation at step %d ===", self.curriculum.global_step)
        for key, val in sorted(metrics.items()):
            logger.info("  %s: %.4f", key, val)

        return metrics

    def _save_checkpoint(self, tag: str = "latest") -> str:
        """
        Save a training checkpoint.

        In distributed mode, only rank 0 saves to avoid file corruption.

        Args:
            tag: Filename tag (e.g., "latest", "best", "phase1_complete").

        Returns:
            Path to the saved checkpoint.
        """
        path = os.path.join(self.config.checkpoint_dir, f"checkpoint_{tag}.pt")

        if not self._is_main:
            # Non-main ranks wait at the barrier below (if distributed)
            if self._distributed:
                dist.barrier()
            return path

        checkpoint = {
            "model_state_dict": self._raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "curriculum_state": self.curriculum.state_dict(),
            "config": {
                "model": {
                    "d_model": self.config.model.d_model,
                    "d_state": self.config.model.d_state,
                    "n_bands": self.config.model.n_bands,
                    "d_ffn": self.config.model.d_ffn,
                    "n_layers": self.config.model.n_layers,
                    "n_input_slots": self.config.model.n_input_slots,
                    "d_slot": self.config.model.d_slot,
                    "n_actions": self.config.model.n_actions,
                },
            },
            "train_losses": self._train_losses[-1000:],
        }

        torch.save(checkpoint, path)
        logger.info("Saved checkpoint: %s", path)

        # Barrier so all ranks wait for rank 0 to finish saving
        if self._distributed:
            dist.barrier()

        return path

    def _load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint and restore all state."""
        logger.info("Loading checkpoint: %s", path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self._raw_model.load_state_dict(checkpoint["model_state_dict"])
        self.curriculum.load_state_dict(checkpoint["curriculum_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if checkpoint.get("train_losses"):
            self._train_losses = checkpoint["train_losses"]

        # Recreate optimizer for the restored phase and load state
        if not self.curriculum.is_complete():
            self.optimizer = self._create_optimizer()
            if checkpoint.get("optimizer_state_dict") is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(
            "Restored from checkpoint. Phase=%d, Global step=%d",
            self.curriculum.phase_index + 1 if not self.curriculum.is_complete() else -1,
            self.curriculum.global_step,
        )

    def _train_phase(self) -> None:
        """
        Train a single curriculum phase until the gate condition is met
        or the max steps are reached.
        """
        pc = self.curriculum.current_phase_config()
        logger.info(
            "=== Starting Phase %d: %s ===\n  %s",
            pc.phase,
            pc.name,
            pc.description,
        )

        # Create optimizer for this phase
        self.optimizer = self._create_optimizer()

        # Training loop
        data_iter = self.pipeline.iterate(phase=pc.phase)
        step_start_time = time.time()

        for batch in data_iter:
            batch = batch.to(self.device)

            # Update learning rate
            lr = self._update_learning_rate()

            # Train step
            components = self._train_step(batch)
            self._train_losses.append(components.get("total", 0.0))

            # Record step in curriculum
            self.curriculum.step(components.get("total", 0.0))

            # Logging
            step = self.curriculum.global_step
            if step % self.config.log_every_steps == 0:
                elapsed = time.time() - step_start_time
                steps_per_sec = self.config.log_every_steps / max(elapsed, 1e-6)
                step_start_time = time.time()

                loss_str = " | ".join(
                    f"{k}={v:.4f}" for k, v in sorted(components.items())
                )
                logger.info(
                    "Phase %d | Step %d | lr=%.2e | %.1f steps/s | %s",
                    pc.phase,
                    step,
                    lr,
                    steps_per_sec,
                    loss_str,
                )

            # Checkpointing (all ranks participate for barrier sync)
            if step % self.config.checkpoint_every_steps == 0:
                self._save_checkpoint("latest")
                if self._is_main and components.get("total", float("inf")) < self._best_loss:
                    self._best_loss = components["total"]
                    self._save_checkpoint("best")

            # Evaluation and phase gating (rank 0 evaluates, broadcasts decision)
            if self.curriculum.should_evaluate():
                metrics = self._run_evaluation()
                should_advance = self.curriculum.evaluate(metrics)

                # Broadcast advance decision to all ranks
                if self._distributed:
                    advance_tensor = torch.tensor(
                        [1 if should_advance else 0],
                        dtype=torch.int64,
                        device=self.device,
                    )
                    dist.broadcast(advance_tensor, src=0)
                    should_advance = advance_tensor.item() == 1

                if should_advance:
                    self._save_checkpoint(f"phase{pc.phase}_complete")
                    break

            # Check max steps
            pm = self.curriculum.current_metrics()
            if pm.step_count >= pc.max_steps:
                logger.info("Phase %d reached max_steps (%d)", pc.phase, pc.max_steps)
                self._save_checkpoint(f"phase{pc.phase}_max_steps")
                break

    def train(self) -> None:
        """
        Run the full 3-phase training curriculum.

        This is the main entry point. It trains through all three phases,
        saving checkpoints and evaluating along the way, then exports
        the final model.

        In distributed mode, all ranks participate in training but only
        rank 0 saves checkpoints, logs metrics, and exports the model.
        """
        if self._is_main:
            logger.info("=" * 60)
            logger.info("KAIRO-X Controller Training")
            logger.info("=" * 60)
            logger.info("Model: %d parameters", self._raw_model.count_parameters())
            logger.info("Device: %s", self.device)
            logger.info("Mixed precision: %s", self.config.mixed_precision)
            if self._distributed:
                logger.info("Distributed: %d GPUs", self._world_size)
            logger.info("Output: %s", self.config.output_dir)
            logger.info("=" * 60)

        total_start = time.time()

        try:
            while not self.curriculum.is_complete():
                self._train_phase()
                if not self.curriculum.is_complete():
                    self.curriculum.advance_phase()

            total_time = time.time() - total_start
            if self._is_main:
                logger.info(
                    "Training complete! Total time: %.1f hours",
                    total_time / 3600,
                )

            # Export final model (rank 0 only)
            if self._is_main:
                self._export_final()
        finally:
            # Clean up distributed process group
            if self._distributed:
                dist.destroy_process_group()

    def _export_final(self) -> None:
        """Export the final trained model to binary format.

        Uses the unwrapped model so DDP wrapper state is not included.
        """
        export_path = os.path.join(self.config.export_dir, "kairo_controller.bin")
        logger.info("Exporting final model to %s", export_path)

        file_size = export_model(self._raw_model, export_path, self.config.model)
        logger.info("Export complete: %.2f MB", file_size / (1024 * 1024))

        # Verify
        if verify_export(self._raw_model, export_path):
            logger.info("Export verification passed.")
        else:
            logger.error("Export verification FAILED -- check binary output!")

    def resume(self, checkpoint_path: str) -> None:
        """
        Resume training from a checkpoint.

        In distributed mode, all ranks load the same checkpoint,
        then training proceeds with DDP synchronization.

        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        self._load_checkpoint(checkpoint_path)
        # Ensure all ranks have consistent model state after loading
        if self._distributed:
            dist.barrier()
        self.train()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the KAIRO-X controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file. Overrides all defaults.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (cuda, cpu).",
    )
    parser.add_argument(
        "--synthetic-size",
        type=int,
        default=None,
        help="Number of synthetic traces per dataset (for testing).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision training.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training with DistributedDataParallel (DDP). "
             "Launch with: torchrun --nproc_per_node=N -m training.trainer --distributed",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="Local rank for distributed training. Typically set by torchrun.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainerConfig:
    """Build TrainerConfig from CLI args and optional JSON config."""
    config = TrainerConfig()

    # Load JSON config if provided
    if args.config is not None:
        with open(args.config, "r") as f:
            json_config = json.load(f)
        # Apply JSON overrides (simplified -- production would use a proper merge)
        if "model" in json_config:
            for k, v in json_config["model"].items():
                if hasattr(config.model, k):
                    setattr(config.model, k, v)
        if "data" in json_config:
            for k, v in json_config["data"].items():
                if hasattr(config.data, k):
                    setattr(config.data, k, v)

    # Apply CLI overrides
    if args.output_dir is not None:
        config.output_dir = args.output_dir
        config.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        config.export_dir = os.path.join(args.output_dir, "export")
        config.log_file = os.path.join(args.output_dir, "training.log")

    if args.device is not None:
        config.device = args.device

    if args.synthetic_size is not None:
        config.data.synthetic_context_size = args.synthetic_size
        config.data.synthetic_budget_size = args.synthetic_size
        config.data.synthetic_action_size = args.synthetic_size
        config.data.synthetic_session_size = args.synthetic_size

    if args.batch_size is not None:
        config.data.batch_size = args.batch_size

    if args.no_amp:
        config.mixed_precision = False

    if args.distributed:
        config.distributed = True
        config.local_rank = args.local_rank

    return config


def main() -> None:
    """Main entry point for training."""
    args = parse_args()
    config = build_config(args)

    trainer = Trainer(config)

    if args.resume is not None:
        trainer.resume(args.resume)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
