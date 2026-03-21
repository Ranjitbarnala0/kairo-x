"""
Data Loading and Batching Pipeline
====================================

Assembles the various trace datasets into a unified training pipeline with
curriculum-aware sampling. The pipeline handles:

1. Creating DataLoaders for each trace type
2. Curriculum-phase sampling weights (Phase 1 focuses on context traces,
   Phase 2 adds action traces, Phase 3 adds cost-mode traces)
3. Producing a unified TrainingBatch that the training loop consumes
4. Data augmentation (noise injection, candidate shuffling)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, ConcatDataset

from training.data_factory.context_traces import (
    ContextTraceDataset,
    ContextTrace,
    collate_context_traces,
)
from training.data_factory.budget_traces import (
    BudgetTraceDataset,
    BudgetTrace,
    collate_budget_traces,
)
from training.data_factory.action_traces import (
    ActionTraceDataset,
    ActionSequenceTrace,
    collate_action_sequences,
)
from training.data_factory.session_traces import (
    SessionTraceDataset,
    SessionTrace,
    collate_session_traces,
)

logger = logging.getLogger(__name__)


class CurriculumPhase(IntEnum):
    """Training curriculum phases."""
    PHASE_1 = 1  # Context selection + budget (primary)
    PHASE_2 = 2  # Full execution traces
    PHASE_3 = 3  # End-to-end with cost modes


@dataclass
class TrainingBatch:
    """
    Unified training batch consumed by the training loop.

    Not all fields are populated in every batch -- the active fields depend
    on the curriculum phase and the trace type sampled.
    """

    # Common fields (always present)
    input_slots: torch.Tensor               # (batch, [seq_len], n_slots, d_slot)
    context_candidates: torch.Tensor        # (batch, [seq_len], n_candidates, d_candidate)
    candidate_mask: torch.Tensor            # (batch, [seq_len], n_candidates) bool

    # Context selection labels (Phase 1+)
    context_labels: Optional[torch.Tensor] = None       # (batch, n_candidates) float
    context_budget_target: Optional[torch.Tensor] = None  # (batch, 1) float

    # Action labels (Phase 2+)
    action_labels: Optional[torch.Tensor] = None         # (batch, [seq_len]) int64
    enforcement_labels: Optional[torch.Tensor] = None    # (batch, [seq_len], 1) float
    stop_labels: Optional[torch.Tensor] = None           # (batch, [seq_len]) int64
    itch_active: Optional[torch.Tensor] = None           # (batch, [seq_len]) bool

    # Session labels (Phase 2+)
    session_labels: Optional[torch.Tensor] = None        # (batch, [seq_len]) float

    # Metadata
    trace_type: str = "context"   # "context", "budget", "action", "session"
    is_sequence: bool = False     # Whether the batch has a seq_len dimension
    seq_length: int = 1           # Actual sequence length (before padding)
    phase: CurriculumPhase = CurriculumPhase.PHASE_1

    def to(self, device: torch.device) -> "TrainingBatch":
        """Move all tensors to the given device."""
        def _move(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return t.to(device) if t is not None else None

        return TrainingBatch(
            input_slots=self.input_slots.to(device),
            context_candidates=self.context_candidates.to(device),
            candidate_mask=self.candidate_mask.to(device),
            context_labels=_move(self.context_labels),
            context_budget_target=_move(self.context_budget_target),
            action_labels=_move(self.action_labels),
            enforcement_labels=_move(self.enforcement_labels),
            stop_labels=_move(self.stop_labels),
            itch_active=_move(self.itch_active),
            session_labels=_move(self.session_labels),
            trace_type=self.trace_type,
            is_sequence=self.is_sequence,
            seq_length=self.seq_length,
            phase=self.phase,
        )


@dataclass
class PipelineConfig:
    """Configuration for the training data pipeline."""

    # Data directories
    context_data_dir: Optional[str] = None
    budget_data_dir: Optional[str] = None
    action_data_dir: Optional[str] = None
    session_data_dir: Optional[str] = None

    # Synthetic data sizes (used if no real data is available)
    synthetic_context_size: int = 10000
    synthetic_budget_size: int = 10000
    synthetic_action_size: int = 5000
    synthetic_session_size: int = 3000

    # Dimensions
    n_slots: int = 32
    d_slot: int = 64
    max_candidates: int = 32
    d_candidate: int = 64
    max_seq_len: int = 64

    # Batching
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True

    # Phase-dependent sampling weights
    # [context_weight, budget_weight, action_weight, session_weight]
    phase1_weights: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.0, 0.0])
    phase2_weights: List[float] = field(default_factory=lambda: [0.2, 0.2, 0.4, 0.2])
    phase3_weights: List[float] = field(default_factory=lambda: [0.15, 0.15, 0.4, 0.3])

    # Augmentation
    noise_scale: float = 0.01
    candidate_shuffle_prob: float = 0.3
    slot_dropout_prob: float = 0.1


class TrainingPipeline:
    """
    Unified data pipeline that creates and manages DataLoaders for all
    trace types and handles curriculum-aware sampling.

    Usage:
        config = PipelineConfig(synthetic_context_size=10000)
        pipeline = TrainingPipeline(config)

        for batch in pipeline.iterate(phase=CurriculumPhase.PHASE_1):
            # batch is a TrainingBatch
            loss = compute_loss(model, batch)
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        if config is None:
            config = PipelineConfig()
        self.config = config

        logger.info("Initializing training data pipeline...")

        # Create datasets
        self.context_dataset = ContextTraceDataset(
            data_dir=config.context_data_dir,
            n_slots=config.n_slots,
            d_slot=config.d_slot,
            max_candidates=config.max_candidates,
            d_candidate=config.d_candidate,
            synthetic_size=config.synthetic_context_size,
        )
        logger.info(f"Context dataset: {len(self.context_dataset)} traces")

        self.budget_dataset = BudgetTraceDataset(
            data_dir=config.budget_data_dir,
            n_slots=config.n_slots,
            d_slot=config.d_slot,
            max_candidates=config.max_candidates,
            d_candidate=config.d_candidate,
            synthetic_size=config.synthetic_budget_size,
        )
        logger.info(f"Budget dataset: {len(self.budget_dataset)} traces")

        self.action_dataset = ActionTraceDataset(
            data_dir=config.action_data_dir,
            n_slots=config.n_slots,
            d_slot=config.d_slot,
            max_candidates=config.max_candidates,
            d_candidate=config.d_candidate,
            max_seq_len=config.max_seq_len,
            synthetic_size=config.synthetic_action_size,
        )
        logger.info(f"Action dataset: {len(self.action_dataset)} traces")

        self.session_dataset = SessionTraceDataset(
            data_dir=config.session_data_dir,
            n_slots=config.n_slots,
            d_slot=config.d_slot,
            max_candidates=config.max_candidates,
            d_candidate=config.d_candidate,
            max_seq_len=config.max_seq_len,
            synthetic_size=config.synthetic_session_size,
        )
        logger.info(f"Session dataset: {len(self.session_dataset)} traces")

        # Create DataLoaders
        self._context_loader = DataLoader(
            self.context_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=collate_context_traces,
            drop_last=True,
        )
        self._budget_loader = DataLoader(
            self.budget_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=collate_budget_traces,
            drop_last=True,
        )
        self._action_loader = DataLoader(
            self.action_dataset,
            batch_size=max(1, config.batch_size // 4),  # Smaller batch for sequences
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=collate_action_sequences,
            drop_last=True,
        )
        self._session_loader = DataLoader(
            self.session_dataset,
            batch_size=max(1, config.batch_size // 4),
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=collate_session_traces,
            drop_last=True,
        )

    def _weights_for_phase(self, phase: CurriculumPhase) -> List[float]:
        """Get sampling weights for the given phase."""
        if phase == CurriculumPhase.PHASE_1:
            return self.config.phase1_weights
        elif phase == CurriculumPhase.PHASE_2:
            return self.config.phase2_weights
        else:
            return self.config.phase3_weights

    def _context_to_batch(
        self, trace: ContextTrace, phase: CurriculumPhase
    ) -> TrainingBatch:
        """Convert a ContextTrace batch to a TrainingBatch."""
        return TrainingBatch(
            input_slots=trace.input_slots,
            context_candidates=trace.context_candidates,
            candidate_mask=trace.candidate_mask,
            context_labels=trace.context_labels,
            context_budget_target=trace.context_budget_tokens,
            trace_type="context",
            is_sequence=False,
            seq_length=1,
            phase=phase,
        )

    def _budget_to_batch(
        self, trace: BudgetTrace, phase: CurriculumPhase
    ) -> TrainingBatch:
        """Convert a BudgetTrace batch to a TrainingBatch."""
        return TrainingBatch(
            input_slots=trace.input_slots,
            context_candidates=trace.context_candidates,
            candidate_mask=trace.candidate_mask,
            context_labels=trace.selected_candidates,
            context_budget_target=trace.target_budget,
            trace_type="budget",
            is_sequence=False,
            seq_length=1,
            phase=phase,
        )

    def _action_to_batch(
        self, trace: ActionSequenceTrace, phase: CurriculumPhase
    ) -> TrainingBatch:
        """Convert an ActionSequenceTrace batch to a TrainingBatch."""
        return TrainingBatch(
            input_slots=trace.input_slots,
            context_candidates=trace.context_candidates,
            candidate_mask=trace.candidate_mask,
            action_labels=trace.action_labels,
            enforcement_labels=trace.enforcement_labels,
            stop_labels=trace.stop_labels,
            itch_active=trace.itch_active,
            trace_type="action",
            is_sequence=True,
            seq_length=trace.seq_length,
            phase=phase,
        )

    def _session_to_batch(
        self, trace: SessionTrace, phase: CurriculumPhase
    ) -> TrainingBatch:
        """Convert a SessionTrace batch to a TrainingBatch."""
        return TrainingBatch(
            input_slots=trace.input_slots,
            context_candidates=trace.context_candidates,
            candidate_mask=trace.candidate_mask,
            session_labels=trace.session_labels,
            trace_type="session",
            is_sequence=True,
            seq_length=trace.seq_length,
            phase=phase,
        )

    def _augment_batch(self, batch: TrainingBatch) -> TrainingBatch:
        """Apply data augmentation to a batch.

        Augmentations applied (during training):
            1. Gaussian noise: adds small noise to input slots.
            2. Candidate shuffling: randomly permutes the order of context
               candidates so the model cannot rely on positional ordering.
            3. Slot dropout: randomly zeros out a fraction of input slots
               to improve robustness and regularise the model.
        """
        # 1. Gaussian noise on input slots
        noise_scale = self.config.noise_scale
        if noise_scale > 0:
            noise = torch.randn_like(batch.input_slots) * noise_scale
            batch.input_slots = batch.input_slots + noise

        # 2. Candidate shuffling: randomly permute context candidates
        if self.config.candidate_shuffle_prob > 0 and torch.rand(1).item() < self.config.candidate_shuffle_prob:
            n_candidates = batch.context_candidates.size(-2)
            if n_candidates > 1:
                perm = torch.randperm(n_candidates)
                if batch.context_candidates.dim() == 3:
                    # Non-sequence: (batch, n_candidates, d_candidate)
                    batch.context_candidates = batch.context_candidates[:, perm, :]
                    batch.candidate_mask = batch.candidate_mask[:, perm]
                    if batch.context_labels is not None:
                        batch.context_labels = batch.context_labels[:, perm]
                elif batch.context_candidates.dim() == 4:
                    # Sequence: (batch, seq_len, n_candidates, d_candidate)
                    batch.context_candidates = batch.context_candidates[:, :, perm, :]
                    batch.candidate_mask = batch.candidate_mask[:, :, perm]
                    if batch.context_labels is not None and batch.context_labels.dim() == 3:
                        batch.context_labels = batch.context_labels[:, :, perm]

        # 3. Slot dropout: randomly zero out a fraction of input slots
        if self.config.slot_dropout_prob > 0:
            # Determine the slot dimension position
            if batch.input_slots.dim() == 3:
                # Non-sequence: (batch, n_slots, d_slot)
                n_slots = batch.input_slots.size(1)
                slot_mask = (torch.rand(batch.input_slots.size(0), n_slots, 1) > self.config.slot_dropout_prob).float()
                batch.input_slots = batch.input_slots * slot_mask.to(batch.input_slots.device)
            elif batch.input_slots.dim() == 4:
                # Sequence: (batch, seq_len, n_slots, d_slot)
                n_slots = batch.input_slots.size(2)
                slot_mask = (torch.rand(batch.input_slots.size(0), 1, n_slots, 1) > self.config.slot_dropout_prob).float()
                batch.input_slots = batch.input_slots * slot_mask.to(batch.input_slots.device)

        return batch

    def iterate(
        self,
        phase: CurriculumPhase = CurriculumPhase.PHASE_1,
        max_batches: Optional[int] = None,
    ) -> Iterator[TrainingBatch]:
        """
        Yield TrainingBatch instances according to the curriculum phase.

        This generator cycles through the DataLoaders, sampling trace types
        according to the phase weights. It runs until max_batches is reached
        or forever if max_batches is None.

        Args:
            phase: Current curriculum phase.
            max_batches: Stop after this many batches. None = infinite.

        Yields:
            TrainingBatch instances.
        """
        weights = self._weights_for_phase(phase)
        loaders = [
            ("context", self._context_loader, self._context_to_batch),
            ("budget", self._budget_loader, self._budget_to_batch),
            ("action", self._action_loader, self._action_to_batch),
            ("session", self._session_loader, self._session_to_batch),
        ]

        # Filter out zero-weight loaders
        active_loaders = [
            (name, loader, converter, w)
            for (name, loader, converter), w in zip(loaders, weights)
            if w > 0
        ]

        if not active_loaders:
            logger.warning("No active data loaders for phase %s", phase)
            return

        # Normalize weights
        total_w = sum(w for _, _, _, w in active_loaders)
        probs = [w / total_w for _, _, _, w in active_loaders]

        # Create iterators that auto-reset
        iterators: Dict[str, Iterator] = {}
        for name, loader, _, _ in active_loaders:
            iterators[name] = iter(loader)

        batch_count = 0
        while max_batches is None or batch_count < max_batches:
            # Sample a trace type
            idx = torch.multinomial(
                torch.tensor(probs, dtype=torch.float32), 1
            ).item()
            name, loader, converter, _ = active_loaders[idx]

            # Get next batch, resetting iterator if exhausted
            try:
                raw_batch = next(iterators[name])
            except StopIteration:
                iterators[name] = iter(loader)
                try:
                    raw_batch = next(iterators[name])
                except StopIteration:
                    logger.warning("Loader %s is empty, skipping", name)
                    continue

            batch = converter(raw_batch, phase)
            batch = self._augment_batch(batch)
            yield batch
            batch_count += 1

    def steps_per_epoch(self, phase: CurriculumPhase) -> int:
        """Estimate number of batches per epoch for the given phase."""
        weights = self._weights_for_phase(phase)
        total = 0
        loaders = [
            self._context_loader,
            self._budget_loader,
            self._action_loader,
            self._session_loader,
        ]
        for loader, w in zip(loaders, weights):
            if w > 0:
                total += len(loader)
        return max(1, total)
