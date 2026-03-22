"""
Evaluation Metrics
===================

Computes evaluation metrics for all 6 heads to track training progress
and evaluate gating conditions.

Key metrics:
    - context_recall: fraction of relevant context items correctly selected (Phase 1 gate)
    - context_precision: fraction of selected items that are truly relevant
    - budget_mae: mean absolute error on budget prediction (tokens)
    - action_accuracy: top-1 accuracy on action selection
    - task_completion: estimated task completion rate (Phase 2 gate)
    - stop_accuracy: accuracy on stop decisions (itch-gated)
    - session_f1: F1 score on session reset detection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class EvalAccumulator:
    """Accumulates evaluation statistics across batches."""

    # Context selection
    context_true_positives: int = 0
    context_false_positives: int = 0
    context_false_negatives: int = 0
    context_true_negatives: int = 0

    # Context budget
    budget_abs_errors: List[float] = field(default_factory=list)
    budget_relative_errors: List[float] = field(default_factory=list)

    # Action
    action_correct: int = 0
    action_total: int = 0
    action_top3_correct: int = 0

    # Enforcement intensity
    enforcement_abs_errors: List[float] = field(default_factory=list)

    # Stop
    stop_correct: int = 0
    stop_total: int = 0
    stop_itch_correct: int = 0
    stop_itch_total: int = 0

    # Session
    session_true_positives: int = 0
    session_false_positives: int = 0
    session_false_negatives: int = 0
    session_true_negatives: int = 0

    # Task completion (estimated from action sequences)
    completed_sequences: int = 0
    total_sequences: int = 0

    def reset(self) -> None:
        """Reset all accumulators."""
        self.context_true_positives = 0
        self.context_false_positives = 0
        self.context_false_negatives = 0
        self.context_true_negatives = 0
        self.budget_abs_errors.clear()
        self.budget_relative_errors.clear()
        self.action_correct = 0
        self.action_total = 0
        self.action_top3_correct = 0
        self.enforcement_abs_errors.clear()
        self.stop_correct = 0
        self.stop_total = 0
        self.stop_itch_correct = 0
        self.stop_itch_total = 0
        self.session_true_positives = 0
        self.session_false_positives = 0
        self.session_false_negatives = 0
        self.session_true_negatives = 0
        self.completed_sequences = 0
        self.total_sequences = 0


class Evaluator:
    """
    Evaluates model outputs against ground truth across all heads.

    Usage:
        evaluator = Evaluator()
        evaluator.reset()

        for batch in eval_loader:
            outputs = model(batch)
            evaluator.update(outputs, batch)

        metrics = evaluator.compute()
        # metrics = {'context_recall': 0.85, 'action_accuracy': 0.72, ...}
    """

    def __init__(
        self,
        context_threshold: float = 0.5,
        task_completion_threshold: float = 0.6,
    ) -> None:
        """
        Args:
            context_threshold: Threshold for converting context scores to
                binary predictions.
            task_completion_threshold: Minimum per-sequence action accuracy
                required to count a sequence as "completed". Defaults to 0.6.
        """
        self.context_threshold = context_threshold
        self.task_completion_threshold = task_completion_threshold
        self.acc = EvalAccumulator()

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.acc.reset()

    @torch.no_grad()
    def update_context(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """
        Update context selection metrics.

        Args:
            scores: (batch, n_candidates) raw logits.
            labels: (batch, n_candidates) binary ground truth.
            mask: (batch, n_candidates) boolean mask.
        """
        probs = torch.sigmoid(scores)
        preds = (probs > self.context_threshold).float()

        # Only evaluate on valid (non-padding) candidates
        valid = mask.float()
        tp = ((preds == 1) & (labels == 1) & (valid == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0) & (valid == 1)).sum().item()
        fn = ((preds == 0) & (labels == 1) & (valid == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0) & (valid == 1)).sum().item()

        self.acc.context_true_positives += int(tp)
        self.acc.context_false_positives += int(fp)
        self.acc.context_false_negatives += int(fn)
        self.acc.context_true_negatives += int(tn)

    @torch.no_grad()
    def update_budget(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """
        Update context budget metrics.

        Args:
            predicted: (batch, 1) predicted budget.
            target: (batch, 1) target budget.
        """
        abs_err = (predicted - target).abs()
        rel_err = abs_err / target.clamp(min=1.0)

        self.acc.budget_abs_errors.extend(abs_err.flatten().cpu().tolist())
        self.acc.budget_relative_errors.extend(rel_err.flatten().cpu().tolist())

    @torch.no_grad()
    def update_action(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update action selection metrics.

        Args:
            logits: (batch, [seq_len], 34) raw logits.
            labels: (batch, [seq_len]) int64 labels.
            seq_mask: Optional (batch, seq_len) boolean mask.
        """
        if logits.dim() == 3:
            B, T, C = logits.shape
            flat_logits = logits.reshape(B * T, C)
            flat_labels = labels.reshape(B * T)
            if seq_mask is not None:
                flat_mask = seq_mask.reshape(B * T)
                flat_logits = flat_logits[flat_mask]
                flat_labels = flat_labels[flat_mask]
        else:
            flat_logits = logits
            flat_labels = labels

        if flat_logits.size(0) == 0:
            return

        # Top-1 accuracy
        preds = flat_logits.argmax(dim=-1)
        correct = (preds == flat_labels).sum().item()
        self.acc.action_correct += int(correct)
        self.acc.action_total += flat_labels.size(0)

        # Top-3 accuracy
        if flat_logits.size(-1) >= 3:
            top3 = flat_logits.topk(3, dim=-1).indices
            top3_correct = (top3 == flat_labels.unsqueeze(-1)).any(dim=-1).sum().item()
            self.acc.action_top3_correct += int(top3_correct)

    @torch.no_grad()
    def update_enforcement(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update enforcement intensity metrics.

        Args:
            predicted: (batch, [seq_len], 1) predicted intensity.
            target: (batch, [seq_len], 1) target intensity.
            seq_mask: Optional mask.
        """
        if seq_mask is not None:
            mask = seq_mask.unsqueeze(-1)
            abs_err = ((predicted - target).abs() * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
        else:
            abs_err = (predicted - target).abs().mean()

        self.acc.enforcement_abs_errors.append(abs_err.item())

    @torch.no_grad()
    def update_stop(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        itch_active: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update stop decision metrics.

        Args:
            logits: (batch, [seq_len], 4) raw logits.
            labels: (batch, [seq_len]) int64 labels.
            itch_active: (batch, [seq_len]) boolean.
            seq_mask: Optional mask.
        """
        if logits.dim() == 3:
            B, T, C = logits.shape
            flat_logits = logits.reshape(B * T, C)
            flat_labels = labels.reshape(B * T)
            flat_itch = itch_active.reshape(B * T)
            if seq_mask is not None:
                flat_mask = seq_mask.reshape(B * T)
                flat_logits = flat_logits[flat_mask]
                flat_labels = flat_labels[flat_mask]
                flat_itch = flat_itch[flat_mask]
        else:
            flat_logits = logits
            flat_labels = labels
            flat_itch = itch_active

        if flat_logits.size(0) == 0:
            return

        preds = flat_logits.argmax(dim=-1)

        # Overall accuracy
        correct = (preds == flat_labels).sum().item()
        self.acc.stop_correct += int(correct)
        self.acc.stop_total += flat_labels.size(0)

        # Itch-gated accuracy (only when itch is active)
        if flat_itch.any():
            itch_preds = preds[flat_itch]
            itch_labels = flat_labels[flat_itch]
            itch_correct = (itch_preds == itch_labels).sum().item()
            self.acc.stop_itch_correct += int(itch_correct)
            self.acc.stop_itch_total += itch_labels.size(0)

    @torch.no_grad()
    def update_session(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update session edge-case metrics.

        Args:
            logits: (batch, [seq_len], 1) raw logits.
            labels: (batch, [seq_len]) float 0/1.
            seq_mask: Optional mask.
        """
        if logits.dim() == 3:
            flat_logits = logits.squeeze(-1)  # (batch, seq_len)
            if seq_mask is not None:
                flat_logits = flat_logits[seq_mask]
                flat_labels = labels[seq_mask]
            else:
                flat_logits = flat_logits.reshape(-1)
                flat_labels = labels.reshape(-1)
        else:
            flat_logits = logits.squeeze(-1)
            flat_labels = labels

        if flat_logits.numel() == 0:
            return

        preds = (torch.sigmoid(flat_logits) > 0.5).float()
        flat_labels = flat_labels.float()

        tp = ((preds == 1) & (flat_labels == 1)).sum().item()
        fp = ((preds == 1) & (flat_labels == 0)).sum().item()
        fn = ((preds == 0) & (flat_labels == 1)).sum().item()
        tn = ((preds == 0) & (flat_labels == 0)).sum().item()

        self.acc.session_true_positives += int(tp)
        self.acc.session_false_positives += int(fp)
        self.acc.session_false_negatives += int(fn)
        self.acc.session_true_negatives += int(tn)

    @torch.no_grad()
    def update_task_completion(
        self,
        action_logits: torch.Tensor,
        action_labels: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Estimate task completion from action sequence accuracy.

        A sequence is considered "completed" if the model correctly predicts
        at least ``self.task_completion_threshold`` fraction of the actions
        in the sequence.

        Args:
            action_logits: (batch, seq_len, 34)
            action_labels: (batch, seq_len)
            seq_mask: Optional (batch, seq_len) boolean.
        """
        B, T, C = action_logits.shape
        preds = action_logits.argmax(dim=-1)  # (B, T)
        correct = (preds == action_labels)  # (B, T)

        if seq_mask is not None:
            correct = correct & seq_mask

        for b in range(B):
            if seq_mask is not None:
                valid = seq_mask[b].sum().item()
            else:
                valid = T

            if valid == 0:
                continue

            acc = correct[b].sum().item() / valid
            self.acc.total_sequences += 1
            if acc >= self.task_completion_threshold:
                self.acc.completed_sequences += 1

    def update(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: "TrainingBatch",
    ) -> None:
        """
        Update all relevant metrics from a single batch.

        Args:
            outputs: Model outputs dict.
            batch: The TrainingBatch with labels.
        """
        from training.data_factory.pipeline import TrainingBatch

        # Build per-sample seq_mask from the seq_length tensor
        seq_mask = None
        if batch.is_sequence and batch.action_labels is not None:
            B, T = batch.action_labels.shape
            seq_len = batch.seq_length
            if not isinstance(seq_len, torch.Tensor):
                seq_len = torch.tensor([seq_len] * B, device=batch.action_labels.device)
            seq_mask = torch.arange(T, device=batch.action_labels.device).unsqueeze(
                0
            ).expand(B, -1) < seq_len.unsqueeze(1)

        # Context selection
        if (
            batch.context_labels is not None
            and outputs["context_scores"].size(-1) > 0
        ):
            ctx_mask = batch.candidate_mask
            if ctx_mask.dim() == 3:
                # Sequence: use first step's mask (usually constant)
                ctx_mask = ctx_mask[:, 0]
            ctx_labels = batch.context_labels
            if ctx_labels.dim() == 3:
                ctx_labels = ctx_labels[:, 0]
            ctx_scores = outputs["context_scores"]
            if ctx_scores.dim() == 3:
                ctx_scores = ctx_scores[:, 0]
            self.update_context(ctx_scores, ctx_labels, ctx_mask)

        # Context budget
        if batch.context_budget_target is not None:
            budget_pred = outputs["context_budget"]
            budget_target = batch.context_budget_target
            if budget_pred.dim() == 3:
                budget_pred = budget_pred.mean(dim=1)
            self.update_budget(budget_pred, budget_target)

        # Action
        if batch.action_labels is not None:
            self.update_action(
                outputs["action_logits"], batch.action_labels, seq_mask
            )
            if batch.is_sequence:
                self.update_task_completion(
                    outputs["action_logits"], batch.action_labels, seq_mask
                )

        # Enforcement
        if batch.enforcement_labels is not None:
            self.update_enforcement(
                outputs["enforcement_intensity"],
                batch.enforcement_labels,
                seq_mask,
            )

        # Stop
        if batch.stop_labels is not None and batch.itch_active is not None:
            self.update_stop(
                outputs["stop_logits"],
                batch.stop_labels,
                batch.itch_active,
                seq_mask,
            )

        # Session
        if batch.session_labels is not None:
            ses_mask = None
            if batch.is_sequence and batch.session_labels is not None:
                B, T = batch.session_labels.shape
                seq_len = batch.seq_length
                if not isinstance(seq_len, torch.Tensor):
                    seq_len = torch.tensor([seq_len] * B, device=batch.session_labels.device)
                ses_mask = torch.arange(
                    T, device=batch.session_labels.device
                ).unsqueeze(0).expand(B, -1) < seq_len.unsqueeze(1)
            self.update_session(
                outputs["session_edge_case"], batch.session_labels, ses_mask
            )

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated statistics.

        Returns:
            Dict of metric name -> value.
        """
        metrics: Dict[str, float] = {}

        # --- Context selection ---
        tp = self.acc.context_true_positives
        fp = self.acc.context_false_positives
        fn = self.acc.context_false_negatives
        tn = self.acc.context_true_negatives

        metrics["context_recall"] = tp / max(1, tp + fn)
        metrics["context_precision"] = tp / max(1, tp + fp)
        if metrics["context_precision"] + metrics["context_recall"] > 0:
            metrics["context_f1"] = (
                2
                * metrics["context_precision"]
                * metrics["context_recall"]
                / (metrics["context_precision"] + metrics["context_recall"])
            )
        else:
            metrics["context_f1"] = 0.0
        total_ctx = tp + fp + fn + tn
        metrics["context_accuracy"] = (tp + tn) / max(1, total_ctx)

        # --- Budget ---
        if self.acc.budget_abs_errors:
            metrics["budget_mae"] = sum(self.acc.budget_abs_errors) / len(
                self.acc.budget_abs_errors
            )
            metrics["budget_mre"] = sum(self.acc.budget_relative_errors) / len(
                self.acc.budget_relative_errors
            )
        else:
            metrics["budget_mae"] = 0.0
            metrics["budget_mre"] = 0.0

        # --- Action ---
        metrics["action_accuracy"] = self.acc.action_correct / max(
            1, self.acc.action_total
        )
        metrics["action_top3_accuracy"] = self.acc.action_top3_correct / max(
            1, self.acc.action_total
        )

        # --- Enforcement ---
        if self.acc.enforcement_abs_errors:
            metrics["enforcement_mae"] = sum(
                self.acc.enforcement_abs_errors
            ) / len(self.acc.enforcement_abs_errors)
        else:
            metrics["enforcement_mae"] = 0.0

        # --- Stop ---
        metrics["stop_accuracy"] = self.acc.stop_correct / max(
            1, self.acc.stop_total
        )
        metrics["stop_itch_accuracy"] = self.acc.stop_itch_correct / max(
            1, self.acc.stop_itch_total
        )

        # --- Session ---
        s_tp = self.acc.session_true_positives
        s_fp = self.acc.session_false_positives
        s_fn = self.acc.session_false_negatives
        s_tn = self.acc.session_true_negatives

        session_precision = s_tp / max(1, s_tp + s_fp)
        session_recall = s_tp / max(1, s_tp + s_fn)
        if session_precision + session_recall > 0:
            metrics["session_f1"] = (
                2 * session_precision * session_recall
                / (session_precision + session_recall)
            )
        else:
            metrics["session_f1"] = 0.0
        metrics["session_precision"] = session_precision
        metrics["session_recall"] = session_recall

        # --- Task completion ---
        metrics["task_completion"] = self.acc.completed_sequences / max(
            1, self.acc.total_sequences
        )

        return metrics
