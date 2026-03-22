"""
Loss Functions (section 17)
============================

Six loss functions corresponding to the six output heads, plus a combined
loss with curriculum-phase-dependent weighting.

Loss weights:
    context_selection:      1.0
    context_budget:         0.8
    action:                 0.6
    enforcement_intensity:  0.3
    stop:                   0.5
    session_edge_case:      0.3

The combined loss aggregates only the losses that have valid targets in the
current batch (which depends on the trace type and curriculum phase).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class LossWeights:
    """Weights for the 6 loss components."""

    context_selection: float = 1.0
    context_budget: float = 0.8
    action: float = 0.6
    enforcement_intensity: float = 0.3
    stop: float = 0.5
    session_edge_case: float = 0.3


class ContextSelectionLoss(nn.Module):
    """
    Binary cross-entropy loss for context selection.

    Computes BCE between predicted relevance scores and binary labels
    for each context candidate, masked to ignore padding candidates.
    """

    def __init__(self, label_smoothing: float = 0.05) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            scores: (batch, n_candidates) raw logits from context selection head.
            labels: (batch, n_candidates) binary ground truth.
            mask: (batch, n_candidates) boolean mask (True = valid candidate).

        Returns:
            Scalar loss.
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            labels = labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Compute BCE per element
        loss = F.binary_cross_entropy_with_logits(
            scores, labels, reduction="none"
        )  # (batch, n_candidates)

        # Mask and average
        mask_float = mask.float()
        masked_loss = loss * mask_float
        n_valid = mask_float.sum().clamp(min=1.0)
        return masked_loss.sum() / n_valid


class ContextBudgetLoss(nn.Module):
    """
    MSE loss for context budget prediction.

    Computes mean squared error between predicted and target token budgets.
    The loss is computed in log-space to handle the wide range [512, 16384]
    more stably.
    """

    def __init__(self, use_log_space: bool = True) -> None:
        super().__init__()
        self.use_log_space = use_log_space

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predicted: (batch, 1) predicted token budget.
            target: (batch, 1) ground truth token budget.

        Returns:
            Scalar loss.
        """
        if self.use_log_space:
            # Log-space MSE for better handling of the wide range.
            # Use softplus for a smooth lower bound instead of hard clamp
            # to avoid discontinuous gradients at the boundary.
            pred_log = torch.log(F.softplus(predicted - 1.0) + 1.0)
            target_log = torch.log(F.softplus(target - 1.0) + 1.0)
            return F.mse_loss(pred_log, target_log)
        else:
            # Normalize to [0, 1] range
            pred_norm = (predicted - 512.0) / (16384.0 - 512.0)
            target_norm = (target - 512.0) / (16384.0 - 512.0)
            return F.mse_loss(pred_norm, target_norm)


class ActionLoss(nn.Module):
    """
    Cross-entropy loss for action selection.

    Standard cross-entropy over 34 action classes with optional
    label smoothing.
    """

    def __init__(
        self,
        n_actions: int = 34,
        label_smoothing: float = 0.05,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
            reduction="mean",
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch, [seq_len], 34) raw logits.
            labels: (batch, [seq_len]) int64 class indices.
            seq_mask: Optional (batch, seq_len) boolean mask for valid steps.

        Returns:
            Scalar loss.
        """
        if logits.dim() == 3:
            # Sequence mode: flatten to (batch * seq_len, 34) and (batch * seq_len,)
            B, T, C = logits.shape
            flat_logits = logits.reshape(B * T, C)
            flat_labels = labels.reshape(B * T)

            if seq_mask is not None:
                # Set masked positions to ignore_index
                flat_mask = seq_mask.reshape(B * T)
                flat_labels = flat_labels.clone()
                flat_labels[~flat_mask] = -100

            return self.loss_fn(flat_logits, flat_labels)
        else:
            return self.loss_fn(logits, labels)


class EnforcementIntensityLoss(nn.Module):
    """
    MSE loss for enforcement intensity prediction.

    Target range is [-0.3, +0.3]. The loss is standard MSE.
    """

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            predicted: (batch, [seq_len], 1) predicted intensity.
            target: (batch, [seq_len], 1) ground truth intensity.
            seq_mask: Optional (batch, seq_len) boolean mask.

        Returns:
            Scalar loss.
        """
        loss = F.mse_loss(predicted, target, reduction="none")

        if seq_mask is not None:
            # Expand mask to match loss shape
            mask = seq_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            return (loss * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            return loss.mean()


class StopLoss(nn.Module):
    """
    Cross-entropy loss for the stop head with itch-gating.

    The stop head predicts 4 classes: continue/escalate/retry/terminate.
    When itch is not active, the loss is masked -- only samples where
    itch_active=True contribute to the gradient. This prevents the model
    from learning a trivial "always continue" solution.

    For steps where itch IS active, the loss uses all 4 classes normally.
    """

    def __init__(self, label_smoothing: float = 0.02) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=-100,
            reduction="mean",
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        itch_active: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch, [seq_len], 4) raw logits.
            labels: (batch, [seq_len]) int64 class indices.
            itch_active: (batch, [seq_len]) boolean -- True where itch is active.
            seq_mask: Optional (batch, seq_len) boolean mask for valid steps.

        Returns:
            Scalar loss. Returns 0 if no itch-active samples exist.
        """
        if logits.dim() == 3:
            B, T, C = logits.shape
            flat_logits = logits.reshape(B * T, C)
            flat_labels = labels.reshape(B * T).clone()
            flat_itch = itch_active.reshape(B * T)

            # Mask out non-itch positions
            flat_labels[~flat_itch] = -100

            # Also mask invalid sequence positions
            if seq_mask is not None:
                flat_mask = seq_mask.reshape(B * T)
                flat_labels[~flat_mask] = -100

            # Check if there are any valid samples
            if (flat_labels != -100).sum() == 0:
                # Return zero loss that maintains the computational graph
                # so gradients can still flow through the model parameters.
                return (logits * 0.0).sum()

            return self.loss_fn(flat_logits, flat_labels)
        else:
            masked_labels = labels.clone()
            masked_labels[~itch_active] = -100

            if (masked_labels != -100).sum() == 0:
                # Return zero loss that maintains the computational graph
                return (logits * 0.0).sum()

            return self.loss_fn(logits, masked_labels)


class SessionEdgeCaseLoss(nn.Module):
    """
    Binary cross-entropy loss for session edge-case decisions.

    Predicts whether to continue (0) or reset (1) the session.
    The loss handles class imbalance (resets are rare) with positive
    class weighting.
    """

    def __init__(self, pos_weight: float = 3.0) -> None:
        """
        Args:
            pos_weight: Weight for the positive (reset) class to handle
                class imbalance.
        """
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch, [seq_len], 1) raw logits.
            labels: (batch, [seq_len]) float 0/1.
            seq_mask: Optional (batch, seq_len) boolean mask.

        Returns:
            Scalar loss.
        """
        # Ensure labels match logits shape
        if logits.dim() == 3 and labels.dim() == 2:
            labels = labels.unsqueeze(-1)  # (batch, seq_len, 1)

        pos_weight = torch.tensor(
            [self.pos_weight], device=logits.device, dtype=logits.dtype
        )
        loss = F.binary_cross_entropy_with_logits(
            logits, labels.float(), pos_weight=pos_weight, reduction="none"
        )

        if seq_mask is not None:
            mask = seq_mask.unsqueeze(-1).float()
            return (loss * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined weighted loss across all 6 heads.

    The combined loss aggregates only the losses that have valid targets
    in the current batch. Weights are configurable and can be overridden
    per curriculum phase.
    """

    def __init__(self, weights: Optional[LossWeights] = None) -> None:
        super().__init__()
        if weights is None:
            weights = LossWeights()
        self.weights = weights

        self.context_selection_loss = ContextSelectionLoss()
        self.context_budget_loss = ContextBudgetLoss()
        self.action_loss = ActionLoss()
        self.enforcement_loss = EnforcementIntensityLoss()
        self.stop_loss = StopLoss()
        self.session_loss = SessionEdgeCaseLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: "TrainingBatch",
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the combined weighted loss.

        Args:
            outputs: Dict of model outputs (from KairoController.forward or
                forward_sequence).
            batch: The TrainingBatch with ground-truth labels.

        Returns:
            total_loss: Scalar loss for backpropagation.
            loss_components: Dict of individual loss values (detached) for
                logging.
        """
        # Import here to avoid circular imports
        from training.data_factory.pipeline import TrainingBatch

        total_loss = torch.zeros(1, device=outputs["action_logits"].device, requires_grad=True)
        components: Dict[str, float] = {}

        # --- Context selection loss ---
        if (
            batch.context_labels is not None
            and outputs["context_scores"].size(-1) > 0
        ):
            ctx_loss = self.context_selection_loss(
                outputs["context_scores"],
                batch.context_labels,
                batch.candidate_mask,
            )
            total_loss = total_loss + self.weights.context_selection * ctx_loss
            components["context_selection"] = ctx_loss.item()

        # --- Context budget loss ---
        if batch.context_budget_target is not None:
            budget_loss = self.context_budget_loss(
                outputs["context_budget"],
                batch.context_budget_target,
            )
            total_loss = total_loss + self.weights.context_budget * budget_loss
            components["context_budget"] = budget_loss.item()

        # --- Action loss ---
        if batch.action_labels is not None:
            seq_mask = None
            if batch.is_sequence:
                # Create per-sample sequence mask from seq_length tensor (B,)
                B, T = batch.action_labels.shape
                seq_len = batch.seq_length
                if not isinstance(seq_len, torch.Tensor):
                    seq_len = torch.tensor([seq_len] * B, device=batch.action_labels.device)
                seq_mask = torch.arange(T, device=batch.action_labels.device).unsqueeze(
                    0
                ).expand(B, -1) < seq_len.unsqueeze(1)

            act_loss = self.action_loss(
                outputs["action_logits"],
                batch.action_labels,
                seq_mask=seq_mask,
            )
            total_loss = total_loss + self.weights.action * act_loss
            components["action"] = act_loss.item()

        # --- Enforcement intensity loss ---
        if batch.enforcement_labels is not None:
            seq_mask = None
            if batch.is_sequence:
                B, T = batch.enforcement_labels.shape[:2]
                seq_len = batch.seq_length
                if not isinstance(seq_len, torch.Tensor):
                    seq_len = torch.tensor([seq_len] * B, device=batch.enforcement_labels.device)
                seq_mask = torch.arange(T, device=batch.enforcement_labels.device).unsqueeze(
                    0
                ).expand(B, -1) < seq_len.unsqueeze(1)

            enf_loss = self.enforcement_loss(
                outputs["enforcement_intensity"],
                batch.enforcement_labels,
                seq_mask=seq_mask,
            )
            total_loss = total_loss + self.weights.enforcement_intensity * enf_loss
            components["enforcement_intensity"] = enf_loss.item()

        # --- Stop loss ---
        if batch.stop_labels is not None and batch.itch_active is not None:
            seq_mask = None
            if batch.is_sequence:
                B, T = batch.stop_labels.shape
                seq_len = batch.seq_length
                if not isinstance(seq_len, torch.Tensor):
                    seq_len = torch.tensor([seq_len] * B, device=batch.stop_labels.device)
                seq_mask = torch.arange(T, device=batch.stop_labels.device).unsqueeze(
                    0
                ).expand(B, -1) < seq_len.unsqueeze(1)

            stp_loss = self.stop_loss(
                outputs["stop_logits"],
                batch.stop_labels,
                batch.itch_active,
                seq_mask=seq_mask,
            )
            total_loss = total_loss + self.weights.stop * stp_loss
            components["stop"] = stp_loss.item()

        # --- Session edge-case loss ---
        if batch.session_labels is not None:
            seq_mask = None
            if batch.is_sequence:
                B, T = batch.session_labels.shape
                seq_len = batch.seq_length
                if not isinstance(seq_len, torch.Tensor):
                    seq_len = torch.tensor([seq_len] * B, device=batch.session_labels.device)
                seq_mask = torch.arange(T, device=batch.session_labels.device).unsqueeze(
                    0
                ).expand(B, -1) < seq_len.unsqueeze(1)

            ses_loss = self.session_loss(
                outputs["session_edge_case"],
                batch.session_labels,
                seq_mask=seq_mask,
            )
            total_loss = total_loss + self.weights.session_edge_case * ses_loss
            components["session_edge_case"] = ses_loss.item()

        components["total"] = total_loss.item()
        return total_loss, components
