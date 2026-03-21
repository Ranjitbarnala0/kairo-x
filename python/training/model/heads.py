"""
Output Heads (section 7.5)
===========================

The controller has 6 output heads, each producing a different kind of
decision signal. All heads receive the same d_model representation from
the final liquid block.

Heads:
    1. ActionHead:                34 discrete actions (cross-entropy)
    2. ContextSelectionHead:      score per candidate (binary cross-entropy)
    3. ContextBudgetHead:         token count in [512, 16384] (MSE)
    4. EnforcementIntensityHead:  float in [-0.3, +0.3] (MSE)
    5. SessionEdgeCaseHead:       continue / reset binary (binary cross-entropy)
    6. StopHead:                  continue/escalate/retry/terminate (cross-entropy,
                                  itch-gated with masking)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionHead(nn.Module):
    """
    Head 1: Action selection.

    Produces logits over 34 possible actions that the controller can take
    at each time step (e.g., read_file, write_file, run_test, call_llm, ...).

    Output: (batch, 34) logits -- trained with cross-entropy loss.
    """

    N_ACTIONS = 34

    def __init__(self, d_model: int = 288, d_hidden: int = 1152, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, self.N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, d_model)
        Returns:
            logits: (batch, 34) raw logits (no softmax).
        """
        return self.net(x)


class ContextSelectionHead(nn.Module):
    """
    Head 2: Context selection scoring.

    Given the controller state and a set of context candidates, produces a
    relevance score for each candidate. During inference, candidates above
    the threshold are included in the next LLM call.

    The head uses a bilinear-style scoring: it projects the controller state
    to a query vector, and each candidate embedding to a key vector, then
    computes dot-product scores.

    Output: (batch, n_candidates) scores -- trained with binary cross-entropy.
    """

    def __init__(
        self,
        d_model: int = 288,
        d_candidate: int = 64,
        d_hidden: int = 384,
        max_candidates: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_candidates = max_candidates

        # Project controller state to query space
        self.query_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
        )

        # Project candidate embeddings to key space
        self.key_proj = nn.Sequential(
            nn.Linear(d_candidate, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
        )

        # Temperature parameter for score scaling
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        x: torch.Tensor,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Controller state, shape (batch, d_model).
            candidates: Candidate embeddings, shape (batch, n_candidates, d_candidate).
            candidate_mask: Boolean mask, shape (batch, n_candidates).
                True = valid candidate, False = padding. If None, all candidates
                are treated as valid.

        Returns:
            scores: (batch, n_candidates) relevance scores (logits, no sigmoid).
        """
        query = self.query_proj(x)  # (batch, d_hidden)
        keys = self.key_proj(candidates)  # (batch, n_candidates, d_hidden)

        # Dot-product scoring
        scores = torch.einsum("bd,bnd->bn", query, keys)  # (batch, n_candidates)
        scores = scores / (self.temperature.abs() + 1e-6)

        # Mask out padding candidates with large negative value
        if candidate_mask is not None:
            scores = scores.masked_fill(~candidate_mask, -1e9)

        return scores


class ContextBudgetHead(nn.Module):
    """
    Head 3: Context budget prediction.

    Predicts the total token budget to allocate for the next LLM call.
    Output is clamped to [512, 16384] tokens.

    Output: (batch, 1) token count -- trained with MSE loss.
    """

    MIN_TOKENS = 512
    MAX_TOKENS = 16384

    def __init__(self, d_model: int = 288, d_hidden: int = 576, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, d_model)
        Returns:
            budget: (batch, 1) predicted token budget, clamped to valid range.
        """
        raw = self.net(x)  # (batch, 1)
        # Use sigmoid scaled to range, ensuring differentiable clamping
        budget = torch.sigmoid(raw) * (self.MAX_TOKENS - self.MIN_TOKENS) + self.MIN_TOKENS
        return budget


class EnforcementIntensityHead(nn.Module):
    """
    Head 4: Enforcement intensity adjustment.

    Predicts a float in [-0.3, +0.3] that modulates how strictly the system
    enforces compliance rules. Negative = more lenient, positive = stricter.

    Output: (batch, 1) intensity -- trained with MSE loss.
    """

    MIN_INTENSITY = -0.3
    MAX_INTENSITY = 0.3

    def __init__(self, d_model: int = 288, d_hidden: int = 384, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, d_model)
        Returns:
            intensity: (batch, 1) in [-0.3, +0.3].
        """
        raw = self.net(x)
        # tanh gives [-1, 1], scale to [-0.3, 0.3]
        return torch.tanh(raw) * self.MAX_INTENSITY


class SessionEdgeCaseHead(nn.Module):
    """
    Head 5: Session edge-case decision.

    Binary decision: continue the current session or reset.
    Used for detecting when a session has become unrecoverable.

    Output: (batch, 1) logit -- trained with binary cross-entropy.
    """

    def __init__(self, d_model: int = 288, d_hidden: int = 384, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, d_model)
        Returns:
            logit: (batch, 1) -- apply sigmoid for probability.
        """
        return self.net(x)


class StopHead(nn.Module):
    """
    Head 6: Stop decision (itch-gated).

    4-way classification: continue / escalate / retry / terminate.
    This head is gated by the itch state -- when there is no itch signal,
    the head should always predict "continue". During training, samples
    without an itch signal have their loss masked out (except for the
    "continue" class which should still be predicted correctly).

    Output: (batch, 4) logits -- trained with cross-entropy + itch masking.

    Class indices:
        0 = continue
        1 = escalate
        2 = retry
        3 = terminate
    """

    N_CLASSES = 4
    CONTINUE = 0
    ESCALATE = 1
    RETRY = 2
    TERMINATE = 3

    def __init__(self, d_model: int = 288, d_hidden: int = 576, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, self.N_CLASSES),
        )

    def forward(self, x: torch.Tensor, itch_active: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, d_model)
            itch_active: Optional boolean tensor (batch,). When False, the
                logits are overridden so that "continue" is the only valid
                prediction (used during inference; during training we use
                loss masking instead).

        Returns:
            logits: (batch, 4) raw logits.
        """
        logits = self.net(x)  # (batch, 4)

        if itch_active is not None:
            # Where itch is NOT active, force "continue" by setting a large
            # logit for class 0 and large negative for others
            no_itch = ~itch_active  # (batch,)
            if no_itch.any():
                override = torch.full_like(logits, -1e9)
                override[:, self.CONTINUE] = 1e9
                logits = torch.where(
                    no_itch.unsqueeze(-1).expand_as(logits),
                    override,
                    logits,
                )

        return logits
