"""
Context Budget Training Data (PRIMARY -- Phase 1)
===================================================

Generates and loads traces for training the context budget head.

Each trace captures a decision point where the controller must allocate a
token budget for the next LLM call. The ground truth comes from analyzing
actual calls: the ideal budget is the minimum token count that includes all
necessary context without wasting tokens on irrelevant material.

Budget traces share the same input slot format as context traces and are
often derived from the same underlying session recordings, but focus on the
budget regression target rather than candidate selection.

Budget range: [512, 16384] tokens
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
import numpy as np


@dataclass
class BudgetTrace:
    """A single context budget training example."""

    input_slots: torch.Tensor           # (n_slots, d_slot)
    context_candidates: torch.Tensor    # (n_candidates, d_candidate)
    candidate_mask: torch.Tensor        # (n_candidates,) bool
    selected_candidates: torch.Tensor   # (n_candidates,) float 0/1 -- which were selected
    target_budget: torch.Tensor         # (1,) float in [512, 16384]
    actual_tokens_used: torch.Tensor    # (1,) float -- tokens actually consumed
    budget_efficiency: torch.Tensor     # (1,) float -- actual_used / budget (0-1)

    def to(self, device: torch.device) -> "BudgetTrace":
        """Move all tensors to device."""
        return BudgetTrace(
            input_slots=self.input_slots.to(device),
            context_candidates=self.context_candidates.to(device),
            candidate_mask=self.candidate_mask.to(device),
            selected_candidates=self.selected_candidates.to(device),
            target_budget=self.target_budget.to(device),
            actual_tokens_used=self.actual_tokens_used.to(device),
            budget_efficiency=self.budget_efficiency.to(device),
        )


class BudgetTraceDataset(Dataset):
    """
    Dataset of context budget traces.

    Supports loading from .npz files and synthetic generation.
    Budget traces model the relationship between:
        - Current task state (input slots)
        - Which context was selected (selected_candidates)
        - How many tokens should be allocated (target_budget)

    The target budget is the "ideal" allocation -- tight enough to avoid waste
    but generous enough to include all necessary context.
    """

    MIN_BUDGET = 512
    MAX_BUDGET = 16384

    def __init__(
        self,
        data_dir: Optional[str] = None,
        n_slots: int = 32,
        d_slot: int = 64,
        max_candidates: int = 32,
        d_candidate: int = 64,
        synthetic_size: int = 0,
        seed: int = 123,
    ) -> None:
        """
        Args:
            data_dir: Path to directory containing budget trace files.
            n_slots: Number of input slots.
            d_slot: Dimension per slot.
            max_candidates: Maximum context candidates.
            d_candidate: Dimension per candidate.
            synthetic_size: Number of synthetic traces to generate.
            seed: Random seed for synthetic generation.
        """
        self.n_slots = n_slots
        self.d_slot = d_slot
        self.max_candidates = max_candidates
        self.d_candidate = d_candidate

        self.traces: List[Dict[str, np.ndarray]] = []

        if data_dir is not None and os.path.isdir(data_dir):
            self._load_from_disk(data_dir)

        if synthetic_size > 0:
            self._generate_synthetic(synthetic_size, seed)

    def _load_from_disk(self, data_dir: str) -> None:
        """Load budget traces from .npz files."""
        data_path = Path(data_dir)
        for npz_file in sorted(data_path.glob("budget_*.npz")):
            data = np.load(npz_file)
            self.traces.append(
                {
                    "input_slots": data["input_slots"],
                    "context_candidates": data["context_candidates"],
                    "candidate_mask": data["candidate_mask"],
                    "selected_candidates": data["selected_candidates"],
                    "target_budget": data["target_budget"],
                    "actual_tokens_used": data["actual_tokens_used"],
                    "budget_efficiency": data["budget_efficiency"],
                }
            )

    def _generate_synthetic(self, n_traces: int, seed: int) -> None:
        """
        Generate synthetic budget traces.

        Models a realistic relationship between the number of selected context
        items and the token budget needed. Each selected candidate contributes
        a variable number of tokens (simulating files of different sizes).
        """
        rng = np.random.RandomState(seed)

        # Typical token counts per context type
        token_distributions = {
            "small_file": (200, 800),     # Small utility files
            "medium_file": (800, 2500),   # Typical source files
            "large_file": (2500, 6000),   # Large modules
            "snippet": (50, 200),         # Code snippets
        }

        for _ in range(n_traces):
            n_real = rng.randint(3, self.max_candidates + 1)
            n_selected = rng.randint(1, max(2, n_real // 2 + 1))

            input_slots = rng.randn(self.n_slots, self.d_slot).astype(np.float32) * 0.3

            candidates = np.zeros(
                (self.max_candidates, self.d_candidate), dtype=np.float32
            )
            mask = np.zeros(self.max_candidates, dtype=np.float32)
            selected = np.zeros(self.max_candidates, dtype=np.float32)

            # --- Slots 10-13: recent_llm_responses ---
            # One-hot response classifications correlated with budget needs
            n_response_classes = 8
            history_depth = rng.randint(1, 5)
            for s in range(10, 14):
                input_slots[s, :n_response_classes] = 0.0
                if s - 10 < history_depth:
                    cls = rng.randint(0, n_response_classes)
                    input_slots[s, cls] = 1.0
                    decay = 1.0 - (s - 10) * 0.2
                    input_slots[s, :n_response_classes] *= decay

            # --- Slots 14-16: project_state ---
            task_complexity = n_selected / max(1, n_real)
            n_modified_files = rng.randint(0, 20)
            input_slots[14, 0] = task_complexity
            input_slots[14, 1] = n_modified_files / 20.0
            build_pass = 1.0 if rng.rand() < 0.7 else 0.0
            input_slots[15, 0] = build_pass
            input_slots[15, 1] = 1.0 - build_pass
            input_slots[16, 0] = 1.0 if rng.rand() < 0.8 else 0.0  # Lint pass
            input_slots[16, 1] = 1.0 if rng.rand() < 0.75 else 0.0  # Typecheck pass

            # --- Slots 22-24: verification_state ---
            l1_pass = 1.0 if rng.rand() < 0.6 else 0.0
            l2_pass = 1.0 if (l1_pass > 0 and rng.rand() < 0.7) else 0.0
            input_slots[22, 0] = l1_pass
            input_slots[22, 1] = 1.0 - l1_pass
            input_slots[23, 0] = l2_pass
            input_slots[23, 1] = 1.0 - l2_pass
            # Failed verification => more context needed => higher budget
            input_slots[24, 0] = rng.randint(0, 8) / 8.0

            # --- Slots 25-26: itch_state ---
            # High itch count correlates with higher budget need
            pending_itch = rng.randint(0, 6)
            failed_itch = rng.randint(0, pending_itch + 1)
            input_slots[25, 0] = pending_itch / 6.0
            input_slots[26, 0] = failed_itch / 6.0
            input_slots[26, 1] = (failed_itch / max(1, pending_itch))

            # --- Slots 27-28: session_state ---
            turn_count = rng.randint(1, 50)
            token_spend_so_far = rng.uniform(500, 32000)
            input_slots[27, 0] = turn_count / 50.0
            input_slots[28, 0] = token_spend_so_far / 32000.0

            # --- Slot 29: cost_state ---
            cost_pressure = rng.uniform(0.0, 1.0)
            input_slots[29, 0] = cost_pressure
            cost_modes = 4
            cost_mode = rng.randint(0, cost_modes)
            input_slots[29, 1:1 + cost_modes] = 0.0
            input_slots[29, 1 + cost_mode] = 1.0

            # --- Slot 30: user_state ---
            steps_since_user = rng.randint(0, 30)
            input_slots[30, 0] = np.exp(-steps_since_user / 5.0)
            input_slots[30, 1] = steps_since_user / 30.0

            total_tokens = 0.0
            selected_indices = rng.choice(n_real, size=n_selected, replace=False)

            for i in range(n_real):
                mask[i] = 1.0
                candidates[i] = rng.randn(self.d_candidate).astype(np.float32) * 0.5

                if i in selected_indices:
                    selected[i] = 1.0
                    # Each selected candidate has a token cost
                    ctx_type = rng.choice(list(token_distributions.keys()))
                    lo, hi = token_distributions[ctx_type]
                    tokens = rng.uniform(lo, hi)
                    total_tokens += tokens

                    # Encode size hint in candidate embedding
                    candidates[i, 0] = tokens / 3000.0  # Normalized size

            # Target budget: actual tokens + some headroom (10-30%)
            headroom = rng.uniform(1.1, 1.3)
            target_budget = np.clip(
                total_tokens * headroom,
                self.MIN_BUDGET,
                self.MAX_BUDGET,
            )

            # Under cost pressure, budgets get tighter
            if cost_pressure > 0.7:
                target_budget *= rng.uniform(0.8, 0.95)
                target_budget = max(target_budget, self.MIN_BUDGET)

            # High itch count also inflates budget slightly (need more context
            # to resolve pending items)
            if pending_itch >= 3:
                target_budget *= rng.uniform(1.05, 1.15)
                target_budget = min(target_budget, self.MAX_BUDGET)

            actual_used = min(total_tokens, target_budget)
            efficiency = actual_used / target_budget if target_budget > 0 else 0.0

            self.traces.append(
                {
                    "input_slots": input_slots,
                    "context_candidates": candidates,
                    "candidate_mask": mask,
                    "selected_candidates": selected,
                    "target_budget": np.array([target_budget], dtype=np.float32),
                    "actual_tokens_used": np.array([actual_used], dtype=np.float32),
                    "budget_efficiency": np.array([efficiency], dtype=np.float32),
                }
            )

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int) -> BudgetTrace:
        trace = self.traces[idx]
        return BudgetTrace(
            input_slots=torch.from_numpy(trace["input_slots"]),
            context_candidates=torch.from_numpy(trace["context_candidates"]),
            candidate_mask=torch.from_numpy(trace["candidate_mask"]).bool(),
            selected_candidates=torch.from_numpy(trace["selected_candidates"]),
            target_budget=torch.from_numpy(trace["target_budget"]),
            actual_tokens_used=torch.from_numpy(trace["actual_tokens_used"]),
            budget_efficiency=torch.from_numpy(trace["budget_efficiency"]),
        )


def collate_budget_traces(batch: List[BudgetTrace]) -> BudgetTrace:
    """Collate a list of BudgetTrace into a batched BudgetTrace."""
    return BudgetTrace(
        input_slots=torch.stack([t.input_slots for t in batch]),
        context_candidates=torch.stack([t.context_candidates for t in batch]),
        candidate_mask=torch.stack([t.candidate_mask for t in batch]),
        selected_candidates=torch.stack([t.selected_candidates for t in batch]),
        target_budget=torch.stack([t.target_budget for t in batch]),
        actual_tokens_used=torch.stack([t.actual_tokens_used for t in batch]),
        budget_efficiency=torch.stack([t.budget_efficiency for t in batch]),
    )
