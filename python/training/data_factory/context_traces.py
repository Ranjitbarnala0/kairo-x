"""
Context Selection Training Data (PRIMARY -- Phase 1)
=====================================================

Generates and loads traces for training the context selection head.

Each trace represents a time step where the controller must decide which
context items to include in an LLM call. The ground truth is derived from
actual agentic coding sessions where we know which files/symbols were
actually needed for successful task completion.

Trace format:
    - input_slots: (n_slots, d_slot) the assembled 32-slot input packet
    - context_candidates: (n_candidates, d_candidate) available context items
    - candidate_mask: (n_candidates,) boolean mask for valid candidates
    - context_labels: (n_candidates,) binary labels -- 1 if selected, 0 if not
    - context_budget_tokens: (1,) actual token count used

Data sources:
    - Recorded agentic coding sessions
    - Synthetic traces from codebase snapshots
    - Augmented traces with random context shuffling
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np


@dataclass
class ContextTrace:
    """A single context selection training example."""

    input_slots: torch.Tensor          # (n_slots, d_slot)
    context_candidates: torch.Tensor   # (n_candidates, d_candidate)
    candidate_mask: torch.Tensor       # (n_candidates,) bool
    context_labels: torch.Tensor       # (n_candidates,) float 0/1
    context_budget_tokens: torch.Tensor  # (1,) float

    def to(self, device: torch.device) -> "ContextTrace":
        """Move all tensors to device."""
        return ContextTrace(
            input_slots=self.input_slots.to(device),
            context_candidates=self.context_candidates.to(device),
            candidate_mask=self.candidate_mask.to(device),
            context_labels=self.context_labels.to(device),
            context_budget_tokens=self.context_budget_tokens.to(device),
        )


class ContextTraceDataset(Dataset):
    """
    Dataset of context selection traces loaded from disk.

    File format:
        Each trace file is a directory containing:
            - meta.json: metadata (n_candidates, d_slot, etc.)
            - input_slots.bin: raw f32 array (n_slots * d_slot)
            - candidates.bin: raw f32 array (n_candidates * d_candidate)
            - labels.bin: raw f32 array (n_candidates)
            - budget.bin: raw f32 (1 value)

        Alternatively, a single .npz file per trace.

    The dataset also supports in-memory synthetic generation for bootstrapping
    training before real traces are available.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        n_slots: int = 32,
        d_slot: int = 64,
        max_candidates: int = 32,
        d_candidate: int = 64,
        synthetic_size: int = 0,
        seed: int = 42,
    ) -> None:
        """
        Args:
            data_dir: Path to directory containing trace files. If None and
                synthetic_size > 0, only synthetic data is used.
            n_slots: Number of input slots.
            d_slot: Dimension of each input slot.
            max_candidates: Maximum number of context candidates.
            d_candidate: Dimension of each candidate embedding.
            synthetic_size: Number of synthetic traces to generate if no
                data_dir is provided or to supplement real data.
            seed: Random seed for synthetic generation.
        """
        self.n_slots = n_slots
        self.d_slot = d_slot
        self.max_candidates = max_candidates
        self.d_candidate = d_candidate

        self.traces: List[Dict[str, np.ndarray]] = []

        # Load real traces from disk
        if data_dir is not None and os.path.isdir(data_dir):
            self._load_from_disk(data_dir)

        # Generate synthetic traces
        if synthetic_size > 0:
            self._generate_synthetic(synthetic_size, seed)

    def _load_from_disk(self, data_dir: str) -> None:
        """Load traces from .npz files in the data directory."""
        data_path = Path(data_dir)
        npz_files = sorted(data_path.glob("*.npz"))

        for npz_file in npz_files:
            data = np.load(npz_file)
            trace = {
                "input_slots": data["input_slots"],
                "context_candidates": data["context_candidates"],
                "candidate_mask": data["candidate_mask"],
                "context_labels": data["context_labels"],
                "context_budget_tokens": data["context_budget_tokens"],
            }
            self.traces.append(trace)

        # Also look for binary trace directories
        for subdir in sorted(data_path.iterdir()):
            if subdir.is_dir() and (subdir / "meta.json").exists():
                trace = self._load_binary_trace(subdir)
                if trace is not None:
                    self.traces.append(trace)

    def _load_binary_trace(self, trace_dir: Path) -> Optional[Dict[str, np.ndarray]]:
        """Load a single trace from binary files."""
        meta_path = trace_dir / "meta.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)

        n_candidates = meta.get("n_candidates", self.max_candidates)

        def read_f32(name: str, shape: Tuple[int, ...]) -> np.ndarray:
            path = trace_dir / name
            raw = path.read_bytes()
            arr = np.frombuffer(raw, dtype=np.float32)
            return arr.reshape(shape)

        try:
            return {
                "input_slots": read_f32(
                    "input_slots.bin", (self.n_slots, self.d_slot)
                ),
                "context_candidates": read_f32(
                    "candidates.bin", (n_candidates, self.d_candidate)
                ),
                "candidate_mask": np.ones(n_candidates, dtype=np.float32),
                "context_labels": read_f32("labels.bin", (n_candidates,)),
                "context_budget_tokens": read_f32("budget.bin", (1,)),
            }
        except (FileNotFoundError, ValueError):
            return None

    def _generate_synthetic(self, n_traces: int, seed: int) -> None:
        """
        Generate synthetic context selection traces.

        Synthetic traces model a simple scenario: some context candidates are
        "relevant" to the current task state (their embeddings correlate with
        the input), and the labels reflect that relevance. This provides a
        reasonable training signal for Phase 1 bootstrapping.
        """
        rng = np.random.RandomState(seed)

        for _ in range(n_traces):
            # Random number of actual candidates (rest are padding)
            n_real = rng.randint(3, self.max_candidates + 1)

            # Generate a "task direction" vector that determines relevance
            task_direction = rng.randn(self.d_candidate).astype(np.float32)
            task_direction /= np.linalg.norm(task_direction) + 1e-8

            # Input slots: embed task info across multiple slot ranges.
            # Vary which slot groups carry task-correlated features so the
            # model cannot overfit to a single slot range.
            input_slots = rng.randn(self.n_slots, self.d_slot).astype(np.float32) * 0.3

            # Always inject into context-related slots (17-22)
            for s in range(17, 22):
                input_slots[s, : self.d_candidate] += task_direction * 0.5

            # Randomly also inject correlated features into other slot groups
            # to better match the distribution of real traces.
            # Execution graph summary slots (0-5)
            if rng.rand() < 0.5:
                scale = 0.2 + rng.rand() * 0.3
                for s in range(0, 6):
                    input_slots[s, : self.d_candidate] += task_direction * scale

            # Active node context slots (6-9)
            if rng.rand() < 0.4:
                scale = 0.15 + rng.rand() * 0.35
                for s in range(6, 10):
                    input_slots[s, : self.d_candidate] += task_direction * scale

            # --- Slots 10-13: recent_llm_responses ---
            # One-hot response classifications with random history depth.
            # Categories: Plan=0, Implementation=1, VerificationPass=2,
            #             VerificationFail=3, Debug=4, Review=5, Edit=6, Other=7
            n_response_classes = 8
            history_depth = rng.randint(1, 5)  # 1-4 recent responses
            for s in range(10, 14):
                # Each slot holds one-hot for a historical response
                input_slots[s, :n_response_classes] = 0.0
                if s - 10 < history_depth:
                    cls = rng.randint(0, n_response_classes)
                    input_slots[s, cls] = 1.0
                    # Decay older responses
                    decay = 1.0 - (s - 10) * 0.2
                    input_slots[s, :n_response_classes] *= decay
                    # Correlate implementation/edit responses with task direction
                    if cls in (1, 6):
                        input_slots[s, n_response_classes: self.d_candidate] += (
                            task_direction[n_response_classes:] * 0.3
                        )

            # --- Slots 14-16: project_state ---
            # File modification counts, build pass/fail flags
            n_modified_files = rng.randint(0, 20)
            input_slots[14, 0] = n_modified_files / 20.0  # Normalized file mod count
            input_slots[14, 1] = rng.rand()               # Fraction of files touched
            build_pass = 1.0 if rng.rand() < 0.7 else 0.0
            input_slots[15, 0] = build_pass               # Build pass flag
            input_slots[15, 1] = 1.0 - build_pass         # Build fail flag
            input_slots[15, 2] = rng.rand()               # Build confidence
            # Lint / typecheck status
            input_slots[16, 0] = 1.0 if rng.rand() < 0.8 else 0.0  # Lint pass
            input_slots[16, 1] = 1.0 if rng.rand() < 0.75 else 0.0  # Typecheck pass
            input_slots[16, 2] = rng.randint(0, 10) / 10.0  # Warning count norm

            # --- Slots 22-24: verification_state ---
            # L1/L2 pass/fail patterns
            l1_pass = 1.0 if rng.rand() < 0.6 else 0.0
            l2_pass = 1.0 if (l1_pass > 0 and rng.rand() < 0.7) else 0.0
            input_slots[22, 0] = l1_pass
            input_slots[22, 1] = 1.0 - l1_pass  # L1 fail
            input_slots[23, 0] = l2_pass
            input_slots[23, 1] = 1.0 - l2_pass  # L2 fail
            # Verification attempt count and recency
            input_slots[24, 0] = rng.randint(0, 8) / 8.0  # Attempt count normalized
            input_slots[24, 1] = rng.rand()  # Recency of last verification

            # --- Slots 25-26: itch_state ---
            pending_count = rng.randint(0, 6)
            failed_count = rng.randint(0, pending_count + 1)
            input_slots[25, 0] = pending_count / 6.0  # Normalized pending count
            input_slots[25, 1] = rng.rand()  # Urgency score
            input_slots[26, 0] = failed_count / 6.0   # Normalized failed count
            input_slots[26, 1] = (failed_count / max(1, pending_count))  # Fail ratio

            # --- Slots 27-28: session_state ---
            turn_count = rng.randint(1, 50)
            token_spend = rng.uniform(500, 32000)
            input_slots[27, 0] = turn_count / 50.0       # Normalized turn count
            input_slots[27, 1] = rng.rand()               # Session progress ratio
            input_slots[28, 0] = token_spend / 32000.0    # Normalized token spend
            input_slots[28, 1] = rng.rand()               # Remaining capacity

            # --- Slot 29: cost_state ---
            budget_remaining = rng.uniform(0.0, 1.0)
            cost_modes = 4  # e.g., normal=0, frugal=1, generous=2, critical=3
            cost_mode = rng.randint(0, cost_modes)
            input_slots[29, 0] = budget_remaining  # Budget remaining fraction
            input_slots[29, 1:1 + cost_modes] = 0.0
            input_slots[29, 1 + cost_mode] = 1.0  # Cost mode one-hot

            # --- Slot 30: user_state ---
            # User message recency as exponentially decaying signal
            steps_since_user = rng.randint(0, 30)
            recency_signal = np.exp(-steps_since_user / 5.0)
            input_slots[30, 0] = recency_signal  # Decaying recency
            input_slots[30, 1] = steps_since_user / 30.0  # Normalized steps

            # Candidate embeddings: some correlated with task, some not
            candidates = np.zeros(
                (self.max_candidates, self.d_candidate), dtype=np.float32
            )
            labels = np.zeros(self.max_candidates, dtype=np.float32)
            mask = np.zeros(self.max_candidates, dtype=np.float32)

            # Decide relevance: ~30-50% of real candidates are relevant
            n_relevant = rng.randint(1, max(2, int(n_real * 0.5) + 1))
            relevant_indices = rng.choice(n_real, size=n_relevant, replace=False)

            for i in range(n_real):
                mask[i] = 1.0
                if i in relevant_indices:
                    # Relevant candidate: correlated with task direction
                    candidates[i] = (
                        task_direction * (0.5 + rng.rand() * 0.5)
                        + rng.randn(self.d_candidate).astype(np.float32) * 0.2
                    )
                    labels[i] = 1.0
                else:
                    # Irrelevant candidate: random direction
                    candidates[i] = rng.randn(self.d_candidate).astype(np.float32) * 0.5

            # Budget: proportional to number of relevant candidates
            # Each relevant candidate uses ~1000-3000 tokens
            budget = float(n_relevant * rng.randint(1000, 3000) + 512)
            budget = np.clip(budget, 512, 16384)

            self.traces.append(
                {
                    "input_slots": input_slots,
                    "context_candidates": candidates,
                    "candidate_mask": mask,
                    "context_labels": labels,
                    "context_budget_tokens": np.array([budget], dtype=np.float32),
                }
            )

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int) -> ContextTrace:
        trace = self.traces[idx]
        return ContextTrace(
            input_slots=torch.from_numpy(trace["input_slots"]),
            context_candidates=torch.from_numpy(trace["context_candidates"]),
            candidate_mask=torch.from_numpy(trace["candidate_mask"]).bool(),
            context_labels=torch.from_numpy(trace["context_labels"]),
            context_budget_tokens=torch.from_numpy(trace["context_budget_tokens"]),
        )


def collate_context_traces(batch: List[ContextTrace]) -> ContextTrace:
    """Collate a list of ContextTrace into a batched ContextTrace."""
    return ContextTrace(
        input_slots=torch.stack([t.input_slots for t in batch]),
        context_candidates=torch.stack([t.context_candidates for t in batch]),
        candidate_mask=torch.stack([t.candidate_mask for t in batch]),
        context_labels=torch.stack([t.context_labels for t in batch]),
        context_budget_tokens=torch.stack([t.context_budget_tokens for t in batch]),
    )
