"""
Session Decision Training Traces (Phase 2+)
=============================================

Generates and loads traces for training the session edge-case head.

Session traces capture moments where the controller must decide whether to
continue or reset a session. These are relatively rare events (most steps
just continue), so the dataset is enriched with edge cases:
    - Token budget exhaustion
    - Repeated failures / error loops
    - Context window overflow
    - Stale session detection
    - User inactivity timeout approach

Each trace provides a sequence of steps leading up to a session decision
point, plus the ground-truth decision (continue=0, reset=1).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
import numpy as np


@dataclass
class SessionTrace:
    """A single session decision training example."""

    input_slots: torch.Tensor           # (seq_len, n_slots, d_slot)
    context_candidates: torch.Tensor    # (seq_len, n_candidates, d_candidate)
    candidate_mask: torch.Tensor        # (seq_len, n_candidates) bool
    session_labels: torch.Tensor        # (seq_len,) float 0/1
    # Additional context for the session head
    tokens_spent_ratio: torch.Tensor    # (seq_len, 1) cumulative tokens / budget
    error_count: torch.Tensor           # (seq_len, 1) rolling error count
    steps_since_progress: torch.Tensor  # (seq_len, 1) steps since meaningful progress
    seq_length: Any                     # int for single trace, (batch,) int64 tensor after collation

    def to(self, device: torch.device) -> "SessionTrace":
        """Move all tensors to device."""
        sl = self.seq_length
        if isinstance(sl, torch.Tensor):
            sl = sl.to(device)
        return SessionTrace(
            input_slots=self.input_slots.to(device),
            context_candidates=self.context_candidates.to(device),
            candidate_mask=self.candidate_mask.to(device),
            session_labels=self.session_labels.to(device),
            tokens_spent_ratio=self.tokens_spent_ratio.to(device),
            error_count=self.error_count.to(device),
            steps_since_progress=self.steps_since_progress.to(device),
            seq_length=sl,
        )


class SessionTraceDataset(Dataset):
    """
    Dataset of session decision traces.

    The dataset is heavily enriched with edge cases since session reset
    events are rare in normal operation. Approximately 40% of synthetic
    traces contain at least one reset event.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        n_slots: int = 32,
        d_slot: int = 64,
        max_candidates: int = 32,
        d_candidate: int = 64,
        max_seq_len: int = 64,
        synthetic_size: int = 0,
        seed: int = 789,
    ) -> None:
        self.n_slots = n_slots
        self.d_slot = d_slot
        self.max_candidates = max_candidates
        self.d_candidate = d_candidate
        self.max_seq_len = max_seq_len

        self.traces: List[Dict[str, np.ndarray]] = []

        if data_dir is not None and os.path.isdir(data_dir):
            self._load_from_disk(data_dir)

        if synthetic_size > 0:
            self._generate_synthetic(synthetic_size, seed)

    def _load_from_disk(self, data_dir: str) -> None:
        """Load session traces from .npz files."""
        data_path = Path(data_dir)
        for npz_file in sorted(data_path.glob("session_*.npz")):
            data = np.load(npz_file)
            self.traces.append(
                {
                    "input_slots": data["input_slots"],
                    "context_candidates": data["context_candidates"],
                    "candidate_mask": data["candidate_mask"],
                    "session_labels": data["session_labels"],
                    "tokens_spent_ratio": data["tokens_spent_ratio"],
                    "error_count": data["error_count"],
                    "steps_since_progress": data["steps_since_progress"],
                    "seq_length": int(data["seq_length"]),
                }
            )

    def _generate_synthetic(self, n_traces: int, seed: int) -> None:
        """
        Generate synthetic session traces with enriched edge cases.

        Simulates four scenarios:
            1. Normal session (continues throughout) -- 60%
            2. Budget exhaustion (reset near end) -- 15%
            3. Error loop (reset after repeated failures) -- 15%
            4. Stale session (reset after no progress) -- 10%
        """
        rng = np.random.RandomState(seed)

        scenario_probs = [0.60, 0.15, 0.15, 0.10]
        scenarios = ["normal", "budget_exhaustion", "error_loop", "stale"]

        for _ in range(n_traces):
            scenario = rng.choice(scenarios, p=scenario_probs)
            seq_len = rng.randint(10, self.max_seq_len + 1)

            input_slots = rng.randn(
                self.max_seq_len, self.n_slots, self.d_slot
            ).astype(np.float32) * 0.3
            candidates = rng.randn(
                self.max_seq_len, self.max_candidates, self.d_candidate
            ).astype(np.float32) * 0.5
            n_real = rng.randint(3, self.max_candidates + 1)
            mask = np.zeros(
                (self.max_seq_len, self.max_candidates), dtype=np.float32
            )
            mask[:, :n_real] = 1.0

            session_labels = np.zeros(self.max_seq_len, dtype=np.float32)
            tokens_ratio = np.zeros((self.max_seq_len, 1), dtype=np.float32)
            error_count = np.zeros((self.max_seq_len, 1), dtype=np.float32)
            steps_no_progress = np.zeros((self.max_seq_len, 1), dtype=np.float32)

            if scenario == "normal":
                # Token ratio grows slowly, errors stay low, progress is steady
                for t in range(seq_len):
                    tokens_ratio[t, 0] = (t + 1) / (seq_len * 2)  # Never exceeds 0.5
                    error_count[t, 0] = rng.poisson(0.3)
                    steps_no_progress[t, 0] = rng.randint(0, 3)
                    # Embed session health in session_state slots (27-28)
                    input_slots[t, 27, 0] = tokens_ratio[t, 0]
                    input_slots[t, 28, 0] = 0.9  # Healthy

            elif scenario == "budget_exhaustion":
                # Token ratio grows to near 1.0, triggering reset
                reset_step = rng.randint(max(1, seq_len - 5), seq_len)
                for t in range(seq_len):
                    tokens_ratio[t, 0] = min(1.0, (t + 1) / (reset_step + 1))
                    error_count[t, 0] = rng.poisson(0.5)
                    steps_no_progress[t, 0] = rng.randint(0, 4)
                    input_slots[t, 27, 0] = tokens_ratio[t, 0]
                    input_slots[t, 28, 0] = max(0.0, 1.0 - tokens_ratio[t, 0])
                    if t >= reset_step:
                        session_labels[t] = 1.0  # Reset

            elif scenario == "error_loop":
                # Error count spikes, triggering reset
                error_start = rng.randint(seq_len // 2, max(seq_len // 2 + 1, seq_len - 3))
                for t in range(seq_len):
                    tokens_ratio[t, 0] = (t + 1) / (seq_len * 1.5)
                    if t >= error_start:
                        error_count[t, 0] = rng.randint(3, 8)
                        steps_no_progress[t, 0] = t - error_start + rng.randint(2, 5)
                        if error_count[t, 0] >= 5:
                            session_labels[t] = 1.0
                    else:
                        error_count[t, 0] = rng.poisson(0.3)
                        steps_no_progress[t, 0] = rng.randint(0, 2)
                    input_slots[t, 27, 0] = tokens_ratio[t, 0]
                    input_slots[t, 28, 0] = max(0.0, 1.0 - error_count[t, 0] / 8.0)

            elif scenario == "stale":
                # No progress for many steps, triggering reset
                stale_start = rng.randint(seq_len // 3, max(seq_len // 3 + 1, seq_len - 5))
                for t in range(seq_len):
                    tokens_ratio[t, 0] = (t + 1) / (seq_len * 1.5)
                    error_count[t, 0] = rng.poisson(0.5)
                    if t >= stale_start:
                        steps_no_progress[t, 0] = t - stale_start
                        if steps_no_progress[t, 0] >= 5:
                            session_labels[t] = 1.0
                    else:
                        steps_no_progress[t, 0] = rng.randint(0, 2)
                    input_slots[t, 27, 0] = tokens_ratio[t, 0]
                    input_slots[t, 28, 0] = max(
                        0.0, 1.0 - steps_no_progress[t, 0] / 10.0
                    )

            # --- Populate all slot ranges with structured features ---
            n_response_classes = 8
            pending_itch = rng.randint(0, 6)
            failed_itch = rng.randint(0, pending_itch + 1)
            cost_mode = rng.randint(0, 4)
            steps_since_user = rng.randint(0, 30)

            for t in range(seq_len):
                # Slots 10-13: recent_llm_responses
                for s in range(10, 14):
                    input_slots[t, s, :n_response_classes] = 0.0
                    if s - 10 < min(4, t + 1):
                        cls = rng.randint(0, n_response_classes)
                        input_slots[t, s, cls] = 1.0 - (s - 10) * 0.2

                # Slots 14-16: project_state
                file_mods = min(20, t + rng.randint(0, 3))
                input_slots[t, 14, 0] = file_mods / 20.0
                input_slots[t, 14, 1] = rng.rand()
                build_pass = 1.0 if rng.rand() < 0.7 else 0.0
                input_slots[t, 15, 0] = build_pass
                input_slots[t, 15, 1] = 1.0 - build_pass
                input_slots[t, 16, 0] = 1.0 if rng.rand() < 0.8 else 0.0
                input_slots[t, 16, 1] = 1.0 if rng.rand() < 0.75 else 0.0

                # Slots 22-24: verification_state (correlate with error_count)
                l1_fail = 1.0 if error_count[t, 0] >= 3 else 0.0
                l2_fail = 1.0 if error_count[t, 0] >= 5 else 0.0
                input_slots[t, 22, 0] = 1.0 - l1_fail       # L1 pass
                input_slots[t, 22, 1] = l1_fail               # L1 fail
                input_slots[t, 23, 0] = 1.0 - l2_fail        # L2 pass
                input_slots[t, 23, 1] = l2_fail               # L2 fail
                input_slots[t, 24, 0] = error_count[t, 0] / 10.0  # attempt count

                # Slots 25-26: itch_state (correlate with stale/error scenarios)
                cur_pending = min(6, pending_itch + (1 if error_count[t, 0] > 2 else 0))
                cur_failed = min(cur_pending, failed_itch + (1 if error_count[t, 0] > 4 else 0))
                input_slots[t, 25, 0] = cur_pending / 6.0
                input_slots[t, 25, 1] = rng.rand()
                input_slots[t, 26, 0] = cur_failed / 6.0
                input_slots[t, 26, 1] = cur_failed / max(1, cur_pending)

                # Slots 27-28: session_state (already partially set in scenario blocks)
                # Augment with turn count
                input_slots[t, 27, 1] = (t + 1) / max(1, seq_len)  # Progress ratio

                # Slot 29: cost_state (correlate tokens_ratio with cost mode)
                input_slots[t, 29, 0] = tokens_ratio[t, 0]
                input_slots[t, 29, 1:5] = 0.0
                input_slots[t, 29, 1 + cost_mode] = 1.0

                # Slot 30: user_state
                steps_since_user += 1
                if rng.rand() < 0.05:
                    steps_since_user = 0
                input_slots[t, 30, 0] = np.exp(-steps_since_user / 5.0)
                input_slots[t, 30, 1] = steps_since_user / 30.0

            self.traces.append(
                {
                    "input_slots": input_slots,
                    "context_candidates": candidates,
                    "candidate_mask": mask,
                    "session_labels": session_labels,
                    "tokens_spent_ratio": tokens_ratio,
                    "error_count": error_count,
                    "steps_since_progress": steps_no_progress,
                    "seq_length": seq_len,
                }
            )

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int) -> SessionTrace:
        trace = self.traces[idx]
        return SessionTrace(
            input_slots=torch.from_numpy(trace["input_slots"]),
            context_candidates=torch.from_numpy(trace["context_candidates"]),
            candidate_mask=torch.from_numpy(trace["candidate_mask"]).bool(),
            session_labels=torch.from_numpy(trace["session_labels"]),
            tokens_spent_ratio=torch.from_numpy(trace["tokens_spent_ratio"]),
            error_count=torch.from_numpy(trace["error_count"]),
            steps_since_progress=torch.from_numpy(trace["steps_since_progress"]),
            seq_length=trace["seq_length"],
        )


def collate_session_traces(batch: List[SessionTrace]) -> SessionTrace:
    """Collate a list of SessionTrace into a batched version.

    seq_length becomes a per-sample int64 tensor of shape (batch,) so that
    sequence masks can be built correctly per sample in the loss and eval code.
    """
    return SessionTrace(
        input_slots=torch.stack([t.input_slots for t in batch]),
        context_candidates=torch.stack([t.context_candidates for t in batch]),
        candidate_mask=torch.stack([t.candidate_mask for t in batch]),
        session_labels=torch.stack([t.session_labels for t in batch]),
        tokens_spent_ratio=torch.stack([t.tokens_spent_ratio for t in batch]),
        error_count=torch.stack([t.error_count for t in batch]),
        steps_since_progress=torch.stack([t.steps_since_progress for t in batch]),
        seq_length=torch.tensor([t.seq_length for t in batch], dtype=torch.int64),
    )
