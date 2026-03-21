"""
Action Selection Training Traces (Phase 2+)
=============================================

Generates and loads traces for training the action head, enforcement
intensity head, and stop head.

Each trace represents a sequence of time steps from an agentic coding
session. At each step the controller observes the current state and must
choose one of 34 actions (e.g., read_file, write_file, run_test, etc.).

Action traces also carry labels for:
    - enforcement_intensity: how strictly rules should be applied
    - stop_decision: continue / escalate / retry / terminate
    - itch_active: whether the itch signal was present

These traces are used from Phase 2 onward, after the context selection
heads have reached baseline quality.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
import numpy as np


# The 34 actions the controller can take
ACTION_NAMES = [
    "read_file",               # 0
    "write_file",              # 1
    "create_file",             # 2
    "delete_file",             # 3
    "search_codebase",         # 4
    "search_symbols",          # 5
    "list_directory",          # 6
    "run_test",                # 7
    "run_test_suite",          # 8
    "run_lint",                # 9
    "run_typecheck",           # 10
    "run_build",               # 11
    "run_command",             # 12
    "call_llm_generate",       # 13
    "call_llm_edit",           # 14
    "call_llm_review",         # 15
    "call_llm_plan",           # 16
    "call_llm_debug",          # 17
    "git_status",              # 18
    "git_diff",                # 19
    "git_commit",              # 20
    "advance_graph_node",      # 21
    "spawn_parallel_branch",   # 22
    "merge_parallel_results",  # 23
    "set_checkpoint",          # 24
    "rollback_to_checkpoint",  # 25
    "request_user_input",      # 26
    "report_progress",         # 27
    "escalate_to_user",        # 28
    "apply_enforcement",       # 29
    "relax_enforcement",       # 30
    "refresh_context",         # 31
    "trim_context",            # 32
    "noop",                    # 33
]

N_ACTIONS = len(ACTION_NAMES)
assert N_ACTIONS == 34, f"Expected 34 actions, got {N_ACTIONS}"


@dataclass
class ActionTrace:
    """A single action selection training example (one time step)."""

    input_slots: torch.Tensor           # (n_slots, d_slot)
    context_candidates: torch.Tensor    # (n_candidates, d_candidate)
    candidate_mask: torch.Tensor        # (n_candidates,) bool
    action_label: torch.Tensor          # () int64 in [0, 33]
    enforcement_label: torch.Tensor     # (1,) float in [-0.3, 0.3]
    stop_label: torch.Tensor            # () int64 in [0, 3]
    itch_active: torch.Tensor           # () bool

    def to(self, device: torch.device) -> "ActionTrace":
        """Move all tensors to device."""
        return ActionTrace(
            input_slots=self.input_slots.to(device),
            context_candidates=self.context_candidates.to(device),
            candidate_mask=self.candidate_mask.to(device),
            action_label=self.action_label.to(device),
            enforcement_label=self.enforcement_label.to(device),
            stop_label=self.stop_label.to(device),
            itch_active=self.itch_active.to(device),
        )


@dataclass
class ActionSequenceTrace:
    """A sequence of action traces from a single session segment."""

    input_slots: torch.Tensor           # (seq_len, n_slots, d_slot)
    context_candidates: torch.Tensor    # (seq_len, n_candidates, d_candidate)
    candidate_mask: torch.Tensor        # (seq_len, n_candidates) bool
    action_labels: torch.Tensor         # (seq_len,) int64
    enforcement_labels: torch.Tensor    # (seq_len, 1) float
    stop_labels: torch.Tensor           # (seq_len,) int64
    itch_active: torch.Tensor           # (seq_len,) bool
    seq_length: int                     # Actual length (before padding)

    def to(self, device: torch.device) -> "ActionSequenceTrace":
        """Move all tensors to device."""
        return ActionSequenceTrace(
            input_slots=self.input_slots.to(device),
            context_candidates=self.context_candidates.to(device),
            candidate_mask=self.candidate_mask.to(device),
            action_labels=self.action_labels.to(device),
            enforcement_labels=self.enforcement_labels.to(device),
            stop_labels=self.stop_labels.to(device),
            itch_active=self.itch_active.to(device),
            seq_length=self.seq_length,
        )


class ActionTraceDataset(Dataset):
    """
    Dataset of action selection traces.

    Each item is a sequence of time steps (ActionSequenceTrace). Sequences
    are padded to max_seq_len. The dataset supports:
        - Loading from .npz files
        - Synthetic generation from simulated coding sessions
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
        seed: int = 456,
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
        """Load action sequence traces from .npz files."""
        data_path = Path(data_dir)
        for npz_file in sorted(data_path.glob("action_*.npz")):
            data = np.load(npz_file)
            self.traces.append(
                {
                    "input_slots": data["input_slots"],
                    "context_candidates": data["context_candidates"],
                    "candidate_mask": data["candidate_mask"],
                    "action_labels": data["action_labels"],
                    "enforcement_labels": data["enforcement_labels"],
                    "stop_labels": data["stop_labels"],
                    "itch_active": data["itch_active"],
                    "seq_length": int(data["seq_length"]),
                }
            )

    def _generate_synthetic(self, n_traces: int, seed: int) -> None:
        """
        Generate synthetic action sequences.

        Simulates a simplified coding session: the agent progresses through a
        series of actions following a typical pattern (plan -> read -> edit ->
        test -> commit), with occasional branches and error recovery.
        """
        rng = np.random.RandomState(seed)

        # Typical action subsequences
        common_patterns = [
            [16, 4, 0, 0, 14, 7, 21],          # plan, search, read x2, edit, test, advance
            [0, 0, 0, 13, 1, 7, 21],            # read x3, generate, write, test, advance
            [4, 0, 14, 7, 9, 10, 20],           # search, read, edit, test, lint, typecheck, commit
            [0, 17, 25, 0, 14, 7, 21],          # read, debug, rollback, read, edit, test, advance
            [22, 13, 13, 23, 7, 21],            # spawn_parallel, generate x2, merge, test, advance
            [26, 0, 14, 7, 27],                 # request_input, read, edit, test, report_progress
        ]

        for _ in range(n_traces):
            # Build a random-length sequence from patterns
            seq = []
            while len(seq) < self.max_seq_len - 5:
                pattern = common_patterns[rng.randint(len(common_patterns))]
                # Add some noise: occasionally substitute actions
                noisy_pattern = []
                for a in pattern:
                    if rng.rand() < 0.1:
                        a = rng.randint(N_ACTIONS)
                    noisy_pattern.append(a)
                seq.extend(noisy_pattern)

            seq = seq[: self.max_seq_len]
            actual_len = len(seq)

            # Pad to max_seq_len
            while len(seq) < self.max_seq_len:
                seq.append(33)  # noop

            action_labels = np.array(seq, dtype=np.int64)

            # Generate input slots for each step
            input_slots = rng.randn(
                self.max_seq_len, self.n_slots, self.d_slot
            ).astype(np.float32) * 0.3

            # Evolve the input slightly along the sequence (simulate changing state)
            drift = rng.randn(self.n_slots, self.d_slot).astype(np.float32) * 0.01
            for t in range(1, self.max_seq_len):
                input_slots[t] = input_slots[t - 1] + drift + rng.randn(
                    self.n_slots, self.d_slot
                ).astype(np.float32) * 0.05

            # --- Populate all slot ranges with structured features ---
            n_response_classes = 8
            pending_itch = rng.randint(0, 6)
            failed_itch = rng.randint(0, pending_itch + 1)
            budget_remaining = rng.uniform(0.1, 1.0)
            cost_mode = rng.randint(0, 4)
            steps_since_user = rng.randint(0, 30)

            for t in range(actual_len):
                # Slots 10-13: recent_llm_responses (evolving along sequence)
                for s in range(10, 14):
                    input_slots[t, s, :n_response_classes] = 0.0
                    if s - 10 < min(4, t + 1):
                        # Response class correlates with the action taken
                        prev_action = action_labels[max(0, t - (s - 10))]
                        if prev_action in (16,):      # call_llm_plan
                            cls = 0  # Plan
                        elif prev_action in (13, 14):  # generate/edit
                            cls = 1  # Implementation
                        elif prev_action in (7, 8):    # test
                            cls = 2 if rng.rand() < 0.6 else 3  # Pass/Fail
                        elif prev_action in (17,):     # debug
                            cls = 4  # Debug
                        elif prev_action in (15,):     # review
                            cls = 5  # Review
                        else:
                            cls = rng.randint(0, n_response_classes)
                        input_slots[t, s, cls] = 1.0 - (s - 10) * 0.2

                # Slots 14-16: project_state (evolving file mods, build status)
                file_mods = min(20, t + rng.randint(0, 3))
                input_slots[t, 14, 0] = file_mods / 20.0
                input_slots[t, 14, 1] = rng.rand()
                build_pass = 1.0 if rng.rand() < 0.7 else 0.0
                input_slots[t, 15, 0] = build_pass
                input_slots[t, 15, 1] = 1.0 - build_pass
                input_slots[t, 16, 0] = 1.0 if rng.rand() < 0.8 else 0.0
                input_slots[t, 16, 1] = 1.0 if rng.rand() < 0.75 else 0.0

                # Slots 22-24: verification_state
                l1_pass = 1.0 if rng.rand() < 0.6 else 0.0
                l2_pass = 1.0 if (l1_pass > 0 and rng.rand() < 0.7) else 0.0
                input_slots[t, 22, 0] = l1_pass
                input_slots[t, 22, 1] = 1.0 - l1_pass
                input_slots[t, 23, 0] = l2_pass
                input_slots[t, 23, 1] = 1.0 - l2_pass
                input_slots[t, 24, 0] = min(1.0, t / max(1, actual_len))

                # Slots 25-26: itch_state
                # Itch grows over time, correlated with action distribution
                cur_pending = min(6, pending_itch + (1 if rng.rand() < 0.15 else 0))
                cur_failed = min(cur_pending, failed_itch + (1 if rng.rand() < 0.1 else 0))
                input_slots[t, 25, 0] = cur_pending / 6.0
                input_slots[t, 25, 1] = rng.rand()
                input_slots[t, 26, 0] = cur_failed / 6.0
                input_slots[t, 26, 1] = cur_failed / max(1, cur_pending)

                # Slots 27-28: session_state
                input_slots[t, 27, 0] = (t + 1) / 50.0       # Turn count
                input_slots[t, 27, 1] = (t + 1) / max(1, actual_len)  # Progress
                token_spend = rng.uniform(500, 32000) * ((t + 1) / actual_len)
                input_slots[t, 28, 0] = token_spend / 32000.0

                # Slot 29: cost_state
                budget_remaining = max(0.0, budget_remaining - rng.uniform(0, 0.03))
                input_slots[t, 29, 0] = budget_remaining
                input_slots[t, 29, 1:5] = 0.0
                input_slots[t, 29, 1 + cost_mode] = 1.0

                # Slot 30: user_state
                steps_since_user += 1
                if rng.rand() < 0.05:
                    steps_since_user = 0  # User sent a message
                input_slots[t, 30, 0] = np.exp(-steps_since_user / 5.0)
                input_slots[t, 30, 1] = steps_since_user / 30.0

            # High itch count biases action distribution toward debug/rollback
            if pending_itch >= 3:
                for t in range(actual_len):
                    if rng.rand() < 0.3:
                        action_labels[t] = rng.choice([17, 25, 7, 9])  # debug, rollback, test, lint

            # Context candidates (constant across the sequence for simplicity)
            n_real = rng.randint(3, self.max_candidates + 1)
            candidates = np.zeros(
                (self.max_seq_len, self.max_candidates, self.d_candidate),
                dtype=np.float32,
            )
            mask = np.zeros(
                (self.max_seq_len, self.max_candidates), dtype=np.float32
            )
            base_candidates = rng.randn(
                self.max_candidates, self.d_candidate
            ).astype(np.float32) * 0.5
            for t in range(self.max_seq_len):
                candidates[t] = base_candidates
                mask[t, :n_real] = 1.0

            # Enforcement intensity: mostly neutral, occasionally non-zero
            enforcement = np.zeros((self.max_seq_len, 1), dtype=np.float32)
            for t in range(actual_len):
                if rng.rand() < 0.2:
                    enforcement[t, 0] = rng.uniform(-0.3, 0.3)

            # Stop labels: almost always continue (0), rarely escalate/retry/terminate
            stop_labels = np.zeros(self.max_seq_len, dtype=np.int64)
            itch_active = np.zeros(self.max_seq_len, dtype=np.float32)

            # Occasionally activate itch near end of sequence
            if rng.rand() < 0.3 and actual_len > 5:
                itch_start = rng.randint(actual_len - 3, actual_len)
                for t in range(itch_start, actual_len):
                    itch_active[t] = 1.0
                    stop_labels[t] = rng.choice([0, 1, 2, 3], p=[0.4, 0.2, 0.2, 0.2])

            self.traces.append(
                {
                    "input_slots": input_slots,
                    "context_candidates": candidates,
                    "candidate_mask": mask,
                    "action_labels": action_labels,
                    "enforcement_labels": enforcement,
                    "stop_labels": stop_labels,
                    "itch_active": itch_active,
                    "seq_length": actual_len,
                }
            )

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int) -> ActionSequenceTrace:
        trace = self.traces[idx]
        return ActionSequenceTrace(
            input_slots=torch.from_numpy(trace["input_slots"]),
            context_candidates=torch.from_numpy(trace["context_candidates"]),
            candidate_mask=torch.from_numpy(trace["candidate_mask"]).bool(),
            action_labels=torch.from_numpy(trace["action_labels"]),
            enforcement_labels=torch.from_numpy(trace["enforcement_labels"]),
            stop_labels=torch.from_numpy(trace["stop_labels"]),
            itch_active=torch.from_numpy(trace["itch_active"]).bool(),
            seq_length=trace["seq_length"],
        )


def collate_action_sequences(batch: List[ActionSequenceTrace]) -> ActionSequenceTrace:
    """Collate a list of ActionSequenceTrace into a batched version."""
    return ActionSequenceTrace(
        input_slots=torch.stack([t.input_slots for t in batch]),
        context_candidates=torch.stack([t.context_candidates for t in batch]),
        candidate_mask=torch.stack([t.candidate_mask for t in batch]),
        action_labels=torch.stack([t.action_labels for t in batch]),
        enforcement_labels=torch.stack([t.enforcement_labels for t in batch]),
        stop_labels=torch.stack([t.stop_labels for t in batch]),
        itch_active=torch.stack([t.itch_active for t in batch]),
        seq_length=max(t.seq_length for t in batch),
    )
