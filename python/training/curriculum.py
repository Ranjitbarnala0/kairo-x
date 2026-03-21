"""
3-Phase Curriculum (section 17)
================================

The training curriculum progresses through three phases, each building on
the capabilities learned in the previous phase.

Phase 1: Context Selection + Budget
    Focus on the two primary heads (context selection and budget).
    Uses only context and budget traces.
    Gate to Phase 2: context recall >= 80%

Phase 2: Full Execution Traces
    Adds action selection, enforcement intensity, stop, and session heads.
    Uses all trace types with balanced sampling.
    Gate to Phase 3: task completion >= 50%

Phase 3: End-to-End with Cost Modes
    Full pipeline training with cost-aware optimization.
    Session edge cases are emphasized.
    No gate -- runs until convergence.

The curriculum manager tracks metrics, evaluates gating conditions, and
adjusts learning rate schedules per phase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from training.data_factory.pipeline import CurriculumPhase

logger = logging.getLogger(__name__)


@dataclass
class PhaseConfig:
    """Configuration for a single curriculum phase."""

    phase: CurriculumPhase
    name: str
    min_steps: int
    max_steps: int
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    gradient_clip: float
    eval_every_steps: int
    # Gate condition thresholds
    gate_metric: Optional[str] = None
    gate_threshold: Optional[float] = None
    # Description
    description: str = ""


@dataclass
class CurriculumConfig:
    """Full curriculum configuration."""

    phases: List[PhaseConfig] = field(default_factory=lambda: [
        PhaseConfig(
            phase=CurriculumPhase.PHASE_1,
            name="context_mastery",
            min_steps=5000,
            max_steps=50000,
            learning_rate=3e-4,
            warmup_steps=500,
            weight_decay=0.01,
            gradient_clip=1.0,
            eval_every_steps=500,
            gate_metric="context_recall",
            gate_threshold=0.80,
            description="Phase 1: Learn context selection and budget allocation",
        ),
        PhaseConfig(
            phase=CurriculumPhase.PHASE_2,
            name="execution_mastery",
            min_steps=10000,
            max_steps=100000,
            learning_rate=1e-4,
            warmup_steps=1000,
            weight_decay=0.01,
            gradient_clip=1.0,
            eval_every_steps=1000,
            gate_metric="task_completion",
            gate_threshold=0.50,
            description="Phase 2: Learn full execution flow with all heads",
        ),
        PhaseConfig(
            phase=CurriculumPhase.PHASE_3,
            name="cost_optimization",
            min_steps=10000,
            max_steps=200000,
            learning_rate=5e-5,
            warmup_steps=1000,
            weight_decay=0.01,
            gradient_clip=0.5,
            eval_every_steps=2000,
            gate_metric=None,
            gate_threshold=None,
            description="Phase 3: End-to-end training with cost optimization",
        ),
    ])

    # Early stopping
    patience: int = 10  # Number of evals without improvement before stopping
    min_improvement: float = 0.001  # Minimum improvement to count as progress


@dataclass
class PhaseMetrics:
    """Tracked metrics for a single phase."""

    step_count: int = 0
    best_gate_metric: float = 0.0
    best_total_loss: float = float("inf")
    evals_without_improvement: int = 0
    loss_history: List[float] = field(default_factory=list)
    gate_metric_history: List[float] = field(default_factory=list)
    is_complete: bool = False
    gate_passed: bool = False


class CurriculumManager:
    """
    Manages the 3-phase training curriculum.

    Tracks metrics, evaluates gating conditions, adjusts optimizer
    settings, and determines when to advance phases.

    Usage:
        curriculum = CurriculumManager()

        while not curriculum.is_complete():
            phase_config = curriculum.current_phase_config()
            # ... train for one step ...
            curriculum.step(loss, metrics)

            if curriculum.should_evaluate():
                eval_metrics = evaluate(model)
                advance = curriculum.evaluate(eval_metrics)
                if advance:
                    curriculum.advance_phase()
    """

    def __init__(self, config: Optional[CurriculumConfig] = None) -> None:
        if config is None:
            config = CurriculumConfig()
        self.config = config
        self._phase_idx = 0
        self._phase_metrics: List[PhaseMetrics] = [
            PhaseMetrics() for _ in config.phases
        ]
        self._global_step = 0

    @property
    def current_phase(self) -> CurriculumPhase:
        """The current curriculum phase enum."""
        return self.config.phases[self._phase_idx].phase

    @property
    def phase_index(self) -> int:
        """Zero-based index of the current phase."""
        return self._phase_idx

    def current_phase_config(self) -> PhaseConfig:
        """Get configuration for the current phase."""
        return self.config.phases[self._phase_idx]

    def current_metrics(self) -> PhaseMetrics:
        """Get metrics tracker for the current phase."""
        return self._phase_metrics[self._phase_idx]

    @property
    def global_step(self) -> int:
        """Total steps across all phases."""
        return self._global_step

    def is_complete(self) -> bool:
        """Whether all phases have been completed."""
        return self._phase_idx >= len(self.config.phases)

    def step(self, loss: float, metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Record one training step.

        Args:
            loss: The training loss for this step.
            metrics: Optional dict of additional metrics.
        """
        if self.is_complete():
            return

        pm = self._phase_metrics[self._phase_idx]
        pm.step_count += 1
        pm.loss_history.append(loss)
        self._global_step += 1

        if loss < pm.best_total_loss - self.config.min_improvement:
            pm.best_total_loss = loss
            pm.evals_without_improvement = 0

    def should_evaluate(self) -> bool:
        """Whether it's time to run evaluation."""
        if self.is_complete():
            return False
        pc = self.current_phase_config()
        pm = self.current_metrics()
        return pm.step_count > 0 and pm.step_count % pc.eval_every_steps == 0

    def evaluate(self, eval_metrics: Dict[str, float]) -> bool:
        """
        Process evaluation results and check gating conditions.

        Args:
            eval_metrics: Dict of evaluation metric values. Must contain
                the gate_metric key if the current phase has a gate.

        Returns:
            True if the phase should advance (gate condition met).
        """
        if self.is_complete():
            return False

        pc = self.current_phase_config()
        pm = self.current_metrics()

        # Track gate metric
        if pc.gate_metric is not None and pc.gate_metric in eval_metrics:
            gate_val = eval_metrics[pc.gate_metric]
            pm.gate_metric_history.append(gate_val)

            if gate_val > pm.best_gate_metric + self.config.min_improvement:
                pm.best_gate_metric = gate_val
                pm.evals_without_improvement = 0
            else:
                pm.evals_without_improvement += 1

            logger.info(
                "Phase %d [%s] step %d: %s = %.4f (best = %.4f, patience = %d/%d)",
                pc.phase,
                pc.name,
                pm.step_count,
                pc.gate_metric,
                gate_val,
                pm.best_gate_metric,
                pm.evals_without_improvement,
                self.config.patience,
            )

            # Check gate
            if gate_val >= pc.gate_threshold and pm.step_count >= pc.min_steps:
                pm.gate_passed = True
                logger.info(
                    "Gate PASSED for phase %d: %s = %.4f >= %.4f",
                    pc.phase,
                    pc.gate_metric,
                    gate_val,
                    pc.gate_threshold,
                )
                return True
        else:
            pm.evals_without_improvement += 1

        # Check if we should force-advance due to max steps
        if pm.step_count >= pc.max_steps:
            logger.warning(
                "Phase %d hit max_steps (%d). Force-advancing.",
                pc.phase,
                pc.max_steps,
            )
            return True

        # Check early stopping (patience exhausted)
        if pm.evals_without_improvement >= self.config.patience:
            logger.warning(
                "Phase %d patience exhausted (%d evals without improvement). "
                "Force-advancing with best gate metric = %.4f.",
                pc.phase,
                pm.evals_without_improvement,
                pm.best_gate_metric,
            )
            return True

        return False

    def advance_phase(self) -> bool:
        """
        Advance to the next curriculum phase.

        Returns:
            True if there is a next phase, False if training is complete.
        """
        pm = self.current_metrics()
        pm.is_complete = True

        logger.info(
            "Completing phase %d [%s] after %d steps. Best gate metric: %.4f",
            self.current_phase,
            self.current_phase_config().name,
            pm.step_count,
            pm.best_gate_metric,
        )

        self._phase_idx += 1

        if self.is_complete():
            logger.info("All curriculum phases complete!")
            return False

        pc = self.current_phase_config()
        logger.info(
            "Advancing to phase %d [%s]: %s",
            pc.phase,
            pc.name,
            pc.description,
        )
        return True

    def get_learning_rate(self) -> float:
        """
        Get the current learning rate with warmup.

        Returns the learning rate for the current phase, with linear warmup
        applied during the initial warmup_steps.
        """
        if self.is_complete():
            return 0.0

        pc = self.current_phase_config()
        pm = self.current_metrics()

        if pm.step_count < pc.warmup_steps:
            # Linear warmup
            warmup_factor = (pm.step_count + 1) / pc.warmup_steps
            return pc.learning_rate * warmup_factor
        else:
            # Cosine decay within the phase
            progress = (pm.step_count - pc.warmup_steps) / max(
                1, pc.max_steps - pc.warmup_steps
            )
            progress = min(progress, 1.0)
            import math
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Don't decay below 10% of peak LR
            return pc.learning_rate * max(0.1, cosine_factor)

    def state_dict(self) -> Dict:
        """Serialize curriculum state for checkpointing."""
        return {
            "phase_idx": self._phase_idx,
            "global_step": self._global_step,
            "phase_metrics": [
                {
                    "step_count": pm.step_count,
                    "best_gate_metric": pm.best_gate_metric,
                    "best_total_loss": pm.best_total_loss,
                    "evals_without_improvement": pm.evals_without_improvement,
                    "loss_history": pm.loss_history[-100:],  # Keep last 100
                    "gate_metric_history": pm.gate_metric_history,
                    "is_complete": pm.is_complete,
                    "gate_passed": pm.gate_passed,
                }
                for pm in self._phase_metrics
            ],
        }

    def load_state_dict(self, state: Dict) -> None:
        """Restore curriculum state from a checkpoint."""
        self._phase_idx = state["phase_idx"]
        self._global_step = state["global_step"]
        for pm, saved in zip(self._phase_metrics, state["phase_metrics"]):
            pm.step_count = saved["step_count"]
            pm.best_gate_metric = saved["best_gate_metric"]
            pm.best_total_loss = saved["best_total_loss"]
            pm.evals_without_improvement = saved["evals_without_improvement"]
            pm.loss_history = saved["loss_history"]
            pm.gate_metric_history = saved["gate_metric_history"]
            pm.is_complete = saved["is_complete"]
            pm.gate_passed = saved["gate_passed"]
