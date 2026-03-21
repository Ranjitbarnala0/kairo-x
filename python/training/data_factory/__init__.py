"""
KAIRO-X Data Factory
====================

Data loading, trace generation, and batching for controller training.

Trace types:
    - context_traces: Context selection training data (primary for Phase 1)
    - budget_traces: Context budget allocation training data (primary for Phase 1)
    - action_traces: Action selection traces (Phase 2+)
    - session_traces: Session decision traces (Phase 2+)

The pipeline module assembles these into batched DataLoaders with
curriculum-aware sampling.
"""

from training.data_factory.pipeline import TrainingPipeline, TrainingBatch
from training.data_factory.context_traces import ContextTraceDataset
from training.data_factory.budget_traces import BudgetTraceDataset
from training.data_factory.action_traces import ActionTraceDataset
from training.data_factory.session_traces import SessionTraceDataset

__all__ = [
    "TrainingPipeline",
    "TrainingBatch",
    "ContextTraceDataset",
    "BudgetTraceDataset",
    "ActionTraceDataset",
    "SessionTraceDataset",
]
