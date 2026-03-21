"""
KAIRO-X Controller Model
=========================

The 42M-parameter liquid-hybrid recurrent controller.

Modules:
    - controller: Full model assembling liquid blocks, dense blocks, and heads
    - liquid_block: Liquid-hybrid recurrent block with 4 state bands
    - dense_block: Feedforward block with SiLU activation
    - heads: 6 output heads for action, context, budget, enforcement, session, stop
"""

from training.model.controller import KairoController
from training.model.liquid_block import LiquidBlock
from training.model.dense_block import DenseBlock
from training.model.heads import (
    ActionHead,
    ContextSelectionHead,
    ContextBudgetHead,
    EnforcementIntensityHead,
    SessionEdgeCaseHead,
    StopHead,
)

__all__ = [
    "KairoController",
    "LiquidBlock",
    "DenseBlock",
    "ActionHead",
    "ContextSelectionHead",
    "ContextBudgetHead",
    "EnforcementIntensityHead",
    "SessionEdgeCaseHead",
    "StopHead",
]
