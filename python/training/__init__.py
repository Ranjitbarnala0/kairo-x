"""
KAIRO-X Training Pipeline
==========================

PyTorch training pipeline for the 42M-parameter KAIRO-X controller.

The controller is a liquid-hybrid recurrent network with 4 state bands
that learns to orchestrate agentic coding workflows: selecting context,
budgeting tokens, choosing actions, and managing sessions.

Architecture (section 7):
    - d_model = 288, d_state = 192 per band (4 bands = 768 total)
    - d_ffn = 1152, n_layers = 12, n_input_slots = 32
    - 6 output heads: action, context_selection, context_budget,
      enforcement_intensity, session_edge_case, stop

Training (section 17):
    - 3-phase curriculum with gating conditions
    - 6 weighted losses
    - Export to binary format for Rust inference
"""

__version__ = "0.1.0"
