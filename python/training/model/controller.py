"""
Full 42M-Parameter KAIRO-X Controller (section 7.2)
=====================================================

The controller is the central nervous system of KAIRO-X. It receives a
32-slot input packet at each time step and produces 6 decision signals
through specialized output heads. Between time steps it carries 768 dims
of recurrent state (4 bands x 192 dims).

Architecture summary:
    Input:  32 slots x 64 dims  ->  InputEmbedding  ->  d_model (288)
    Body:   12 x LiquidBlock (d_model=288, d_state=192, 4 bands)
    Output: 6 heads (action, context_selection, context_budget,
            enforcement_intensity, session_edge_case, stop)

Parameter count:
    ~41.5M parameters total
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from training.model.liquid_block import LiquidBlock, LiquidBlockConfig
from training.model.dense_block import InputEmbedding
from training.model.heads import (
    ActionHead,
    ContextSelectionHead,
    ContextBudgetHead,
    EnforcementIntensityHead,
    SessionEdgeCaseHead,
    StopHead,
)


@dataclass
class ControllerConfig:
    """Full configuration for the KAIRO-X controller."""

    # Core dimensions
    d_model: int = 288
    d_state: int = 192
    n_bands: int = 4
    d_ffn: int = 1152
    n_layers: int = 12

    # Input
    n_input_slots: int = 32
    d_slot: int = 64

    # Liquid block
    cross_band_heads: int = 4
    dropout: float = 0.1
    gru_bias: bool = True

    # Heads
    n_actions: int = 34
    max_context_candidates: int = 32
    d_candidate: int = 64

    def liquid_block_config(self) -> LiquidBlockConfig:
        """Create a LiquidBlockConfig from this controller config."""
        return LiquidBlockConfig(
            d_model=self.d_model,
            d_state=self.d_state,
            n_bands=self.n_bands,
            d_ffn=self.d_ffn,
            cross_band_heads=self.cross_band_heads,
            dropout=self.dropout,
            gru_bias=self.gru_bias,
        )


class KairoController(nn.Module):
    """
    The full 42M-parameter KAIRO-X controller.

    This model processes a sequence of input packets (one per time step) and
    produces decision signals through 6 output heads. It maintains recurrent
    state across time steps via the liquid blocks' GRU-style band updates.

    Usage:

        config = ControllerConfig()
        model = KairoController(config)

        # Initialize recurrent state
        state = model.initial_state(batch_size=16, device='cuda')

        # Process one time step
        outputs, state = model(input_slots, state,
                               context_candidates=candidates,
                               candidate_mask=mask,
                               itch_active=itch)

        # outputs is a dict with keys:
        #   'action_logits', 'context_scores', 'context_budget',
        #   'enforcement_intensity', 'session_edge_case', 'stop_logits'
    """

    def __init__(self, config: Optional[ControllerConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = ControllerConfig()
        self.config = config

        # --- Input embedding ---
        self.input_embedding = InputEmbedding(
            n_slots=config.n_input_slots,
            d_slot=config.d_slot,
            d_model=config.d_model,
            dropout=config.dropout,
        )

        # --- Liquid blocks (the recurrent body) ---
        block_config = config.liquid_block_config()
        self.layers = nn.ModuleList(
            [LiquidBlock(block_config) for _ in range(config.n_layers)]
        )

        # --- Output normalization ---
        self.output_norm = nn.LayerNorm(config.d_model)

        # --- 6 Output heads ---
        self.action_head = ActionHead(
            d_model=config.d_model, dropout=config.dropout
        )
        self.context_selection_head = ContextSelectionHead(
            d_model=config.d_model,
            d_candidate=config.d_candidate,
            max_candidates=config.max_context_candidates,
            dropout=config.dropout,
        )
        self.context_budget_head = ContextBudgetHead(
            d_model=config.d_model, dropout=config.dropout
        )
        self.enforcement_intensity_head = EnforcementIntensityHead(
            d_model=config.d_model, dropout=config.dropout
        )
        self.session_edge_case_head = SessionEdgeCaseHead(
            d_model=config.d_model, dropout=config.dropout
        )
        self.stop_head = StopHead(
            d_model=config.d_model, dropout=config.dropout
        )

    def initial_state(
        self, batch_size: int, device: torch.device
    ) -> List[List[torch.Tensor]]:
        """
        Create initial recurrent state for all layers.

        Returns:
            A list of length n_layers, where each element is a list of
            n_bands tensors of shape (batch_size, d_state).
        """
        return [
            layer.initial_state(batch_size, device) for layer in self.layers
        ]

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_slots: torch.Tensor,
        state: List[List[torch.Tensor]],
        context_candidates: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
        itch_active: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[List[torch.Tensor]]]:
        """
        Forward pass for one time step.

        Args:
            input_slots: Shape (batch, n_slots, d_slot). The 32-slot input
                packet assembled by the orchestrator.
            state: Recurrent state from the previous step, as returned by
                initial_state() or a previous forward() call.
            context_candidates: Optional, shape (batch, n_candidates, d_candidate).
                Embeddings of available context items for the context selection
                head. If None, context_scores will be zeros.
            candidate_mask: Optional, shape (batch, n_candidates). Boolean mask
                for valid candidates.
            itch_active: Optional, shape (batch,). Boolean indicating whether
                the itch signal is active, used by the stop head.

        Returns:
            outputs: Dict with keys:
                - 'action_logits': (batch, 34)
                - 'context_scores': (batch, n_candidates) or (batch, 0)
                - 'context_budget': (batch, 1)
                - 'enforcement_intensity': (batch, 1)
                - 'session_edge_case': (batch, 1)
                - 'stop_logits': (batch, 4)
            new_state: Updated recurrent state.
        """
        batch_size = input_slots.size(0)

        # 1. Embed input slots to d_model
        x = self.input_embedding(input_slots)  # (batch, d_model)

        # 2. Pass through liquid blocks
        new_state = []
        for layer, layer_state in zip(self.layers, state):
            x, updated_layer_state = layer(x, layer_state)
            new_state.append(updated_layer_state)

        # 3. Output normalization
        x = self.output_norm(x)  # (batch, d_model)

        # 4. Compute all head outputs
        outputs: Dict[str, torch.Tensor] = {}

        # Head 1: Action
        outputs["action_logits"] = self.action_head(x)

        # Head 2: Context selection
        if context_candidates is not None and context_candidates.size(1) > 0:
            outputs["context_scores"] = self.context_selection_head(
                x, context_candidates, candidate_mask
            )
        else:
            # Always return at least (batch, 1) to avoid shape (batch, 0)
            # which breaks loss computation. The zero tensor signals no
            # real candidates to the loss function.
            outputs["context_scores"] = torch.zeros(
                batch_size, 1, device=x.device
            )

        # Head 3: Context budget
        outputs["context_budget"] = self.context_budget_head(x)

        # Head 4: Enforcement intensity
        outputs["enforcement_intensity"] = self.enforcement_intensity_head(x)

        # Head 5: Session edge case
        outputs["session_edge_case"] = self.session_edge_case_head(x)

        # Head 6: Stop (itch-gated)
        outputs["stop_logits"] = self.stop_head(x, itch_active)

        return outputs, new_state

    def forward_sequence(
        self,
        input_sequence: torch.Tensor,
        state: Optional[List[List[torch.Tensor]]] = None,
        context_candidates_seq: Optional[torch.Tensor] = None,
        candidate_mask_seq: Optional[torch.Tensor] = None,
        itch_active_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[List[torch.Tensor]]]:
        """
        Process a full sequence of time steps. Used during training where we
        have a complete trace to process.

        Args:
            input_sequence: Shape (batch, seq_len, n_slots, d_slot).
            state: Optional initial state. If None, uses initial_state().
            context_candidates_seq: Optional, shape (batch, seq_len, n_candidates, d_candidate).
            candidate_mask_seq: Optional, shape (batch, seq_len, n_candidates).
            itch_active_seq: Optional, shape (batch, seq_len).

        Returns:
            outputs: Dict where each value has an added seq_len dimension:
                - 'action_logits': (batch, seq_len, 34)
                - 'context_scores': (batch, seq_len, n_candidates)
                - 'context_budget': (batch, seq_len, 1)
                - 'enforcement_intensity': (batch, seq_len, 1)
                - 'session_edge_case': (batch, seq_len, 1)
                - 'stop_logits': (batch, seq_len, 4)
            final_state: Recurrent state after the last time step.
        """
        batch_size, seq_len = input_sequence.shape[:2]
        device = input_sequence.device

        if state is None:
            state = self.initial_state(batch_size, device)

        # Collect outputs for each step
        all_outputs: Dict[str, list] = {
            "action_logits": [],
            "context_scores": [],
            "context_budget": [],
            "enforcement_intensity": [],
            "session_edge_case": [],
            "stop_logits": [],
        }

        for t in range(seq_len):
            slots_t = input_sequence[:, t]  # (batch, n_slots, d_slot)

            candidates_t = None
            mask_t = None
            itch_t = None

            if context_candidates_seq is not None:
                candidates_t = context_candidates_seq[:, t]
            if candidate_mask_seq is not None:
                mask_t = candidate_mask_seq[:, t]
            if itch_active_seq is not None:
                itch_t = itch_active_seq[:, t]

            step_outputs, state = self.forward(
                slots_t, state, candidates_t, mask_t, itch_t
            )

            for key in all_outputs:
                all_outputs[key].append(step_outputs[key])

        # Stack along time dimension
        outputs = {
            key: torch.stack(vals, dim=1) for key, vals in all_outputs.items()
        }

        return outputs, state
