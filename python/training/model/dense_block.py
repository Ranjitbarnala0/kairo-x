"""
Dense Feedforward Block
========================

A standard feedforward block used in the input embedding stage and output
projection stage of the controller. Uses SiLU (Swish) activation following
modern best practices for small transformer-like models.

This block is simpler than LiquidBlock -- it has no recurrent state or
cross-band attention. It is used for:
    - Projecting the 32 input slots into d_model space
    - Intermediate transforms in output heads
    - Any non-recurrent path in the architecture
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class DenseBlockConfig:
    """Configuration for a DenseBlock."""

    d_input: int = 288
    d_output: int = 288
    d_hidden: int = 1152
    dropout: float = 0.1
    use_residual: bool = True
    use_layer_norm: bool = True


class DenseBlock(nn.Module):
    """
    Dense feedforward block with SiLU activation.

    Architecture:
        LayerNorm -> Linear(d_input, d_hidden) -> SiLU -> Dropout
        -> Linear(d_hidden, d_output) -> Dropout -> Residual (if d_input == d_output)

    When use_residual=True and d_input == d_output, the block adds its output
    to its input (pre-norm residual). When dimensions differ, the residual
    connection is skipped or a projection is applied.
    """

    def __init__(self, config: Optional[DenseBlockConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = DenseBlockConfig()
        self.config = config

        self.norm = nn.LayerNorm(config.d_input) if config.use_layer_norm else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(config.d_input, config.d_hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_hidden, config.d_output),
            nn.Dropout(config.dropout),
        )

        # Residual projection when dimensions differ
        self.use_residual = config.use_residual
        if config.use_residual and config.d_input != config.d_output:
            self.residual_proj = nn.Linear(config.d_input, config.d_output, bias=False)
        else:
            self.residual_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (..., d_input).

        Returns:
            Output tensor, shape (..., d_output).
        """
        normed = self.norm(x)
        out = self.ffn(normed)

        if self.use_residual:
            if self.residual_proj is not None:
                out = out + self.residual_proj(x)
            elif self.config.d_input == self.config.d_output:
                out = out + x

        return out


class InputEmbedding(nn.Module):
    """
    Embeds the 32 input slots into d_model space.

    Each of the 32 input slots is a fixed-size feature vector. The slots have
    different semantic meanings (execution graph, active node, LLM responses,
    etc.) but are all projected to the same d_model dimensionality, then
    summed with learned positional embeddings for each slot.

    Slot layout (section 7.4):
        execution_graph_summary: 6 slots
        active_node_context:     4 slots
        recent_llm_responses:    4 slots
        project_state:           3 slots
        available_context_candidates: 5 slots
        verification_state:      3 slots
        itch_state:              2 slots
        session_state:           2 slots
        cost_state:              1 slot
        user_state:              1 slot
        padding:                 1 slot
        ----------------------------------
        Total:                  32 slots
    """

    # Slot boundaries for semantic grouping
    SLOT_GROUPS = {
        "execution_graph_summary": (0, 6),
        "active_node_context": (6, 10),
        "recent_llm_responses": (10, 14),
        "project_state": (14, 17),
        "available_context_candidates": (17, 22),
        "verification_state": (22, 25),
        "itch_state": (25, 27),
        "session_state": (27, 29),
        "cost_state": (29, 30),
        "user_state": (30, 31),
        "padding": (31, 32),
    }

    def __init__(
        self,
        n_slots: int = 32,
        d_slot: int = 64,
        d_model: int = 288,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_slots = n_slots
        self.d_slot = d_slot
        self.d_model = d_model

        # Per-slot linear projection
        self.slot_proj = nn.Linear(d_slot, d_model)

        # Learned positional embedding per slot
        self.slot_positions = nn.Embedding(n_slots, d_model)

        # Group-level embedding so the model knows which semantic group a slot
        # belongs to (11 groups)
        n_groups = len(self.SLOT_GROUPS)
        self.group_embedding = nn.Embedding(n_groups, d_model)

        # Build a mapping from slot index -> group index.
        # Registered as a buffer so it moves to GPU with .cuda()/.to(device).
        _slot_to_group = torch.zeros(n_slots, dtype=torch.long)
        for group_idx, (_, (start, end)) in enumerate(self.SLOT_GROUPS.items()):
            for s in range(start, end):
                _slot_to_group[s] = group_idx
        self.register_buffer('_slot_to_group', _slot_to_group)

        # Determine attention head count: use 6 for standard config, fall back
        # to the largest divisor <= 6 for non-standard d_model.
        n_attn_heads = 6
        while d_model % n_attn_heads != 0 and n_attn_heads > 1:
            n_attn_heads -= 1

        # Slot-level self-attention: lets slots interact before pooling.
        # This is critical for cross-slot reasoning (e.g., relating
        # execution graph state to available context candidates).
        self.slot_self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_attn_heads, dropout=dropout, batch_first=True
        )
        self.slot_attn_norm = nn.LayerNorm(d_model)

        # Combine: 2-layer FFN per slot after self-attention
        d_combine = d_model * 4  # 1152 for d_model=288
        self.combine = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_combine),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_combine, d_model),
            nn.Dropout(dropout),
        )
        self.combine_norm = nn.LayerNorm(d_model)

        # Attention pooling over slots to produce a single d_model vector
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_attn_heads, dropout=dropout, batch_first=True
        )
        self.pool_norm = nn.LayerNorm(d_model)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Embed input slots into a single d_model vector.

        Args:
            slots: Input tensor, shape (batch, n_slots, d_slot).

        Returns:
            Embedded representation, shape (batch, d_model).
        """
        B, N, D = slots.shape
        assert N == self.n_slots, f"Expected {self.n_slots} slots, got {N}"
        assert D == self.d_slot, f"Expected slot dim {self.d_slot}, got {D}"

        device = slots.device

        # Project each slot
        projected = self.slot_proj(slots)  # (B, N, d_model)

        # Add positional embeddings
        positions = torch.arange(N, device=device)
        pos_emb = self.slot_positions(positions)  # (N, d_model)
        projected = projected + pos_emb.unsqueeze(0)

        # Add group embeddings
        group_ids = self._slot_to_group.to(device)
        group_emb = self.group_embedding(group_ids)  # (N, d_model)
        projected = projected + group_emb.unsqueeze(0)

        # Slot-level self-attention: cross-slot reasoning
        attn_input = self.slot_attn_norm(projected)
        attn_out, _ = self.slot_self_attn(attn_input, attn_input, attn_input)
        projected = projected + attn_out  # Residual

        # Per-slot FFN
        combine_out = self.combine(projected)  # (B, N, d_model)
        projected = self.combine_norm(projected + combine_out)  # Residual + norm

        # Attention pooling: use a learnable query to attend over all slots
        query = self.pool_query.expand(B, -1, -1)  # (B, 1, d_model)
        pooled, _ = self.pool_attn(query, projected, projected)  # (B, 1, d_model)
        pooled = self.pool_norm(pooled.squeeze(1))  # (B, d_model)

        return pooled
