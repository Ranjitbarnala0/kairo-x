"""
Liquid-Hybrid Recurrent Block (section 7.3)
============================================

A single layer of the KAIRO-X controller. Each block processes the 4 state
bands independently with GRU-style gating, then applies lightweight
cross-band attention so bands can share information, followed by a
feedforward network with SiLU activation.

State bands:
    Band 0 -- Context (192 dims): what context has been sent / available
    Band 1 -- Execution (192 dims): graph position, parallel states
    Band 2 -- Quality (192 dims): compliance patterns, verification outcomes
    Band 3 -- Communication (192 dims): session health, token spend

Data flow per block:
    1. Project input to d_model
    2. Split into 4 band projections
    3. Each band: GRU-style gating with state update
    4. Cross-band attention (lightweight)
    5. FFN with SiLU activation
    6. Residual + LayerNorm
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LiquidBlockConfig:
    """Configuration for a single LiquidBlock."""

    d_model: int = 288
    d_state: int = 192
    n_bands: int = 4
    d_ffn: int = 1152
    cross_band_heads: int = 4
    dropout: float = 0.1
    gru_bias: bool = True


class BandGRU(nn.Module):
    """
    GRU-style gating for a single state band.

    Each band maintains a recurrent state of size d_state. On each step the
    band receives a projection from the main d_model stream and uses it to
    compute update and reset gates, then blends the candidate state into the
    carried state.
    """

    def __init__(self, d_input: int, d_state: int, bias: bool = True) -> None:
        super().__init__()
        self.d_state = d_state

        # Gates: update (z), reset (r), candidate (n)
        # Input projection: from d_input to 3 * d_state
        self.W_input = nn.Linear(d_input, 3 * d_state, bias=bias)
        # State projection: from d_state to 3 * d_state
        self.W_state = nn.Linear(d_state, 3 * d_state, bias=bias)

        # Learnable initial state
        self.h0 = nn.Parameter(torch.zeros(d_state))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize with orthogonal weights and neutral gate biases.

        The update gate bias is set to 0.0 (neutral) so the network can
        learn the appropriate bias direction during training. A positive
        bias (e.g. +1.0) would push z toward 1, causing h_new = z*h_old +
        (1-z)*candidate to heavily favour the old state, hindering learning
        of new representations.
        """
        nn.init.orthogonal_(self.W_input.weight)
        nn.init.orthogonal_(self.W_state.weight)
        if self.W_input.bias is not None:
            nn.init.zeros_(self.W_input.bias)
            nn.init.zeros_(self.W_state.bias)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return the learnable initial state expanded to batch size.

        Returns:
            Tensor of shape (batch_size, d_state).
        """
        return self.h0.unsqueeze(0).expand(batch_size, -1)

    def forward(
        self, x: torch.Tensor, h_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for one time step.

        Args:
            x: Input projection, shape (batch, d_input).
            h_prev: Previous hidden state, shape (batch, d_state).

        Returns:
            output: Band output, shape (batch, d_state).
            h_new: Updated hidden state, shape (batch, d_state).
        """
        # Compute all gates in one matmul each
        x_proj = self.W_input(x)  # (batch, 3 * d_state)
        h_proj = self.W_state(h_prev)  # (batch, 3 * d_state)

        x_z, x_r, x_n = x_proj.chunk(3, dim=-1)
        h_z, h_r, h_n = h_proj.chunk(3, dim=-1)

        z = torch.sigmoid(x_z + h_z)  # Update gate
        r = torch.sigmoid(x_r + h_r)  # Reset gate
        n = torch.tanh(x_n + r * h_n)  # Candidate

        h_new = (1.0 - z) * n + z * h_prev
        return h_new, h_new


class CrossBandAttention(nn.Module):
    """
    Lightweight multi-head attention across the 4 state bands.

    After each band has been updated independently, cross-band attention
    lets bands exchange information. This is cheaper than full self-attention
    over the sequence because it operates on the 4 band vectors (each d_state)
    rather than on the full sequence length.
    """

    def __init__(
        self,
        d_state: int,
        n_bands: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_state = d_state
        self.n_bands = n_bands
        self.d_head = d_state // n_heads
        assert d_state % n_heads == 0, "d_state must be divisible by n_heads"

        self.qkv = nn.Linear(d_state, 3 * d_state, bias=False)
        self.out_proj = nn.Linear(d_state, d_state, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(self, band_states: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-band attention.

        Args:
            band_states: Shape (batch, n_bands, d_state).

        Returns:
            Updated band states, shape (batch, n_bands, d_state).
        """
        B, N, D = band_states.shape  # (batch, n_bands, d_state)

        qkv = self.qkv(band_states)  # (B, N, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, N, D)

        # Reshape for multi-head: (B, n_heads, N, d_head)
        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, N, d_head)
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # (B, N, D)
        return self.out_proj(out)


class LiquidBlock(nn.Module):
    """
    Liquid-hybrid recurrent block.

    One layer of the KAIRO-X controller. Each block independently updates 4
    state bands with GRU-style gating, applies cross-band attention, then
    passes through a feedforward network with residual connections.
    """

    def __init__(self, config: Optional[LiquidBlockConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = LiquidBlockConfig()
        self.config = config

        d_model = config.d_model
        d_state = config.d_state
        n_bands = config.n_bands
        d_ffn = config.d_ffn

        # --- Input projection ---
        # Project d_model to n_bands * d_state for band splitting
        self.input_proj = nn.Linear(d_model, n_bands * d_state)
        self.input_norm = nn.LayerNorm(d_model)

        # --- Per-band GRU ---
        self.band_grus = nn.ModuleList(
            [BandGRU(d_state, d_state, bias=config.gru_bias) for _ in range(n_bands)]
        )

        # --- Cross-band attention ---
        self.cross_band_attn = CrossBandAttention(
            d_state=d_state,
            n_bands=n_bands,
            n_heads=config.cross_band_heads,
            dropout=config.dropout,
        )
        self.cross_band_norm = nn.LayerNorm(d_state)

        # --- Per-band FFN ---
        # Each band gets its own small FFN for independent processing after
        # cross-band attention, adding capacity for band-specific learning.
        self.band_ffns = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_state),
                nn.Linear(d_state, d_state * 2),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(d_state * 2, d_state),
                nn.Dropout(config.dropout),
            )
            for _ in range(n_bands)
        ])

        # --- Main FFN (SwiGLU-style gated FFN) ---
        # Operates on the concatenated band outputs projected back to d_model.
        # Uses gated linear unit for better gradient flow and expressivity.
        self.band_to_model = nn.Linear(n_bands * d_state, d_model)
        self.ffn_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.ffn_up = nn.Linear(d_model, d_ffn, bias=False)
        self.ffn_down = nn.Linear(d_ffn, d_model, bias=False)
        self.ffn_dropout = nn.Dropout(config.dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

        # Learned scale factor for band FFN contributions to the carried
        # recurrent state. Initialized to 0.1 to prevent large state updates
        # early in training, which can cause vanishing or exploding gradients
        # in the recurrent path. The network can learn a larger or smaller
        # scale as needed.
        self.band_ffn_state_scale = nn.Parameter(
            torch.tensor(0.1, dtype=torch.float32)
        )

        self.dropout = nn.Dropout(config.dropout)

    def initial_state(
        self, batch_size: int, device: torch.device
    ) -> list[torch.Tensor]:
        """
        Create the initial recurrent state for all bands.

        Returns:
            List of 4 tensors, each shape (batch_size, d_state).
        """
        return [
            gru.initial_state(batch_size, device) for gru in self.band_grus
        ]

    def forward(
        self,
        x: torch.Tensor,
        band_states: list[torch.Tensor],
    ) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass for one time step through this block.

        Args:
            x: Input tensor, shape (batch, d_model).
            band_states: List of 4 tensors, each (batch, d_state).

        Returns:
            output: Transformed input, shape (batch, d_model).
            new_band_states: Updated list of 4 state tensors.
        """
        residual = x
        x_normed = self.input_norm(x)

        # 1. Project input and split into bands
        band_inputs = self.input_proj(x_normed)  # (batch, n_bands * d_state)
        band_inputs = band_inputs.chunk(
            self.config.n_bands, dim=-1
        )  # list of (batch, d_state)

        # 2. Update each band independently with GRU gating
        new_states = []
        band_outputs = []
        for i, (gru, b_input, h_prev) in enumerate(
            zip(self.band_grus, band_inputs, band_states)
        ):
            b_out, h_new = gru(b_input, h_prev)
            band_outputs.append(b_out)
            new_states.append(h_new)

        # 3. Cross-band attention
        stacked = torch.stack(band_outputs, dim=1)  # (batch, n_bands, d_state)
        stacked_normed = self.cross_band_norm(stacked)
        cross_attn_out = self.cross_band_attn(stacked_normed)
        stacked = stacked + self.dropout(cross_attn_out)  # Residual

        # Update states with cross-band information
        for i in range(self.config.n_bands):
            new_states[i] = new_states[i] + self.dropout(cross_attn_out[:, i, :])

        # 3.5. Per-band FFN (with residual) -- no in-place ops
        band_ffn_outputs = []
        for i in range(self.config.n_bands):
            band_i = stacked[:, i, :]
            band_ffn_out = band_i + self.band_ffns[i](band_i)
            band_ffn_outputs.append(band_ffn_out)
            # Also update the carried state (out-of-place), scaled by the
            # learned band_ffn_state_scale parameter.
            new_states[i] = new_states[i] + self.band_ffns[i](new_states[i]) * self.band_ffn_state_scale
        stacked = torch.stack(band_ffn_outputs, dim=1)  # (batch, n_bands, d_state)

        # 4. Project back to d_model
        flat = stacked.reshape(
            stacked.size(0), -1
        )  # (batch, n_bands * d_state)
        x_model = self.band_to_model(flat)  # (batch, d_model)

        # 5. SwiGLU FFN with residual
        x_model = residual + self.dropout(x_model)
        ffn_residual = x_model
        x_normed = self.ffn_norm(x_model)
        gate = F.silu(self.ffn_gate(x_normed))
        up = self.ffn_up(x_normed)
        x_model = ffn_residual + self.ffn_dropout(self.ffn_down(gate * up))

        return x_model, new_states
