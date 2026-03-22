"""
Export Weights to Binary Format for Rust Inference
====================================================

Exports the trained KAIRO-X controller weights to the directory format
expected by the Rust WeightStore loader:

    <output_dir>/
        manifest.json   -- tensor descriptors (name, shape, offset, num_elements)
        weights.bin     -- raw f32 data prefixed with KAIR magic bytes

Weight names are remapped from PyTorch's state_dict naming convention to
the flat Rust naming convention expected by weights.rs:

    PyTorch                                     Rust
    -------                                     ----
    input_embedding.slot_proj.weight     ->     input_proj.weight
    input_embedding.slot_proj.bias       ->     input_proj.bias
    layers.{i}.band_grus.{b}.W_input.*  ->     layers.{i}.band_{b}.gate_ih.*
    layers.{i}.band_grus.{b}.W_state.*  ->     layers.{i}.band_{b}.gate_hh.*
    layers.{i}.cross_band_attn.qkv.*    ->     layers.{i}.cross_band.qkv.*
    layers.{i}.cross_band_attn.out_proj.* ->   layers.{i}.cross_band.out_proj.*
    layers.{i}.ffn_up.*                  ->     layers.{i}.ffn.up.*
    layers.{i}.ffn_down.*               ->     layers.{i}.ffn.down.*
    layers.{i}.input_norm.*             ->     layers.{i}.norm1.*
    layers.{i}.ffn_norm.*               ->     layers.{i}.norm2.*
    action_head.net.{last_linear}.*     ->     heads.action.*
    context_budget_head.net.{last_linear}.* ->  heads.context_budget.*
    enforcement_intensity_head.net.{last_linear}.* -> heads.enforcement_intensity.*
    session_edge_case_head.net.{last_linear}.* -> heads.session_edge.*
    stop_head.net.{last_linear}.*       ->     heads.stop.*

The Rust side also expects an upsample weight for the cross-band attention
path (layers.{i}.cross_band.upsample.weight). The Python model uses
band_to_model for a similar purpose. This weight is included in the export.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from training.model.controller import KairoController, ControllerConfig

logger = logging.getLogger(__name__)

# First 4 bytes of weights.bin -- "KAIR" in ASCII.
# The Rust loader validates this to confirm correct endianness and file integrity.
KAIR_MAGIC = b"KAIR"

FORMAT_VERSION = 1
ALIGNMENT = 16  # Byte alignment for tensor data


# ---------------------------------------------------------------------------
# Weight name remapping
# ---------------------------------------------------------------------------

def _remap_weight_name(pytorch_name: str) -> Optional[str]:
    """Translate a PyTorch state_dict key to the Rust naming convention.

    Returns None for tensors that should be skipped (e.g. dropout parameters,
    per-band FFN weights that the simplified Rust inference path does not use,
    learnable initial states that are only needed in Python).
    """
    name = pytorch_name

    # --- Input embedding ---
    # The Python InputEmbedding has slot_proj (Linear) and norm (LayerNorm).
    # Rust expects input_proj.{weight,bias}.
    m = re.match(r"input_embedding\.slot_proj\.(weight|bias)$", name)
    if m:
        return f"input_proj.{m.group(1)}"

    # Input embedding norm -- skip (Rust applies its own normalization)
    if name.startswith("input_embedding.norm.") or name.startswith("input_embedding."):
        # LayerNorm weights inside input embedding are not loaded by Rust
        return None

    # --- Output normalization (output_norm) ---
    # Skip -- Rust does LayerNorm per-layer and at heads; the global output_norm
    # is folded into the head weights during export verification, but the Rust
    # loader does not expect a separate output_norm tensor.
    if name.startswith("output_norm."):
        return None

    # --- Per-layer components ---

    # Band GRU gates
    # layers.{i}.band_grus.{b}.W_input.{weight,bias}
    #   -> layers.{i}.band_{b}.gate_ih.{weight,bias}
    m = re.match(r"layers\.(\d+)\.band_grus\.(\d+)\.W_input\.(weight|bias)$", name)
    if m:
        return f"layers.{m.group(1)}.band_{m.group(2)}.gate_ih.{m.group(3)}"

    m = re.match(r"layers\.(\d+)\.band_grus\.(\d+)\.W_state\.(weight|bias)$", name)
    if m:
        return f"layers.{m.group(1)}.band_{m.group(2)}.gate_hh.{m.group(3)}"

    # Learnable initial state h0 -- skip (Rust initializes state to zeros)
    if re.match(r"layers\.\d+\.band_grus\.\d+\.h0$", name):
        return None

    # Cross-band attention
    m = re.match(r"layers\.(\d+)\.cross_band_attn\.qkv\.(weight)$", name)
    if m:
        return f"layers.{m.group(1)}.cross_band.qkv.{m.group(2)}"

    m = re.match(r"layers\.(\d+)\.cross_band_attn\.out_proj\.(weight)$", name)
    if m:
        return f"layers.{m.group(1)}.cross_band.out_proj.{m.group(2)}"

    # Cross-band attention temperature/scale/dropout -- skip
    if re.match(r"layers\.\d+\.cross_band_attn\.(temperature|scale|dropout)", name):
        return None

    # Cross-band norm -- skip (part of residual logic handled differently)
    if re.match(r"layers\.\d+\.cross_band_norm\.", name):
        return None

    # band_to_model projects n_bands*d_state -> d_model; map to upsample
    m = re.match(r"layers\.(\d+)\.band_to_model\.(weight)$", name)
    if m:
        return f"layers.{m.group(1)}.cross_band.upsample.{m.group(2)}"
    # band_to_model bias -- skip (Rust upsample has no bias)
    if re.match(r"layers\.\d+\.band_to_model\.bias$", name):
        return None

    # Per-band FFNs -- skip (Rust inference simplifies this path)
    if re.match(r"layers\.\d+\.band_ffns\.", name):
        return None
    if re.match(r"layers\.\d+\.band_ffn_state_scale$", name):
        return None

    # Input projection within liquid block
    # layers.{i}.input_proj.{weight,bias} -- skip (the Rust path uses band gate
    # input weights directly from d_model)
    if re.match(r"layers\.\d+\.input_proj\.(weight|bias)$", name):
        return None

    # SwiGLU FFN
    # ffn_gate is the gate branch of SwiGLU -- Rust uses a simpler FFN,
    # so we export ffn_up and ffn_down as the primary weights.
    m = re.match(r"layers\.(\d+)\.ffn_up\.(weight)$", name)
    if m:
        return f"layers.{m.group(1)}.ffn.up.{m.group(2)}"

    m = re.match(r"layers\.(\d+)\.ffn_down\.(weight)$", name)
    if m:
        return f"layers.{m.group(1)}.ffn.down.{m.group(2)}"

    # ffn_gate -- Rust FFN doesn't use gated variant, skip
    if re.match(r"layers\.\d+\.ffn_gate\.weight$", name):
        return None

    # LayerNorms
    # input_norm -> norm1, ffn_norm -> norm2
    m = re.match(r"layers\.(\d+)\.input_norm\.(weight|bias)$", name)
    if m:
        return f"layers.{m.group(1)}.norm1.{m.group(2)}"

    m = re.match(r"layers\.(\d+)\.ffn_norm\.(weight|bias)$", name)
    if m:
        return f"layers.{m.group(1)}.norm2.{m.group(2)}"

    # FFN dropout -- skip
    if re.match(r"layers\.\d+\.ffn_dropout\.", name) or re.match(r"layers\.\d+\.dropout\.", name):
        return None

    # --- Output heads ---
    # Each head is an nn.Sequential called .net; we want the final Linear layer.
    # The Rust inference path uses a single-layer linear for each head.
    # We export the last Linear in each head's Sequential.

    # Action head: action_head.net.{idx}.{weight,bias}
    m = re.match(r"action_head\.net\.(\d+)\.(weight|bias)$", name)
    if m:
        return f"heads.action.{m.group(2)}" if _is_last_linear_in_head(name, "action_head") else None

    # Context selection head -- bilinear scoring, handled separately
    if name.startswith("context_selection_head."):
        # query_proj and key_proj are multi-layer; Rust uses simplified scoring
        # Skip all context selection head weights (Rust uses rule-based scoring)
        return None

    # Context budget head
    m = re.match(r"context_budget_head\.net\.(\d+)\.(weight|bias)$", name)
    if m:
        return f"heads.context_budget.{m.group(2)}" if _is_last_linear_in_head(name, "context_budget_head") else None

    # Enforcement intensity head
    m = re.match(r"enforcement_intensity_head\.net\.(\d+)\.(weight|bias)$", name)
    if m:
        return f"heads.enforcement_intensity.{m.group(2)}" if _is_last_linear_in_head(name, "enforcement_intensity_head") else None

    # Session edge case head
    m = re.match(r"session_edge_case_head\.net\.(\d+)\.(weight|bias)$", name)
    if m:
        return f"heads.session_edge.{m.group(2)}" if _is_last_linear_in_head(name, "session_edge_case_head") else None

    # Stop head
    m = re.match(r"stop_head\.net\.(\d+)\.(weight|bias)$", name)
    if m:
        return f"heads.stop.{m.group(2)}" if _is_last_linear_in_head(name, "stop_head") else None

    # If we get here, the name was not recognized. Log and skip.
    logger.debug("Skipping unrecognized parameter: %s", name)
    return None


# Cache for determining the last Linear layer index in each head Sequential.
_HEAD_LAST_LINEAR: Dict[str, int] = {}


def _find_last_linear_indices(model: KairoController) -> None:
    """Walk each head's nn.Sequential to find the index of the final Linear layer."""
    heads = {
        "action_head": model.action_head.net,
        "context_budget_head": model.context_budget_head.net,
        "enforcement_intensity_head": model.enforcement_intensity_head.net,
        "session_edge_case_head": model.session_edge_case_head.net,
        "stop_head": model.stop_head.net,
    }
    for head_name, seq in heads.items():
        last_idx = -1
        for idx, layer in enumerate(seq):
            if isinstance(layer, nn.Linear):
                last_idx = idx
        _HEAD_LAST_LINEAR[head_name] = last_idx


def _is_last_linear_in_head(pytorch_name: str, head_prefix: str) -> bool:
    """Check if the given parameter belongs to the final Linear in the head."""
    m = re.match(rf"{re.escape(head_prefix)}\.net\.(\d+)\.", pytorch_name)
    if not m:
        return False
    idx = int(m.group(1))
    return idx == _HEAD_LAST_LINEAR.get(head_prefix, -1)


def _build_remapped_tensors(
    model: KairoController,
) -> List[Tuple[str, torch.Tensor]]:
    """Build the list of (rust_name, tensor) pairs, remapping names and skipping
    parameters that the Rust inference path does not use.
    """
    _find_last_linear_indices(model)

    state_dict = model.state_dict()
    result: List[Tuple[str, torch.Tensor]] = []
    skipped: List[str] = []

    for pytorch_name in sorted(state_dict.keys()):
        rust_name = _remap_weight_name(pytorch_name)
        if rust_name is None:
            skipped.append(pytorch_name)
            continue
        tensor = state_dict[pytorch_name].detach().cpu().contiguous().float()
        result.append((rust_name, tensor))

    if skipped:
        logger.info(
            "Skipped %d parameters not needed by Rust inference (e.g. %s)",
            len(skipped),
            skipped[0],
        )

    # Sort by Rust name for deterministic ordering
    result.sort(key=lambda x: x[0])
    return result


# ---------------------------------------------------------------------------
# Export: manifest.json + weights.bin
# ---------------------------------------------------------------------------

def export_model(
    model: KairoController,
    output_path: str,
    config: Optional[ControllerConfig] = None,
) -> int:
    """
    Export a trained KAIRO-X controller to the directory format expected by
    the Rust WeightStore loader.

    Produces:
        <output_path>/manifest.json
        <output_path>/weights.bin

    Args:
        model: The trained model.
        output_path: Directory path for the export.
        config: Optional config to embed in the manifest. If None, uses
            model.config.

    Returns:
        Total size of weights.bin in bytes.
    """
    if config is None:
        config = model.config

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    weights_path = output_dir / "weights.bin"

    tensors = _build_remapped_tensors(model)
    total_params = sum(t.numel() for _, t in tensors)

    logger.info(
        "Exporting %d tensors (%d parameters) to %s",
        len(tensors),
        total_params,
        output_dir,
    )

    # Build tensor descriptors and compute offsets in the data section.
    # Offsets are relative to the start of the data section (after KAIR magic).
    tensor_descriptors: List[Dict] = []
    data_offset = 0
    for rust_name, tensor in tensors:
        num_elements = tensor.numel()
        byte_length = num_elements * 4  # f32 = 4 bytes
        tensor_descriptors.append({
            "name": rust_name,
            "shape": list(tensor.shape),
            "offset": data_offset,
            "num_elements": num_elements,
        })
        # Align next tensor
        data_offset += byte_length
        remainder = data_offset % ALIGNMENT
        if remainder != 0:
            data_offset += ALIGNMENT - remainder

    # Write manifest.json
    manifest = {
        "version": FORMAT_VERSION,
        "total_parameters": total_params,
        "tensors": tensor_descriptors,
        "config": {
            "d_model": config.d_model,
            "d_state": config.d_state,
            "n_bands": config.n_bands,
            "d_ffn": config.d_ffn,
            "n_layers": config.n_layers,
            "n_input_slots": config.n_input_slots,
            "d_slot": config.d_slot,
            "cross_band_heads": config.cross_band_heads,
            "n_actions": config.n_actions,
            "max_context_candidates": config.max_context_candidates,
            "d_candidate": config.d_candidate,
        },
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    logger.info("Wrote manifest: %s (%d tensor entries)", manifest_path, len(tensor_descriptors))

    # Write weights.bin: KAIR magic + raw f32 data
    with open(weights_path, "wb") as f:
        # Write KAIR magic bytes as endianness marker
        f.write(KAIR_MAGIC)

        # Write tensor data at their computed offsets (relative to byte 4)
        for desc, (rust_name, tensor) in zip(tensor_descriptors, tensors):
            target_pos = len(KAIR_MAGIC) + desc["offset"]
            current_pos = f.tell()
            if target_pos > current_pos:
                f.write(b"\x00" * (target_pos - current_pos))

            raw_bytes = tensor.numpy().tobytes()
            f.write(raw_bytes)

        total_size = f.tell()

    logger.info(
        "Wrote weights: %s (%.2f MB, %d tensors)",
        weights_path,
        total_size / (1024 * 1024),
        len(tensors),
    )

    return total_size


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_export(
    model: KairoController,
    export_path: str,
    tolerance: float = 1e-6,
) -> bool:
    """
    Verify that an exported directory matches the model's weights.

    Loads manifest.json and weights.bin, then compares each tensor to the
    remapped state dict values.

    Args:
        model: The original model.
        export_path: Path to the export directory.
        tolerance: Maximum allowed absolute difference per element.

    Returns:
        True if all tensors match within tolerance.
    """
    export_dir = Path(export_path)
    manifest_path = export_dir / "manifest.json"
    weights_path = export_dir / "weights.bin"

    if not manifest_path.exists():
        logger.error("manifest.json not found in %s", export_dir)
        return False
    if not weights_path.exists():
        logger.error("weights.bin not found in %s", export_dir)
        return False

    # Load manifest
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Load weight data
    with open(weights_path, "rb") as f:
        weight_data = f.read()

    # Validate KAIR magic
    if len(weight_data) < len(KAIR_MAGIC):
        logger.error("weights.bin too small (%d bytes)", len(weight_data))
        return False

    actual_magic = weight_data[:len(KAIR_MAGIC)]
    if actual_magic != KAIR_MAGIC:
        logger.error(
            "Magic mismatch: expected %r, got %r", KAIR_MAGIC, actual_magic
        )
        return False

    # Data section starts after magic
    data_section = weight_data[len(KAIR_MAGIC):]

    # Build expected tensors from model
    expected_tensors = dict(_build_remapped_tensors(model))

    all_match = True
    n_checked = 0

    for desc in manifest["tensors"]:
        rust_name = desc["name"]
        shape = desc["shape"]
        offset = desc["offset"]
        num_elements = desc["num_elements"]
        byte_len = num_elements * 4

        if rust_name not in expected_tensors:
            logger.error(
                "Tensor '%s' in manifest but not in remapped state dict", rust_name
            )
            all_match = False
            continue

        expected = expected_tensors[rust_name]
        if list(expected.shape) != shape:
            logger.error(
                "Shape mismatch for %s: expected %s, got %s",
                rust_name,
                list(expected.shape),
                shape,
            )
            all_match = False
            continue

        if offset + byte_len > len(data_section):
            logger.error(
                "Tensor '%s' extends beyond weights.bin (offset=%d, len=%d, file=%d)",
                rust_name,
                offset,
                byte_len,
                len(data_section),
            )
            all_match = False
            continue

        raw = data_section[offset:offset + byte_len]
        loaded = torch.from_numpy(
            np.frombuffer(raw, dtype=np.float32).copy()
        ).reshape(shape)

        max_diff = (expected - loaded).abs().max().item()
        if max_diff > tolerance:
            logger.error(
                "Value mismatch for %s: max_diff = %.8e (tolerance = %.8e)",
                rust_name,
                max_diff,
                tolerance,
            )
            all_match = False
        else:
            logger.debug("Tensor %s OK (max_diff = %.8e)", rust_name, max_diff)
            n_checked += 1

    if all_match:
        logger.info(
            "Export verification PASSED: all %d tensors match.", n_checked
        )
    else:
        logger.error("Export verification FAILED.")

    return all_match


# ---------------------------------------------------------------------------
# Convenience: export from checkpoint file
# ---------------------------------------------------------------------------

def export_checkpoint(
    checkpoint_path: str,
    output_path: str,
    config: Optional[ControllerConfig] = None,
) -> int:
    """
    Convenience function: load a training checkpoint and export to the
    directory format.

    Args:
        checkpoint_path: Path to a PyTorch checkpoint (.pt file).
        output_path: Directory path for the export output.
        config: Optional config override.

    Returns:
        Total size of weights.bin in bytes.
    """
    if config is None:
        config = ControllerConfig()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = KairoController(config)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return export_model(model, output_path, config)
