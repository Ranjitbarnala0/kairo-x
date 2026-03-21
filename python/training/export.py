"""
Export Weights to Binary Format for Rust Inference
====================================================

Exports the trained KAIRO-X controller weights to a binary format that
the Rust inference engine can load directly.

Binary format:
    Header (variable length):
        - Magic bytes: b"KAIROX01" (8 bytes)
        - Version: u32 (4 bytes)
        - Number of tensors: u32 (4 bytes)
        - Config JSON length: u32 (4 bytes)
        - Config JSON: UTF-8 bytes
        - Tensor table: for each tensor:
            - Name length: u32 (4 bytes)
            - Name: UTF-8 bytes
            - Number of dimensions: u32 (4 bytes)
            - Shape: [u32] * n_dims (4 * n_dims bytes)
            - Data offset: u64 (8 bytes) -- offset from start of data section
            - Data length: u64 (8 bytes) -- length in bytes

    Data section:
        - Raw f32 arrays, contiguous, aligned to 16 bytes
        - Order matches the tensor table
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from training.model.controller import KairoController, ControllerConfig

logger = logging.getLogger(__name__)

MAGIC = b"KAIROX01"
FORMAT_VERSION = 1
ALIGNMENT = 16  # Byte alignment for data section


@dataclass
class TensorEntry:
    """Metadata for a single tensor in the binary file."""

    name: str
    shape: List[int]
    data_offset: int = 0
    data_length: int = 0


def _align_offset(offset: int, alignment: int = ALIGNMENT) -> int:
    """Round up offset to the next alignment boundary."""
    remainder = offset % alignment
    if remainder == 0:
        return offset
    return offset + (alignment - remainder)


def _flatten_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> List[Tuple[str, torch.Tensor]]:
    """
    Flatten a state dict into a sorted list of (name, tensor) pairs.
    All tensors are converted to contiguous f32 on CPU.
    """
    items = []
    for name in sorted(state_dict.keys()):
        tensor = state_dict[name].detach().cpu().contiguous().float()
        items.append((name, tensor))
    return items


def export_model(
    model: KairoController,
    output_path: str,
    config: Optional[ControllerConfig] = None,
) -> int:
    """
    Export a trained KAIRO-X controller to binary format.

    Args:
        model: The trained model.
        output_path: Path to write the binary file.
        config: Optional config to embed in the header. If None, uses
            model.config.

    Returns:
        Total file size in bytes.
    """
    if config is None:
        config = model.config

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    tensors = _flatten_state_dict(state_dict)

    logger.info(
        "Exporting %d tensors (%d parameters) to %s",
        len(tensors),
        sum(t.numel() for _, t in tensors),
        output_path,
    )

    # Serialize config
    config_dict = {
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
    }
    config_json = json.dumps(config_dict, indent=2).encode("utf-8")

    # Build tensor table entries (offsets computed relative to data start)
    entries: List[TensorEntry] = []
    data_offset = 0
    for name, tensor in tensors:
        data_length = tensor.numel() * 4  # f32 = 4 bytes
        entries.append(
            TensorEntry(
                name=name,
                shape=list(tensor.shape),
                data_offset=data_offset,
                data_length=data_length,
            )
        )
        data_offset = _align_offset(data_offset + data_length)

    with open(output_path, "wb") as f:
        # --- Write header ---
        f.write(MAGIC)
        f.write(struct.pack("<I", FORMAT_VERSION))
        f.write(struct.pack("<I", len(tensors)))
        f.write(struct.pack("<I", len(config_json)))
        f.write(config_json)

        # --- Write tensor table ---
        for entry in entries:
            name_bytes = entry.name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", len(entry.shape)))
            for dim in entry.shape:
                f.write(struct.pack("<I", dim))
            f.write(struct.pack("<Q", entry.data_offset))
            f.write(struct.pack("<Q", entry.data_length))

        # --- Align to data section start ---
        current = f.tell()
        aligned = _align_offset(current)
        if aligned > current:
            f.write(b"\x00" * (aligned - current))

        data_section_start = f.tell()

        # --- Write tensor data ---
        for entry, (name, tensor) in zip(entries, tensors):
            # Seek to the correct offset within the data section
            target_pos = data_section_start + entry.data_offset
            current_pos = f.tell()
            if target_pos > current_pos:
                f.write(b"\x00" * (target_pos - current_pos))

            # Write raw f32 bytes
            raw_bytes = tensor.numpy().tobytes()
            f.write(raw_bytes)

        total_size = f.tell()

    logger.info(
        "Exported model: %d tensors, %.2f MB total",
        len(tensors),
        total_size / (1024 * 1024),
    )

    return total_size


def verify_export(
    model: KairoController,
    export_path: str,
    tolerance: float = 1e-6,
) -> bool:
    """
    Verify that an exported binary file matches the model's weights.

    Loads the binary file and compares each tensor to the model's state dict.

    Args:
        model: The original model.
        export_path: Path to the exported binary file.
        tolerance: Maximum allowed absolute difference per element.

    Returns:
        True if all tensors match within tolerance.
    """
    export_path = Path(export_path)
    state_dict = model.state_dict()

    with open(export_path, "rb") as f:
        # Read header
        magic = f.read(8)
        assert magic == MAGIC, f"Bad magic: {magic!r}"

        version = struct.unpack("<I", f.read(4))[0]
        assert version == FORMAT_VERSION, f"Bad version: {version}"

        n_tensors = struct.unpack("<I", f.read(4))[0]
        config_len = struct.unpack("<I", f.read(4))[0]
        _config_json = f.read(config_len)  # Skip config

        # Read tensor table
        entries = []
        for _ in range(n_tensors):
            name_len = struct.unpack("<I", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")
            n_dims = struct.unpack("<I", f.read(4))[0]
            shape = [struct.unpack("<I", f.read(4))[0] for _ in range(n_dims)]
            data_offset = struct.unpack("<Q", f.read(8))[0]
            data_length = struct.unpack("<Q", f.read(8))[0]
            entries.append(TensorEntry(name, shape, data_offset, data_length))

        # Align to data section
        current = f.tell()
        aligned = _align_offset(current)
        f.seek(aligned)
        data_section_start = f.tell()

        # Verify each tensor
        all_match = True
        for entry in entries:
            if entry.name not in state_dict:
                logger.error("Tensor %s not found in model state dict", entry.name)
                all_match = False
                continue

            expected = state_dict[entry.name].detach().cpu().float()
            if list(expected.shape) != entry.shape:
                logger.error(
                    "Shape mismatch for %s: expected %s, got %s",
                    entry.name,
                    list(expected.shape),
                    entry.shape,
                )
                all_match = False
                continue

            f.seek(data_section_start + entry.data_offset)
            raw = f.read(entry.data_length)
            import numpy as np

            loaded = torch.from_numpy(
                np.frombuffer(raw, dtype=np.float32).copy()
            ).reshape(entry.shape)

            max_diff = (expected - loaded).abs().max().item()
            if max_diff > tolerance:
                logger.error(
                    "Value mismatch for %s: max_diff = %.8e (tolerance = %.8e)",
                    entry.name,
                    max_diff,
                    tolerance,
                )
                all_match = False
            else:
                logger.debug(
                    "Tensor %s OK (max_diff = %.8e)", entry.name, max_diff
                )

    if all_match:
        logger.info("Export verification PASSED: all %d tensors match.", n_tensors)
    else:
        logger.error("Export verification FAILED.")

    return all_match


def export_checkpoint(
    checkpoint_path: str,
    output_path: str,
    config: Optional[ControllerConfig] = None,
) -> int:
    """
    Convenience function: load a training checkpoint and export to binary.

    Args:
        checkpoint_path: Path to a PyTorch checkpoint (.pt file).
        output_path: Path for the binary output.
        config: Optional config override.

    Returns:
        Total file size in bytes.
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
