"""
Entry point for running the KAIRO-X training pipeline as a module.

Usage:
    python -m training.trainer
    python -m training.trainer --synthetic-size 1000 --batch-size 32 --device cpu
    python -m training.trainer --resume outputs/kairo_training/checkpoints/checkpoint_latest.pt
"""

from training.trainer import main

main()
