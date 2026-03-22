"""
Entry point for running the KAIRO-X training pipeline as a module.

Usage:
    python -m training
    python -m training --synthetic-size 1000 --batch-size 32 --device cpu
    python -m training --resume outputs/kairo_training/checkpoints/checkpoint_latest.pt
"""

from training.trainer import main

main()
