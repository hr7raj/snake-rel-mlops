"""
Compatibility entry point for the original RL submission.

For the MLOps workflow, prefer:
    python -m src.train --config config/default.json
    python -m src.evaluate --config config/default.json
"""

from src.config import load_config
from src.train import run_training


if __name__ == "__main__":
    run_training(load_config("config/default.json"))

