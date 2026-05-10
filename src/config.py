from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    grid_size: int = 10
    cell_size: int = 40
    fps: int = 15
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    episodes: int = 500
    max_steps_per_episode: int = 500
    random_seed: int = 42
    model_path: str = "models/q_table.pkl"
    metrics_path: str = "metrics/training_metrics.json"
    plot_path: str = "reports/results.png"
    monitoring_path: str = "monitoring/performance_log.csv"


def load_config(path: str | Path | None = None) -> TrainingConfig:
    if path is None:
        return TrainingConfig()

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    defaults = asdict(TrainingConfig())
    defaults.update(raw)
    return TrainingConfig(**defaults)


def save_config(config: TrainingConfig, path: str | Path) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

