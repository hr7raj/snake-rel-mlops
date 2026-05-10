from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import TrainingConfig


def summarize_scores(scores: list[int], rewards: list[float], config: TrainingConfig) -> dict:
    last_window = scores[-50:] if scores else []
    return {
        "episodes": len(scores),
        "best_score": int(max(scores)) if scores else 0,
        "average_score": float(np.mean(scores)) if scores else 0.0,
        "average_score_last_50": float(np.mean(last_window)) if last_window else 0.0,
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "config": asdict(config),
    }


def save_metrics(metrics: dict, path: str | Path) -> None:
    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def save_monitoring_log(scores: list[int], rewards: list[float], path: str | Path) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode", "score", "reward", "rolling_avg_50"])
        writer.writeheader()
        for index, (score, reward) in enumerate(zip(scores, rewards), start=1):
            rolling_avg = float(np.mean(scores[max(0, index - 50) : index]))
            writer.writerow(
                {
                    "episode": index,
                    "score": score,
                    "reward": round(reward, 4),
                    "rolling_avg_50": round(rolling_avg, 4),
                }
            )


def plot_results(scores: list[int], avg_scores: list[float], path: str | Path) -> None:
    plot_path = Path(path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0f0f14")

    for ax in axes:
        ax.set_facecolor("#1a1a24")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#333333")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color("white")

    axes[0].plot(scores, color="#50c878", alpha=0.4, linewidth=0.8, label="Score")
    axes[0].plot(avg_scores, color="#ff6b6b", linewidth=2, label="Avg (50 ep)")
    axes[0].set_title("Score per Episode", color="white", fontsize=13)
    axes[0].set_xlabel("Episode", color="white")
    axes[0].set_ylabel("Score", color="white")
    axes[0].legend(facecolor="#1a1a24", labelcolor="white")

    axes[1].plot(avg_scores, color="#50c878", linewidth=2)
    axes[1].fill_between(range(len(avg_scores)), avg_scores, alpha=0.2, color="#50c878")
    axes[1].set_title("Learning Curve (50-ep Rolling Avg)", color="white", fontsize=13)
    axes[1].set_xlabel("Episode", color="white")
    axes[1].set_ylabel("Avg Score", color="white")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
