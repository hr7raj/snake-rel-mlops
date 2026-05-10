from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from src.agent import QLearningAgent
from src.config import load_config
from src.environment import SnakeGame


def evaluate(model_path: str, grid_size: int, episodes: int = 20, max_steps: int = 500) -> dict:
    agent = QLearningAgent.load(model_path)
    env = SnakeGame(grid_size=grid_size)
    scores: list[int] = []

    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            action = agent.act(state, training=False)
            state, _, done = env.step(action)
            if done:
                break
        scores.append(env.score)

    return {
        "episodes": episodes,
        "best_score": int(max(scores)) if scores else 0,
        "average_score": float(np.mean(scores)) if scores else 0.0,
        "scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Snake Q-table.")
    parser.add_argument("--config", default="config/default.json")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", default="metrics/evaluation_metrics.json")
    args = parser.parse_args()

    config = load_config(args.config)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    metrics = evaluate(
        model_path=config.model_path,
        grid_size=config.grid_size,
        episodes=args.episodes,
        max_steps=config.max_steps_per_episode,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

