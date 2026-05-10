from __future__ import annotations

import argparse
import random

import numpy as np

from src.agent import QLearningAgent
from src.artifacts import plot_results, save_metrics, save_monitoring_log, summarize_scores
from src.config import TrainingConfig, load_config
from src.environment import SnakeGame


def train(config: TrainingConfig) -> tuple[QLearningAgent, list[int], list[float], list[float]]:
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    env = SnakeGame(grid_size=config.grid_size)
    agent = QLearningAgent(
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon=config.epsilon,
        epsilon_min=config.epsilon_min,
        epsilon_decay=config.epsilon_decay,
    )
    scores: list[int] = []
    rewards: list[float] = []
    avg_scores: list[float] = []

    for episode in range(1, config.episodes + 1):
        state = env.reset()
        total_reward = 0.0

        for _ in range(config.max_steps_per_episode):
            action = agent.act(state, training=True)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        agent.decay_exploration()
        scores.append(env.score)
        rewards.append(total_reward)
        avg_scores.append(float(np.mean(scores[-50:])))

        if episode % 50 == 0 or episode == config.episodes:
            print(
                f"Episode {episode:4d} | Score {env.score:3d} | "
                f"Avg(50) {avg_scores[-1]:.2f} | epsilon {agent.epsilon:.3f}"
            )

    return agent, scores, rewards, avg_scores


def run_training(config: TrainingConfig) -> dict:
    agent, scores, rewards, avg_scores = train(config)
    metrics = summarize_scores(scores, rewards, config)

    agent.save(config.model_path)
    save_metrics(metrics, config.metrics_path)
    save_monitoring_log(scores, rewards, config.monitoring_path)
    plot_results(scores, avg_scores, config.plot_path)

    print(f"Saved model: {config.model_path}")
    print(f"Saved metrics: {config.metrics_path}")
    print(f"Saved plot: {config.plot_path}")
    print(f"Saved monitoring log: {config.monitoring_path}")
    print(f"Best score: {metrics['best_score']}")
    print(f"Average score last 50: {metrics['average_score_last_50']:.2f}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Snake Q-learning agent.")
    parser.add_argument("--config", default="config/default.json", help="Path to JSON config file.")
    parser.add_argument("--episodes", type=int, help="Override number of training episodes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.episodes is not None:
        config = TrainingConfig(**{**config.__dict__, "episodes": args.episodes})
    run_training(config)


if __name__ == "__main__":
    main()

