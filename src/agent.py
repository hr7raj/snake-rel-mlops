from __future__ import annotations

import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.environment import State


class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: defaultdict[State, np.ndarray] = defaultdict(
            lambda: np.zeros(4, dtype=float)
        )

    def act(self, state: State, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, 3)
        return int(np.argmax(self.q_table[state]))

    def learn(self, state: State, action: int, reward: float, next_state: State, done: bool) -> None:
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])

    def decay_exploration(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {state: values for state, values in self.q_table.items()}
        with model_path.open("wb") as handle:
            pickle.dump(
                {
                    "q_table": serializable,
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "epsilon_min": self.epsilon_min,
                    "epsilon_decay": self.epsilon_decay,
                },
                handle,
            )

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)

        agent = cls(
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            epsilon_min=payload["epsilon_min"],
            epsilon_decay=payload["epsilon_decay"],
        )
        agent.q_table.update(payload["q_table"])
        return agent

