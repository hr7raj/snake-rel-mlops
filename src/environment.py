from __future__ import annotations

import random
from dataclasses import dataclass


State = tuple[int, ...]


@dataclass
class SnakeGame:
    """Minimal Snake environment used by the Q-learning agent."""

    grid_size: int = 10

    ACTIONS = [0, 1, 2, 3]  # up, down, left, right
    DX = [0, 0, -1, 1]
    DY = [-1, 1, 0, 0]
    OPPOSITES = {0: 1, 1: 0, 2: 3, 3: 2}

    def reset(self) -> State:
        mid = self.grid_size // 2
        self.snake = [(mid, mid), (mid, mid + 1), (mid, mid + 2)]
        self.direction = 0
        self.score = 0
        self.done = False
        self._place_food()
        return self.state()

    def _place_food(self) -> None:
        empty_cells = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in self.snake
        ]
        self.food = random.choice(empty_cells) if empty_cells else (0, 0)

    def step(self, action: int) -> tuple[State, float, bool]:
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown action: {action}")

        if action != self.OPPOSITES[self.direction]:
            self.direction = action

        head_x, head_y = self.snake[0]
        next_x = head_x + self.DX[self.direction]
        next_y = head_y + self.DY[self.direction]

        if self._is_collision(next_x, next_y):
            self.done = True
            return self.state(), -10.0, True

        self.snake.insert(0, (next_x, next_y))

        if (next_x, next_y) == self.food:
            self.score += 1
            reward = 10.0
            self._place_food()
        else:
            self.snake.pop()
            reward = -0.01

        return self.state(), reward, False

    def state(self) -> State:
        head_x, head_y = self.snake[0]
        direction = self.direction

        def danger(action: int) -> int:
            next_direction = (
                action if action != self.OPPOSITES[direction] else direction
            )
            next_x = head_x + self.DX[next_direction]
            next_y = head_y + self.DY[next_direction]
            return int(self._is_collision(next_x, next_y))

        right_direction = [2, 3, 1, 0][direction]
        left_direction = [3, 2, 0, 1][direction]
        food_x, food_y = self.food

        return (
            danger(direction),
            danger(right_direction),
            danger(left_direction),
            int(direction == 0),
            int(direction == 1),
            int(direction == 2),
            int(direction == 3),
            int(food_y < head_y),
            int(food_y > head_y),
            int(food_x < head_x),
            int(food_x > head_x),
        )

    def _is_collision(self, x: int, y: int) -> bool:
        hits_wall = not (0 <= x < self.grid_size and 0 <= y < self.grid_size)
        hits_body = (x, y) in self.snake[:-1]
        return hits_wall or hits_body

