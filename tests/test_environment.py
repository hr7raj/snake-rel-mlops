from src.environment import SnakeGame


def test_reset_initializes_snake_and_state():
    env = SnakeGame(grid_size=10)
    state = env.reset()

    assert env.score == 0
    assert env.done is False
    assert len(env.snake) == 3
    assert len(state) == 11


def test_food_does_not_spawn_inside_snake():
    env = SnakeGame(grid_size=10)
    env.reset()

    assert env.food not in env.snake


def test_wall_collision_ends_episode():
    env = SnakeGame(grid_size=5)
    env.reset()
    env.snake = [(2, 0), (2, 1), (2, 2)]
    env.direction = 0

    _, reward, done = env.step(0)

    assert done is True
    assert reward == -10.0


def test_food_reward_increases_score():
    env = SnakeGame(grid_size=5)
    env.reset()
    env.snake = [(2, 2), (2, 3), (2, 4)]
    env.direction = 0
    env.food = (2, 1)

    _, reward, done = env.step(0)

    assert done is False
    assert reward == 10.0
    assert env.score == 1

