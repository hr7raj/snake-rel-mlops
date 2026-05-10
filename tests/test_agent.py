import numpy as np

from src.agent import QLearningAgent


def test_learning_updates_q_value():
    agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon=0.0)
    state = (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)
    next_state = (0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0)

    agent.learn(state, action=0, reward=10.0, next_state=next_state, done=False)

    assert agent.q_table[state][0] > 0


def test_agent_save_and_load_round_trip(tmp_path):
    model_path = tmp_path / "q_table.pkl"
    state = (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)
    agent = QLearningAgent(epsilon=0.25)
    agent.q_table[state] = np.array([1.0, 2.0, 3.0, 4.0])

    agent.save(model_path)
    loaded = QLearningAgent.load(model_path)

    assert loaded.epsilon == 0.25
    assert np.array_equal(loaded.q_table[state], np.array([1.0, 2.0, 3.0, 4.0]))

