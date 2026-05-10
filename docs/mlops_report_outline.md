# MLOps Report Outline: Snake Q-Learning Agent

## 1. Introduction

This project extends a reinforcement learning Snake game agent into an MLOps-enabled system. The original agent learns to play Snake using tabular Q-learning. The MLOps version focuses on making the project reproducible, testable, containerized, automatically verifiable, and monitorable.

## 1.1 Background of the Study

Reinforcement learning trains an agent through interaction with an environment. In this project, the Snake game acts as the environment, and the Q-learning algorithm updates a Q-table based on rewards received after each action. MLOps practices are used to manage the complete lifecycle of this model, from training to evaluation and monitoring.

## 1.2 Problem Statement

The initial Snake Q-learning project worked as a standalone script, but it lacked a structured workflow for reproducibility, testing, deployment, artifact management, and monitoring. The problem is to convert this RL experiment into a maintainable MLOps pipeline.

## 1.3 Objectives of the Project

- Refactor the Snake RL code into reusable modules.
- Store hyperparameters in a configuration file.
- Save trained model artifacts and training metrics.
- Add automated unit and integration tests.
- Add a CI/CD workflow using GitHub Actions.
- Containerize the project using Docker.
- Monitor the trained agent using rolling performance metrics.

## 1.4 Significance / Motivation

MLOps helps ensure that machine learning projects can be reliably reproduced, tested, and deployed. Applying MLOps to an RL project demonstrates how even a game-based learning agent can be managed like a production ML system.

## 1.5 Methodology Overview

The agent interacts with the Snake environment and updates a Q-table using Q-learning. Training scores, rewards, plots, and the Q-table are saved as artifacts. Automated tests validate the environment, agent, and training pipeline. GitHub Actions runs the CI/CD workflow, and Docker provides a reproducible runtime.

## 2. System Architecture

High-level flow:

```text
Config -> Training Pipeline -> Q-learning Agent -> Model Artifact
                   |                  |
                   v                  v
              Metrics/Plots      Evaluation Pipeline
                   |
                   v
             Monitoring Log -> Threshold Check
```

## 2.1 High Level Design

The project is divided into environment, agent, training, evaluation, monitoring, and testing components. Each component has a single responsibility, making the system easier to maintain and automate.

## 2.2 Low Level Design

- `src/environment.py`: Snake environment, state generation, rewards, collision handling.
- `src/agent.py`: Q-learning agent, action selection, Q-value update, model save/load.
- `src/train.py`: Training loop and artifact generation.
- `src/evaluate.py`: Runs trained policy without exploration.
- `src/monitor.py`: Checks rolling average score against a threshold.
- `tests/`: Unit and integration tests.

## 3. Modern Tools Usage

## 3.1 Hardware Requirements

The project can run on a standard laptop or desktop CPU. No GPU is required because tabular Q-learning is lightweight.

## 3.2 Software Requirements

- Python 3.12
- NumPy
- Matplotlib
- Pytest
- Ruff
- Docker
- GitHub Actions

## 4. Implementation & Testing

## 4.1 Data Collection and Preprocessing

Unlike supervised ML, this RL project does not use a static dataset. Data is generated dynamically as the agent interacts with the Snake environment. Each transition consists of state, action, reward, next state, and done flag.

## 4.2 Model Development and Evaluation

The model is a Q-table. The agent uses epsilon-greedy exploration during training and greedy action selection during evaluation. Performance is measured using episode score, best score, average score, and rolling average over the last 50 episodes.

## 4.3 CI/CD, GitOps and Deployment

GitHub Actions is configured in `.github/workflows/ci.yml`. The workflow installs dependencies, runs linting, executes tests, performs smoke training, evaluates the trained model, and runs a monitoring threshold check.

## 4.4 Containerization and Workflow Automation

The Dockerfile creates a reproducible environment for training. The default container command runs the training pipeline using `config/default.json`.

## 4.5 Screenshots

Suggested screenshots:

- Training result plot from `reports/results.png`
- Test output showing `8 passed`
- GitHub Actions workflow success page
- Docker build/run terminal output
- Project folder structure

## 4.6 Unit and Integration Testing Results

The test suite validates environment reset, food placement, collision detection, reward behavior, Q-value updates, model save/load, training artifact creation, and monitoring checks.

## 4.7 Monitoring and Scalability

The monitoring log records score, reward, and rolling average score for every episode. A threshold check can alert if the rolling average falls below the expected value. Future scalability improvements include larger grids, Deep Q-Networks, cloud training, and dashboards.

## 4.8 Discussion of Findings

The project shows that Q-learning can learn basic Snake behavior but has limitations because the state space grows quickly. MLOps practices make the experiment easier to reproduce, evaluate, and improve.

## Conclusion and Future Work

The project successfully converts a reinforcement learning experiment into an MLOps-enabled pipeline. Future work can include MLflow experiment tracking, DQN-based learning, model registry integration, and deployment as a web service.

