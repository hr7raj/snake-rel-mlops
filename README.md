# MLOps Pipeline for Snake Q-Learning

This project operationalizes a reinforcement learning Snake agent using a small MLOps workflow. The original model is a tabular Q-learning agent; the MLOps layer adds reproducible configuration, artifact saving, automated tests, CI/CD, containerization, and basic performance monitoring.

## Project Structure

```text
.
├── config/default.json          # Training and artifact configuration
├── src/environment.py           # Snake environment
├── src/agent.py                 # Q-learning agent and model serialization
├── src/train.py                 # Training pipeline
├── src/evaluate.py              # Evaluation pipeline
├── src/monitor.py               # Monitoring threshold check
├── tests/                       # Unit and integration tests
├── Dockerfile                   # Reproducible container build
└── .github/workflows/ci.yml     # CI/CD workflow
```

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Train

```bash
python -m src.train --config config/default.json
```

Training writes:

- `models/q_table.pkl`
- `metrics/training_metrics.json`
- `reports/results.png`
- `monitoring/performance_log.csv`

## Evaluate

```bash
python -m src.evaluate --config config/default.json --episodes 20
```

## Test

```bash
pytest -q
```

## Monitor

```bash
python -m src.monitor --log monitoring/performance_log.csv --threshold 0.2
```

## Docker

```bash
docker build -t snake-rl-mlops .
docker run --rm snake-rl-mlops
```

## MLOps Mapping

- **Data collection**: RL experience is generated through agent-environment interaction.
- **Model development**: Q-learning updates the Q-table using reward signals.
- **Experiment tracking**: Hyperparameters, scores, rewards, and plots are saved as artifacts.
- **CI/CD**: GitHub Actions runs linting, tests, smoke training, evaluation, and monitoring.
- **Containerization**: Docker provides a reproducible runtime.
- **Monitoring**: Rolling average score is logged and checked against a threshold.

