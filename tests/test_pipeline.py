from src.config import TrainingConfig
from src.monitor import check_performance
from src.train import run_training


def test_training_pipeline_writes_artifacts(tmp_path):
    config = TrainingConfig(
        episodes=5,
        max_steps_per_episode=30,
        model_path=str(tmp_path / "models" / "q_table.pkl"),
        metrics_path=str(tmp_path / "metrics" / "training_metrics.json"),
        plot_path=str(tmp_path / "reports" / "results.png"),
        monitoring_path=str(tmp_path / "monitoring" / "performance_log.csv"),
    )

    metrics = run_training(config)

    assert metrics["episodes"] == 5
    assert (tmp_path / "models" / "q_table.pkl").exists()
    assert (tmp_path / "metrics" / "training_metrics.json").exists()
    assert (tmp_path / "reports" / "results.png").exists()
    assert (tmp_path / "monitoring" / "performance_log.csv").exists()


def test_monitoring_check_reads_latest_value(tmp_path):
    log_path = tmp_path / "performance_log.csv"
    log_path.write_text(
        "episode,score,reward,rolling_avg_50\n"
        "1,0,-10,0.0\n"
        "2,1,5,0.5\n",
        encoding="utf-8",
    )

    result = check_performance(str(log_path), min_rolling_average=0.2)

    assert result["status"] == "PASS"
    assert result["latest_rolling_avg_50"] == 0.5

