from __future__ import annotations

import argparse
import csv
from pathlib import Path


def check_performance(log_path: str, min_rolling_average: float = 0.2) -> dict:
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"Monitoring log not found: {log_path}")

    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return {"status": "NO_DATA", "latest_rolling_avg_50": 0.0}

    latest = float(rows[-1]["rolling_avg_50"])
    status = "PASS" if latest >= min_rolling_average else "ALERT"
    return {"status": status, "latest_rolling_avg_50": latest, "threshold": min_rolling_average}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check trained agent monitoring metrics.")
    parser.add_argument("--log", default="monitoring/performance_log.csv")
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()

    result = check_performance(args.log, args.threshold)
    print(result)
    if result["status"] == "ALERT":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

