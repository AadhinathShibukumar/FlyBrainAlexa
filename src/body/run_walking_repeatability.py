import argparse
import json
from pathlib import Path
from typing import Any

from src.body.run_walking_benchmark import run
from src.body.validate_walking_summary import validate_summary


def summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        raise ValueError("at least one trial is required")
    passed = [trial["passed"] for trial in trials]
    speeds = [float(trial["summary"]["average_speed"]) for trial in trials]
    tilts = [float(trial["summary"]["max_tilt_deg"]) for trial in trials]
    return {
        "trial_count": len(trials),
        "passed_count": sum(passed),
        "pass_rate": sum(passed) / len(passed),
        "average_speed_min": min(speeds),
        "average_speed_max": max(speeds),
        "max_tilt_min": min(tilts),
        "max_tilt_max": max(tilts),
        "trials": trials,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repeated walking benchmarks.")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--physics-substeps", type=int, default=10)
    parser.add_argument("--initial-roll-deg", type=float, default=0.0)
    parser.add_argument("--initial-lateral-offset", type=float, default=0.0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/walking_trials")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("data/walking_repeatability.json")
    )
    args = parser.parse_args()
    if args.trials <= 0:
        raise ValueError("trials must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trials: list[dict[str, Any]] = []
    for index in range(args.trials):
        summary = run(
            args.steps,
            args.output_dir / f"trial_{index + 1}.csv",
            args.physics_substeps,
            args.initial_roll_deg,
            args.initial_lateral_offset,
        )
        failures = validate_summary(summary)
        trials.append(
            {
                "trial": index + 1,
                "passed": not failures,
                "failures": failures,
                "summary": summary,
            }
        )

    result = summarize_trials(trials)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed_count"] == result["trial_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
