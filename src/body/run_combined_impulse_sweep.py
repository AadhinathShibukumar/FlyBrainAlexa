import argparse
import json
from pathlib import Path
from typing import Any

from src.body.run_walking_benchmark import run
from src.body.validate_walking_summary import validate_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep combined roll/pitch impulses.")
    parser.add_argument(
        "--rate-deg-s",
        type=float,
        nargs="+",
        default=[0.0, 180.0, 360.0, 540.0, 720.0],
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--physics-substeps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("data/combined_impulses"))
    parser.add_argument("--summary", type=Path, default=Path("data/combined_impulse_sweep.json"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trials: list[dict[str, Any]] = []
    for index, rate in enumerate(args.rate_deg_s, start=1):
        result = run(
            args.steps,
            args.output_dir / f"rate_{rate:g}.csv",
            args.physics_substeps,
            initial_roll_rate_deg_s=rate,
            initial_pitch_rate_deg_s=rate,
        )
        failures = validate_summary(result)
        trials.append(
            {
                "trial": index,
                "roll_rate_deg_s": rate,
                "pitch_rate_deg_s": rate,
                "passed": not failures,
                "failures": failures,
                "summary": result,
            }
        )

    passing_rates = [trial["roll_rate_deg_s"] for trial in trials if trial["passed"]]
    output = {
        "trial_count": len(trials),
        "passed_count": len(passing_rates),
        "largest_passing_combined_rate_deg_s": max(passing_rates)
        if passing_rates
        else None,
        "trials": trials,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0 if len(passing_rates) == len(trials) else 1


if __name__ == "__main__":
    raise SystemExit(main())
