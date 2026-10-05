import argparse
import json
from pathlib import Path
from typing import Any

from src.body.run_walking_benchmark import run
from src.body.validate_walking_summary import validate_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep walking recovery over roll angles.")
    parser.add_argument(
        "--roll-deg",
        type=float,
        nargs="+",
        default=[0.0, 5.0, 10.0, 15.0, 20.0],
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--physics-substeps", type=int, default=10)
    parser.add_argument("--lateral-offset", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=Path("data/recovery_sweep"))
    parser.add_argument("--summary", type=Path, default=Path("data/recovery_sweep.json"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trials: list[dict[str, Any]] = []
    for index, roll_deg in enumerate(args.roll_deg, start=1):
        result = run(
            args.steps,
            args.output_dir / f"roll_{roll_deg:g}.csv",
            args.physics_substeps,
            initial_roll_deg=roll_deg,
            initial_lateral_offset=args.lateral_offset,
        )
        failures = validate_summary(result)
        trials.append(
            {
                "trial": index,
                "roll_deg": roll_deg,
                "passed": not failures,
                "failures": failures,
                "summary": result,
            }
        )

    passing_angles = [trial["roll_deg"] for trial in trials if trial["passed"]]
    output = {
        "trial_count": len(trials),
        "passed_count": len(passing_angles),
        "largest_passing_roll_deg": max(passing_angles)
        if passing_angles
        else None,
        "trials": trials,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0 if len(passing_angles) == len(trials) else 1


if __name__ == "__main__":
    raise SystemExit(main())
