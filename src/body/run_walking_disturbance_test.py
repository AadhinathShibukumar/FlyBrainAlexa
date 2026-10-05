import argparse
import json
from pathlib import Path
from typing import Any

from src.body.run_walking_benchmark import run
from src.body.validate_walking_summary import validate_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Test walking recovery after disturbances.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--physics-substeps", type=int, default=10)
    parser.add_argument("--roll-deg", type=float, default=5.0)
    parser.add_argument("--roll-rate-deg-s", type=float, default=0.0)
    parser.add_argument("--lateral-offset", type=float, default=0.05)
    parser.add_argument("--telemetry", type=Path, default=Path("/tmp/walking_disturbance.csv"))
    parser.add_argument("--summary", type=Path, default=Path("data/walking_disturbance.json"))
    args = parser.parse_args()

    result: dict[str, Any] = run(
        args.steps,
        args.telemetry,
        args.physics_substeps,
        args.roll_deg,
        args.lateral_offset,
        args.roll_rate_deg_s,
    )
    failures = validate_summary(result)
    output = {"passed": not failures, "failures": failures, "summary": result}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
