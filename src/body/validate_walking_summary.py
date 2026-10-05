import argparse
import json
from pathlib import Path
from typing import Any


def validate_summary(
    summary: dict[str, Any],
    *,
    max_falls: int = 0,
    max_tilt_deg: float = 15.0,
    min_support_legs: int = 2,
) -> list[str]:
    failures: list[str] = []
    if int(summary.get("falls", -1)) > max_falls:
        failures.append(f"falls={summary.get('falls')} > {max_falls}")
    if float(summary.get("max_tilt_deg", float("inf"))) > max_tilt_deg:
        failures.append(
            f"max_tilt_deg={summary.get('max_tilt_deg')} > {max_tilt_deg}"
        )
    if int(summary.get("final_support_legs", -1)) < min_support_legs:
        failures.append(
            "final_support_legs="
            f"{summary.get('final_support_legs')} < {min_support_legs}"
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a walking benchmark summary.")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--max-falls", type=int, default=0)
    parser.add_argument("--max-tilt-deg", type=float, default=15.0)
    parser.add_argument("--min-support-legs", type=int, default=2)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    failures = validate_summary(
        summary,
        max_falls=args.max_falls,
        max_tilt_deg=args.max_tilt_deg,
        min_support_legs=args.min_support_legs,
    )
    if failures:
        print("WALKING BASELINE FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("WALKING BASELINE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
