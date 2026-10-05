import argparse
import csv
from pathlib import Path


LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")


def analyze(path: Path) -> dict[str, float | int | str | None]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("telemetry file is empty")

    tilts = [
        (float(row["roll_deg"]) ** 2 + float(row["pitch_deg"]) ** 2) ** 0.5
        for row in rows
    ]
    support_counts = [
        sum(float(row[f"contact_{leg}"]) > 0.0 for leg in LEGS)
        for row in rows
    ]
    peak_index = max(range(len(rows)), key=tilts.__getitem__)
    first_full_support = next(
        (index for index, count in enumerate(support_counts) if count >= 4),
        None,
    )
    baseline_frame = 0 if first_full_support is None else first_full_support
    first_low_support = next(
        (
            index
            for index, count in enumerate(support_counts[baseline_frame:], baseline_frame)
            if count < 2
        ),
        None,
    )
    first_front_loss = next(
        (
            index
            for index, row in enumerate(rows[baseline_frame:], baseline_frame)
            if float(row["contact_LF"]) == 0.0
            and float(row["contact_RF"]) == 0.0
        ),
        None,
    )
    return {
        "frames": len(rows),
        "peak_tilt_deg": tilts[peak_index],
        "peak_tilt_frame": peak_index,
        "first_full_support_frame": first_full_support,
        "first_front_pair_loss_frame": first_front_loss,
        "first_low_support_frame": first_low_support,
        "peak_support_legs": max(support_counts),
        "minimum_support_legs": min(support_counts),
        "first_failure_signal": _first_signal(
            first_front_loss,
            first_low_support,
            peak_index,
        ),
    }


def _first_signal(
    front_loss: int | None,
    low_support: int | None,
    peak_tilt: int,
) -> str:
    candidates = [
        (frame, "front_contact_loss") for frame in (front_loss,) if frame is not None
    ]
    candidates.extend(
        (frame, "low_support") for frame in (low_support,) if frame is not None
    )
    candidates.append((peak_tilt, "peak_tilt"))
    return min(candidates, key=lambda item: item[0])[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze recovery telemetry.")
    parser.add_argument("telemetry", type=Path)
    args = parser.parse_args()
    import json

    print(json.dumps(analyze(args.telemetry), indent=2))


if __name__ == "__main__":
    main()
