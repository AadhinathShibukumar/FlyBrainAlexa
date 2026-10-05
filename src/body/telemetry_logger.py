import csv
from pathlib import Path
from typing import Iterable


class TelemetryLogger:
    """Write one deterministic CSV row per controller frame."""

    FIELDS = (
        "frame",
        "sim_time",
        "motor_drive",
        "left_drive",
        "right_drive",
        "contact_fraction",
        "force_drive",
        "proprioceptive_drive",
        "velocity_drive",
        "recovery_scale",
        "height",
        "roll_deg",
        "pitch_deg",
        "wing_motion",
        "wing_drive",
    )

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def write(self, values: dict[str, float | int]) -> None:
        row = {field: values.get(field, "") for field in self.FIELDS}
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "TelemetryLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
