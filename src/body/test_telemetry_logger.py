import csv

from src.body.telemetry_logger import TelemetryLogger


def test_telemetry_logger_writes_header_and_row(tmp_path) -> None:
    path = tmp_path / "telemetry.csv"
    with TelemetryLogger(path) as logger:
        logger.write({"frame": 1, "motor_drive": 0.5})
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["frame"] == "1"
    assert rows[0]["motor_drive"] == "0.5"
    assert rows[0]["height"] == ""
