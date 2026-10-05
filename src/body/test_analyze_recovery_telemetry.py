import csv

from src.body.analyze_recovery_telemetry import analyze


def test_analyzer_identifies_front_contact_loss(tmp_path) -> None:
    path = tmp_path / "telemetry.csv"
    fields = [
        "roll_deg",
        "pitch_deg",
        *(f"contact_{leg}" for leg in ("LF", "LM", "LH", "RF", "RM", "RH")),
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "roll_deg": 2,
                "pitch_deg": 0,
                **{f"contact_{leg}": 1 for leg in ("LF", "LM", "LH", "RF", "RM", "RH")},
            }
        )
        writer.writerow(
            {
                "roll_deg": 5,
                "pitch_deg": 0,
                **{
                    f"contact_{leg}": int(leg not in ("LF", "RF"))
                    for leg in ("LF", "LM", "LH", "RF", "RM", "RH")
                },
            }
        )
    result = analyze(path)
    assert result["first_front_pair_loss_frame"] == 1
    assert result["first_failure_signal"] == "front_contact_loss"
