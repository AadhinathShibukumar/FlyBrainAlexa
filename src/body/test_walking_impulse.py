from src.body.run_walking_benchmark import run
from src.body.validate_walking_summary import validate_summary


def test_small_roll_impulse_remains_stable(tmp_path) -> None:
    summary = run(
        steps=10,
        telemetry_path=tmp_path / "impulse.csv",
        physics_substeps=2,
        initial_roll_rate_deg_s=60.0,
    )
    assert validate_summary(summary) == []
