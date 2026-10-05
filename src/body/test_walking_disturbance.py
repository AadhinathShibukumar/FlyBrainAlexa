from src.body.run_walking_benchmark import run
from src.body.validate_walking_summary import validate_summary


def test_small_disturbance_remains_stable(tmp_path) -> None:
    summary = run(
        steps=10,
        telemetry_path=tmp_path / "disturbance.csv",
        physics_substeps=2,
        initial_roll_deg=2.0,
        initial_lateral_offset=0.01,
    )
    assert validate_summary(summary) == []
