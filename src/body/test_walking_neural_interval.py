import pytest

from src.body.run_walking_benchmark import run


def test_invalid_neural_interval_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="neural_interval"):
        run(steps=1, telemetry_path=tmp_path / "telemetry.csv", neural_interval=0)
