import numpy as np

from src.brain.leg_drive_mapper import LegDriveMapper


def test_mapper_produces_six_smoothed_channels() -> None:
    mapper = LegDriveMapper(smoothing=1.0)
    drives = mapper.step(0.5, 0.25)
    values = np.array([drives[leg] for leg in mapper.LEGS])
    assert values.shape == (6,)
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)
    assert len(np.unique(values)) == 6


def test_mapper_smooths_step_changes() -> None:
    mapper = LegDriveMapper(smoothing=0.2)
    first = mapper.step(1.0, 1.0)
    second = mapper.step(0.0, 0.0)
    assert second["LF"] < first["LF"]
    assert second["LF"] > 0.0
