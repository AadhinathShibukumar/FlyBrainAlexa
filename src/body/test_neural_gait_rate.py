import numpy as np


def gait_frequency(motor_drive: float) -> float:
    return float(np.clip(6.0 + 18.0 * motor_drive, 4.0, 24.0))


def test_gait_rate_tracks_motor_drive() -> None:
    assert gait_frequency(0.0) == 6.0
    assert gait_frequency(1.0) == 24.0
    assert gait_frequency(0.5) > gait_frequency(0.1)


def test_gait_rate_is_bounded() -> None:
    assert 4.0 <= gait_frequency(-2.0) <= 24.0
    assert 4.0 <= gait_frequency(2.0) <= 24.0
