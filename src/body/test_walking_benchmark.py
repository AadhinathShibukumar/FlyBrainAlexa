import numpy as np

from src.body.run_walking_benchmark import _roll_pitch


def test_roll_pitch_identity_quaternion() -> None:
    roll, pitch = _roll_pitch(np.array([1.0, 0.0, 0.0, 0.0]))
    assert roll == 0.0
    assert pitch == 0.0
