import numpy as np


class FlightController:
    """Bounded attitude feedback for differential wing lift."""

    def __init__(self, roll_gain: float = 0.8, pitch_gain: float = 0.35) -> None:
        self.roll_gain = roll_gain
        self.pitch_gain = pitch_gain

    def corrections(self, quaternion: np.ndarray) -> tuple[float, float, float]:
        quaternion = np.asarray(quaternion, dtype=float)
        if quaternion.shape != (4,):
            raise ValueError("quaternion must contain four values")

        w, x, y, z = quaternion
        roll = np.arctan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (x * x + y * y),
        )
        pitch = np.arcsin(
            np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        )
        roll_correction = float(np.clip(self.roll_gain * roll, -0.35, 0.35))
        pitch_correction = float(np.clip(self.pitch_gain * pitch, -0.2, 0.2))
        return roll, pitch, roll_correction + pitch_correction
