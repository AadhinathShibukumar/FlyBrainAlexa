import numpy as np


class WingbeatController:
    """Generate bilateral wingbeat targets for the available wing joints."""

    def __init__(self, frequency_hz: float = 12.0) -> None:
        self.frequency_hz = frequency_hz
        self.phase = 0.0

    def step(
        self,
        drive: float,
        dt: float,
        left_indices: list[int],
        right_indices: list[int],
    ) -> np.ndarray:
        if len(left_indices) != 3 or len(right_indices) != 3:
            raise ValueError("each wing must have pitch, roll, and yaw indices")

        self.phase = (self.phase + 2.0 * np.pi * self.frequency_hz * dt) % (
            2.0 * np.pi
        )
        amplitude = float(np.clip(drive, 0.0, 1.0))
        pitch = 0.5 + 0.35 * amplitude * np.sin(self.phase)
        roll = 0.5 + 0.05 * amplitude * np.cos(self.phase)
        yaw = 0.5 + 0.05 * amplitude * np.sin(self.phase)

        output = np.full(6, 0.5)
        output[:3] = (pitch, roll, yaw)
        output[3:] = (pitch, 1.0 - roll, 1.0 - yaw)
        return output
