import numpy as np


class FlightSafetyLimiter:
    """Reduce lift when attitude exceeds safe flight-test bounds."""

    def __init__(self, max_tilt_deg: float = 35.0) -> None:
        self.max_tilt = np.radians(max_tilt_deg)

    def limit(self, drive: float, roll: float, pitch: float) -> float:
        tilt = max(abs(float(roll)), abs(float(pitch)))
        if tilt >= self.max_tilt:
            return 0.0
        return float(np.clip(drive, 0.0, 1.0) * (1.0 - tilt / self.max_tilt))
