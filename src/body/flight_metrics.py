from dataclasses import dataclass

import numpy as np


@dataclass
class FlightMetrics:
    initial_height: float | None = None
    max_height: float = 0.0
    max_roll_deg: float = 0.0
    max_pitch_deg: float = 0.0
    airborne_steps: int = 0
    total_steps: int = 0

    def update(
        self,
        height: float,
        quaternion: np.ndarray,
        contacts: np.ndarray,
    ) -> None:
        w, x, y, z = np.asarray(quaternion, dtype=float)
        roll = np.degrees(np.arctan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (x * x + y * y),
        ))
        pitch = np.degrees(np.arcsin(
            np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        ))
        height = float(height)
        if self.initial_height is None:
            self.initial_height = height
        self.max_height = max(self.max_height, height)
        self.max_roll_deg = max(self.max_roll_deg, abs(float(roll)))
        self.max_pitch_deg = max(self.max_pitch_deg, abs(float(pitch)))
        self.total_steps += 1
        if not np.any(np.asarray(contacts) > 0.0):
            self.airborne_steps += 1

    @property
    def airborne_fraction(self) -> float:
        return self.airborne_steps / self.total_steps if self.total_steps else 0.0

    @property
    def altitude_gain(self) -> float:
        if self.initial_height is None:
            return 0.0
        return self.max_height - self.initial_height

    @property
    def controlled_flight(self) -> bool:
        return (
            self.airborne_fraction > 0.1
            and self.max_roll_deg < 35.0
            and self.max_pitch_deg < 35.0
        )
