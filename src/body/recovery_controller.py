import numpy as np


class RecoveryController:
    """Gate propulsion when the body loses safe support or attitude."""

    def __init__(
        self,
        max_tilt_deg: float = 45.0,
        max_horizontal_speed: float = 25.0,
    ) -> None:
        self.max_tilt = np.radians(max_tilt_deg)
        self.max_horizontal_speed = max_horizontal_speed

    def scale(
        self,
        support_count: int,
        roll: float,
        pitch: float,
        root_velocity: np.ndarray,
    ) -> float:
        speed = float(np.linalg.norm(np.asarray(root_velocity, dtype=float)[:2]))
        tilt = max(abs(float(roll)), abs(float(pitch)))
        if support_count < 2 or tilt >= self.max_tilt:
            return 0.0
        if speed >= self.max_horizontal_speed:
            return 0.25
        return float(
            np.clip(
                min(
                    1.0 - tilt / self.max_tilt,
                    1.0 - speed / self.max_horizontal_speed,
                ),
                0.25,
                1.0,
            )
        )
