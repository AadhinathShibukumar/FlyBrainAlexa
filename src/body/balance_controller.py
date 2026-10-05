import numpy as np


class BalanceController:
    """Convert free-body tilt into corrective proximal-leg commands."""

    def __init__(self, gain: float = 0.35, limit: float = 0.2) -> None:
        self.gain = gain
        self.limit = limit

    def apply(
        self,
        normalized_targets: np.ndarray,
        leg_dof_map: dict[str, list[int]],
        quaternion: np.ndarray,
    ) -> np.ndarray:
        targets = np.asarray(normalized_targets, dtype=float).copy()
        if targets.ndim != 1 or quaternion.shape != (4,):
            raise ValueError("invalid balance controller inputs")

        w, x, y, z = quaternion
        sin_roll = 2.0 * (w * x + y * z)
        cos_roll = 1.0 - 2.0 * (x * x + y * y)
        sin_pitch = 2.0 * (w * y - z * x)
        roll = np.arctan2(sin_roll, cos_roll)
        pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))

        for leg, indices in leg_dof_map.items():
            if len(indices) < 2:
                continue
            side = -1.0 if leg.startswith("L") else 1.0
            correction = np.clip(
                self.gain * (side * roll + pitch),
                -self.limit,
                self.limit,
            )
            targets[indices[1]] = np.clip(0.5 + correction, 0.0, 1.0)
            targets[indices[0]] = np.clip(0.5 - correction, 0.0, 1.0)
        return targets
