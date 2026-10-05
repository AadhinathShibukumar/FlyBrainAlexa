import numpy as np


class SupportBalance:
    """Use the weighted foot-support centroid to bias leg targets."""

    def __init__(self, gain: float = 0.12, limit: float = 0.12) -> None:
        self.gain = gain
        self.limit = limit

    def correction(
        self,
        normalized_targets: np.ndarray,
        leg_dof_map: dict[str, list[int]],
        foot_positions: np.ndarray,
        forces: np.ndarray,
        body_position: np.ndarray,
        body_velocity: np.ndarray | None = None,
        quaternion: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        targets = np.asarray(normalized_targets, dtype=float).copy()
        positions = np.asarray(foot_positions, dtype=float)
        force_values = np.asarray(forces, dtype=float)
        body = np.asarray(body_position, dtype=float)
        velocity = (
            np.zeros(2)
            if body_velocity is None
            else np.asarray(body_velocity, dtype=float)[:2]
        )
        if positions.shape != (6, 3) or force_values.shape != (6,):
            raise ValueError("foot positions and forces must describe six legs")
        weights = np.clip(force_values, 0.0, None)
        total = float(weights.sum())
        if total <= 1e-9:
            return targets, np.zeros(2)
        centroid = np.average(positions[:, :2], axis=0, weights=weights)
        error = body[:2] - centroid
        error -= 0.05 * velocity
        if quaternion is not None:
            w, x, y, z = np.asarray(quaternion, dtype=float)
            roll = np.arctan2(
                2.0 * (w * x + y * z),
                1.0 - 2.0 * (x * x + y * y),
            )
            pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
            error += 0.1 * np.array([pitch, roll])
        correction = np.clip(self.gain * error, -self.limit, self.limit)
        for index, leg in enumerate(("LF", "LM", "LH", "RF", "RM", "RH")):
            indices = leg_dof_map[leg]
            if len(indices) < 2:
                continue
            side = -1.0 if leg.startswith("L") else 1.0
            targets[indices[0]] = np.clip(
                targets[indices[0]] + side * correction[1],
                0.0,
                1.0,
            )
            targets[indices[1]] = np.clip(
                targets[indices[1]] - correction[0],
                0.0,
                1.0,
            )
        return targets, error
