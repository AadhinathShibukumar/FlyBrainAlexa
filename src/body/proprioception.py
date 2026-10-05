import numpy as np


class Proprioception:
    """Read normalized joint positions and attenuate unsafe leg commands."""

    def __init__(self, limits_min: np.ndarray, limits_max: np.ndarray) -> None:
        self.limits_min = np.asarray(limits_min, dtype=float)
        self.limits_max = np.asarray(limits_max, dtype=float)
        if self.limits_min.shape != self.limits_max.shape:
            raise ValueError("joint limit arrays must have the same shape")

    def normalized_positions(self, qpos: np.ndarray, addresses: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=float)
        addresses = np.asarray(addresses, dtype=int)
        angles = qpos[addresses]
        span = np.maximum(self.limits_max - self.limits_min, 1e-9)
        return np.clip((angles - self.limits_min) / span, 0.0, 1.0)

    def normalized_velocity(
        self,
        qvel: np.ndarray,
        addresses: np.ndarray,
        reference_speed: float = 20.0,
    ) -> np.ndarray:
        qvel = np.asarray(qvel, dtype=float)
        addresses = np.asarray(addresses, dtype=int)
        if reference_speed <= 0.0:
            raise ValueError("reference_speed must be positive")
        return np.clip(np.abs(qvel[addresses]) / reference_speed, 0.0, 1.0)

    def tracking_scale(
        self,
        target: np.ndarray,
        actual: np.ndarray,
        leg_dof_map: dict[str, list[int]],
    ) -> np.ndarray:
        scale = np.ones(6, dtype=float)
        error = np.abs(np.asarray(target) - np.asarray(actual))
        for leg_index, leg in enumerate(("LF", "LM", "LH", "RF", "RM", "RH")):
            indices = leg_dof_map[leg]
            if indices:
                scale[leg_index] = np.clip(
                    1.0 - float(np.mean(error[indices])) * 1.5,
                    0.25,
                    1.0,
                )
        return scale
