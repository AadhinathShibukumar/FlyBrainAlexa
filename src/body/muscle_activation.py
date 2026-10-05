import numpy as np


class MuscleActivation:
    """First-order activation and velocity limiting for normalized joint targets."""

    def __init__(
        self,
        size: int,
        activation_rate: float = 8.0,
        max_command_rate: float = 3.0,
    ) -> None:
        self.activation = np.full(size, 0.5, dtype=float)
        self.activation_rate = float(activation_rate)
        self.max_command_rate = float(max_command_rate)

    def step(self, target: np.ndarray, dt: float) -> np.ndarray:
        target = np.asarray(target, dtype=float)
        if target.shape != self.activation.shape:
            raise ValueError("target shape must match activation state")
        dt = max(float(dt), 0.0)
        activation_step = self.activation_rate * dt
        delta = np.clip(target - self.activation, -activation_step, activation_step)
        self.activation = np.clip(self.activation + delta, 0.0, 1.0)
        return self.activation.copy()

    def command(self, target: np.ndarray, dt: float) -> np.ndarray:
        target = np.asarray(target, dtype=float)
        rate_step = self.max_command_rate * max(float(dt), 0.0)
        return np.clip(
            target,
            self.activation - rate_step,
            self.activation + rate_step,
        )
