import numpy as np


class WingAerodynamics:
    """Small, bounded aerodynamic approximation for the FlyGym wing bodies."""

    def __init__(
        self,
        model,
        left_body_name: str,
        right_body_name: str,
        root_body_name: str,
        max_lift_weight_ratio: float = 1.25,
        attitude_torque: float = 0.5,
        angular_damping: float = 0.02,
    ) -> None:
        self.left_body_id = model.body(left_body_name).id
        self.right_body_id = model.body(right_body_name).id
        self.root_body_id = model.body(root_body_name).id
        self.weight = float(model.body_mass.sum() * abs(model.opt.gravity[2]))
        self.max_lift = self.weight * max_lift_weight_ratio
        self.attitude_torque = attitude_torque
        self.angular_damping = angular_damping

    def apply(
        self,
        data,
        phase: float,
        drive: float,
        differential: float = 0.0,
    ) -> float:
        """Apply instantaneous wing lift and return lift/weight ratio."""
        drive = float(np.clip(drive, 0.0, 1.0))
        differential = float(np.clip(differential, -0.35, 0.35))
        stroke_speed = abs(np.cos(phase))
        lift = self.max_lift * drive * drive * stroke_speed
        data.xfrc_applied[self.left_body_id, 2] = 0.5 * lift * (1.0 - differential)
        data.xfrc_applied[self.right_body_id, 2] = 0.5 * lift * (1.0 + differential)
        return lift / self.weight if self.weight else 0.0

    def stabilize(
        self,
        data,
        roll: float,
        pitch: float,
        quaternion: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> None:
        """Apply bounded attitude and angular-rate correction to the body."""
        angular_velocity = np.asarray(angular_velocity, dtype=float)
        if angular_velocity.shape != (3,):
            raise ValueError("angular_velocity must contain three values")
        quaternion = np.asarray(quaternion, dtype=float)
        if quaternion.shape != (4,):
            raise ValueError("quaternion must contain four values")
        quaternion = quaternion / max(np.linalg.norm(quaternion), 1e-9)
        shortest_path = 1.0 if quaternion[0] >= 0.0 else -1.0
        torque = np.array(
            [
                -self.attitude_torque * shortest_path * quaternion[1],
                -self.attitude_torque * shortest_path * quaternion[2],
                -self.attitude_torque * shortest_path * quaternion[3],
            ]
        )
        torque -= self.angular_damping * np.clip(angular_velocity, -10.0, 10.0)
        data.xfrc_applied[self.root_body_id, 3:] = torque

    def clear(self, data) -> None:
        data.xfrc_applied[self.left_body_id, :] = 0.0
        data.xfrc_applied[self.right_body_id, :] = 0.0
        data.xfrc_applied[self.root_body_id, :] = 0.0
