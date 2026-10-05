import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from src.body.fly_model import FlyBody
from src.body.gait_controller import TripodGait
from src.body.recovery_controller import RecoveryController
from src.body.telemetry_logger import TelemetryLogger
from src.brain.connectome_brain import ConnectomeBrain


def run(
    steps: int,
    telemetry_path: Path,
    physics_substeps: int = 50,
    initial_roll_deg: float = 0.0,
    initial_lateral_offset: float = 0.0,
) -> dict[str, float | int]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if physics_substeps <= 0:
        raise ValueError("physics_substeps must be positive")
    body = FlyBody()
    sim = body.get_sim()
    fly = body.get_fly()
    neutral = np.full(body.num_dofs, 0.5)
    neutral_targets = body.bio_limits_min + neutral * (
        body.bio_limits_max - body.bio_limits_min
    )
    sim.set_actuator_inputs(fly.name, body.actuator_type, neutral_targets)
    for _ in range(300):
        sim.step()
    sim.mj_data.qpos[1] += initial_lateral_offset
    if initial_roll_deg:
        angle = np.radians(initial_roll_deg)
        sim.mj_data.qpos[3:7] = (
            np.cos(angle / 2.0),
            np.sin(angle / 2.0),
            0.0,
            0.0,
        )
    mujoco.mj_forward(sim.mj_model, sim.mj_data)

    brain = ConnectomeBrain()
    gait = TripodGait()
    gait.frequency = 12.0
    recovery = RecoveryController()
    dt = physics_substeps * sim.mj_model.opt.timestep
    filtered = neutral_targets.copy()
    contacts = body.get_sensory_state()["leg_contact"]
    proprioceptive_drive = 0.0
    contact_force_drive = 0.0
    start_xy = sim.mj_data.qpos[:2].copy()
    max_tilt = 0.0
    falls = 0
    recovery_frames = 0
    previous_fallen = False

    with TelemetryLogger(telemetry_path) as logger:
        for frame in range(steps):
            roll, pitch = _roll_pitch(sim.mj_data.qpos[3:7])
            recovery_scale = recovery.scale(
                support_count=int(np.count_nonzero(contacts > 0.0)),
                roll=roll,
                pitch=pitch,
                root_velocity=sim.mj_data.qvel[:2],
            )
            state = brain.step(
                olfactory_drive=2.0,
                mechanosensory_drive=float(contacts.mean()),
                proprioceptive_drive=proprioceptive_drive,
                contact_force_drive=contact_force_drive,
            )
            amplitudes = np.array(
                [state["leg_drives"][leg] for leg in body.legs],
                dtype=float,
            ) * recovery_scale
            gait_activations = gait.step(
                dt=dt,
                amplitudes=amplitudes,
                contacts=contacts,
            )
            normalized = np.full(body.num_dofs, 0.5)
            for leg_index, leg in enumerate(body.legs):
                indices = body.leg_dof_map[leg]
                count = min(len(indices), gait_activations.shape[1])
                normalized[indices[:count]] = gait_activations[leg_index, :count]
            targets = body.bio_limits_min + normalized * (
                body.bio_limits_max - body.bio_limits_min
            )
            filtered = 0.15 * targets + 0.85 * filtered
            for _ in range(physics_substeps):
                body.step(filtered)

            sensory = body.get_sensory_state()
            contacts = sensory["leg_contact"]
            contact_forces = sensory["contact_force"]
            contact_force_drive = float(np.clip(contact_forces.mean() / 10.0, 0.0, 1.0))
            proprioceptive_drive = float(
                np.clip(np.mean(np.abs(normalized - 0.5)) * 2.0, 0.0, 1.0)
            )
            roll, pitch = _roll_pitch(sim.mj_data.qpos[3:7])
            tilt = float(np.hypot(roll, pitch))
            max_tilt = max(max_tilt, tilt)
            fallen = bool(
                sim.mj_data.qpos[2] < 0.12
                or tilt > np.deg2rad(45.0)
            )
            if fallen and not previous_fallen:
                falls += 1
            previous_fallen = fallen
            if int(np.count_nonzero(contacts > 0.0)) < 2 or fallen:
                recovery_frames += 1
            logger.write(
                {
                    "frame": frame,
                    "sim_time": sim.mj_data.time,
                    "motor_drive": state["motor_drive"],
                    "left_drive": state["left_drive"],
                    "right_drive": state["right_drive"],
                    "contact_fraction": float(contacts.mean()),
                    "force_drive": contact_force_drive,
                    "proprioceptive_drive": proprioceptive_drive,
                    "velocity_drive": float(np.linalg.norm(sim.mj_data.qvel[:2])),
                    "recovery_scale": recovery_scale,
                    "height": sim.mj_data.qpos[2],
                    "roll_deg": np.degrees(roll),
                    "pitch_deg": np.degrees(pitch),
                    "wing_motion": 0.0,
                    "wing_drive": 0.0,
                    **{
                        f"contact_{leg}": float(contacts[index])
                        for index, leg in enumerate(body.legs)
                    },
                    **{
                        f"force_{leg}": float(contact_forces[index])
                        for index, leg in enumerate(body.legs)
                    },
                }
            )

    displacement = sim.mj_data.qpos[:2] - start_xy
    return {
        "steps": steps,
        "duration_s": float(steps * dt),
        "forward_displacement": float(displacement[0]),
        "lateral_displacement": float(displacement[1]),
        "average_speed": float(np.linalg.norm(displacement) / (steps * dt)),
        "falls": falls,
        "recovery_fraction": float(recovery_frames / max(steps, 1)),
        "max_tilt_deg": float(np.degrees(max_tilt)),
        "final_support_legs": int(np.count_nonzero(contacts > 0.0)),
    }


def _roll_pitch(quaternion: np.ndarray) -> tuple[float, float]:
    w, x, y, z = quaternion
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    return float(roll), float(np.arcsin(pitch_arg))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a headless walking benchmark.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--physics-substeps", type=int, default=50)
    parser.add_argument("--telemetry", type=Path, default=Path("data/walking_telemetry.csv"))
    parser.add_argument("--summary", type=Path, default=Path("data/walking_summary.json"))
    args = parser.parse_args()
    summary = run(args.steps, args.telemetry, args.physics_substeps)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
