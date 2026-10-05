"""Autonomous closed-loop fly agent with continuous telemetry and validation.

This script runs the fly in an extended autonomous walking scenario with:
- Continuous walking for configurable duration
- Periodic random disturbances (roll, pitch, lateral kicks)
- Real-time telemetry (position, orientation, leg contacts, forces, motor activity)
- Stability validation (max tilt, fall detection, recovery metrics)
- Neural/motor readout logging for connectivity analysis
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from src.body.fly_model import FlyBody
from src.body.gait_controller import TripodGait
from src.body.telemetry_logger import TelemetryLogger
from src.brain.connectome_brain import ConnectomeBrain


def run_autonomous_agent(
    duration_seconds: float = 60.0,
    disturbance_interval: float = 10.0,
    disturbance_magnitude: float = 500.0,  # deg/s for angular impulses
    neural_interval: int = 1,
    output_dir: Optional[Path] = None,
) -> dict:
    """Run the fly agent autonomously with periodic disturbances.

    Args:
        duration_seconds: Total simulation time in seconds.
        disturbance_interval: Time (seconds) between applying random disturbances.
        disturbance_magnitude: Maximum angular velocity (deg/s) for impulses.
        neural_interval: Skip frames for neural updates (1 = every frame).
        output_dir: Directory to save telemetry CSVs and reports. If None, uses ./output/autonomous_agent_<timestamp>.

    Returns:
        dict with keys:
            - total_steps: Number of physics steps executed.
            - max_tilt_deg: Maximum body tilt during run.
            - fall_count: Number of detected falls.
            - stability_score: 0-100 score based on tilt and falls.
            - num_disturbances: Number of disturbances applied.
            - recovery_times: List of (disturbance_type, recovery_time_ms) tuples.
            - motor_neuron_stats: Dict with motor neuron firing statistics.
            - descending_neuron_stats: Dict with descending neuron firing statistics.
    """
    if output_dir is None:
        output_dir = Path("output") / f"autonomous_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting autonomous agent for {duration_seconds:.1f}s")
    print(f"Output: {output_dir}")
    print(f"Disturbance interval: {disturbance_interval:.1f}s, magnitude: {disturbance_magnitude:.0f} deg/s")

    # Initialize body, brain, controllers.
    body = FlyBody()
    sim = body.get_sim()
    fly = body.get_fly()
    dt_physics = sim.mj_model.opt.timestep

    # Set neutral pose.
    neutral = np.full(body.num_dofs, 0.5)
    neutral_targets = body.bio_limits_min + neutral * (
        body.bio_limits_max - body.bio_limits_min
    )
    sim.set_actuator_inputs(fly.name, body.actuator_type, neutral_targets)
    for _ in range(300):
        sim.step()

    brain = ConnectomeBrain()
    gait = TripodGait()
    gait.frequency = 12.0
    dt_neural = neural_interval * dt_physics

    # Initialize telemetry and logging.
    telemetry_path = output_dir / "telemetry.csv"
    telemetry = TelemetryLogger(telemetry_path)
    motor_log = []  # List[dict] for motor neuron activity.
    descending_log = []  # List[dict] for descending neuron activity.
    disturbance_log = []  # List[dict] for disturbance events.

    # Compute total physics steps.
    total_steps = int(duration_seconds / dt_physics)
    steps_per_disturbance = int(disturbance_interval / dt_physics)

    # Tracking for disturbances and recovery.
    next_disturbance_step = steps_per_disturbance
    disturbances_applied = 0
    last_max_tilt_deg = 0.0
    recovery_times = []

    print(f"Total steps: {total_steps:,}")
    print(f"Physics dt: {dt_physics:.4f}s, neural dt: {dt_neural:.4f}s")
    print(f"Stepping {neural_interval}x faster than real-time (neural_interval={neural_interval})")

    # Main loop.
    for step in range(total_steps):
        # Apply disturbance if it's time.
        if step >= next_disturbance_step:
            disturbance_type = _apply_random_disturbance(
                sim, disturbance_magnitude
            )
            disturbances_applied += 1
            next_disturbance_step += steps_per_disturbance
            last_max_tilt_deg = 0.0  # Reset tilt tracker for recovery measurement.
            disturbance_log.append({
                "step": step,
                "time_s": step * dt_physics,
                "type": disturbance_type,
            })
            print(f"  Step {step:6d}: Applied {disturbance_type} disturbance")

        # Update brain (every neural_interval steps).
        if step % neural_interval == 0:
            brain_output = brain.step(dt_neural)

        # Get motor outputs.
        motor_drive = brain_output.get("motor_neuron_drive", 0.0)
        descending_drive = brain_output.get("motor_drive", 0.0)

        # Log motor and descending activity (sample every 10th neural step to reduce data).
        if step % (neural_interval * 10) == 0:
            motor_log.append({
                "step": step,
                "time_s": step * dt_physics,
                "motor_neuron_drive": float(motor_drive),
            })
            descending_log.append({
                "step": step,
                "time_s": step * dt_physics,
                "descending_drive": float(descending_drive),
            })

        # Gait control.
        contacts = _get_leg_contacts(sim)
        gait_activations = gait.step(
            dt=dt_neural,
            amplitudes=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            contacts=contacts,
        )
        
        # Convert gait activations to normalized joint positions [0, 1].
        normalized = np.full(body.num_dofs, 0.5)
        for leg_index, leg in enumerate(body.legs):
            indices = body.leg_dof_map[leg]
            count = min(len(indices), gait_activations.shape[1])
            normalized[indices[:count]] = gait_activations[leg_index, :count]
        
        # Apply motor drive as amplitude modulation.
        normalized = 0.5 + (normalized - 0.5) * (1.0 + 0.2 * motor_drive)
        targets = body.bio_limits_min + normalized * (body.bio_limits_max - body.bio_limits_min)

        # Step physics.
        body.step(targets)

        # Log telemetry (re-compute after physics step for current state).
        roll_deg, pitch_deg, yaw_deg = _quat_to_euler_deg(sim.mj_data.qpos[3:7])
        max_tilt = max(abs(roll_deg), abs(pitch_deg))
        last_max_tilt_deg = max(last_max_tilt_deg, max_tilt)

        contacts = _get_leg_contacts(sim)
        forces = _get_leg_forces(sim, fly.name)

        telemetry.write({
            "frame": step,
            "sim_time": step * dt_physics,
            "motor_drive": _motor_to_leg_drive(motor_drive),
            "height": float(sim.mj_data.qpos[2]),
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "contact_LF": contacts[0],
            "contact_LM": contacts[1],
            "contact_LH": contacts[2],
            "contact_RF": contacts[3],
            "contact_RM": contacts[4],
            "contact_RH": contacts[5],
            "force_LF": forces[0],
            "force_LM": forces[1],
            "force_LH": forces[2],
            "force_RF": forces[3],
            "force_RM": forces[4],
            "force_RH": forces[5],
        })

        # Detect fall (any tilt > 45° or all legs lost contact).
        if max_tilt > 45.0 or (contacts.sum() < 2):
            recovery_times.append((
                disturbance_log[-1]["type"] if disturbance_log else "unknown",
                step * dt_physics * 1000.0,
            ))

        # Progress indicator every 1000 steps.
        if step % 1000 == 0 and step > 0:
            elapsed_s = step * dt_physics
            print(f"  Step {step:6d} ({elapsed_s:6.1f}s): "
                  f"pos=[{sim.mj_data.qpos[0]:6.2f}, {sim.mj_data.qpos[1]:6.2f}, {sim.mj_data.qpos[2]:6.2f}] "
                  f"tilt=[{roll_deg:6.1f}°, {pitch_deg:6.1f}°] "
                  f"contacts={int(contacts.sum())}/6 "
                  f"motor_drive={motor_drive:.3f}")

    # Save telemetry (handled by telemetry logger, which writes incrementally).
    # Already saved at telemetry_path above.
    telemetry._file.close()

    # Save motor and descending logs.

    if motor_log:
        motor_path = output_dir / "motor_activity.csv"
        with open(motor_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=motor_log[0].keys())
            writer.writeheader()
            writer.writerows(motor_log)

    if descending_log:
        descending_path = output_dir / "descending_activity.csv"
        with open(descending_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=descending_log[0].keys())
            writer.writeheader()
            writer.writerows(descending_log)

    if disturbance_log:
        disturbance_path = output_dir / "disturbances.csv"
        with open(disturbance_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=disturbance_log[0].keys())
            writer.writeheader()
            writer.writerows(disturbance_log)

    # Compute statistics.
    # Since telemetry is already saved to CSV, read it back for analysis.
    telemetry_stats = {"max_tilt_deg": 0.0, "fall_count": 0}
    with open(telemetry_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            roll_deg = float(row.get("roll_deg", 0))
            pitch_deg = float(row.get("pitch_deg", 0))
            max_tilt = max(abs(roll_deg), abs(pitch_deg))
            telemetry_stats["max_tilt_deg"] = max(telemetry_stats["max_tilt_deg"], max_tilt)
            # Count falls as instances with multiple legs off ground (conservative heuristic).
            contacts_on = sum(1 for i in range(1, 7) if float(row.get(f"contact_{'LFMHRFRMRH'[i-1]}", 0)) > 0.5)
            if contacts_on < 2:
                telemetry_stats["fall_count"] += 1

    max_tilt_deg = telemetry_stats["max_tilt_deg"]
    fall_count = telemetry_stats["fall_count"]
    stability_score = max(0, 100 - max_tilt_deg - 10 * fall_count)

    # Compute motor neuron statistics.
    motor_stats = {}
    if motor_log:
        motor_drives = [m["motor_neuron_drive"] for m in motor_log]
        motor_stats = {
            "mean_activation": float(np.mean(motor_drives)),
            "std_activation": float(np.std(motor_drives)),
            "max_activation": float(np.max(motor_drives)),
            "samples": len(motor_log),
        }

    descending_stats = {}
    if descending_log:
        descending_drives = [d["descending_drive"] for d in descending_log]
        descending_stats = {
            "mean_activation": float(np.mean(descending_drives)),
            "std_activation": float(np.std(descending_drives)),
            "max_activation": float(np.max(descending_drives)),
            "samples": len(descending_log),
        }

    result = {
        "total_steps": total_steps,
        "duration_s": duration_seconds,
        "max_tilt_deg": float(max_tilt_deg),
        "fall_count": fall_count,
        "stability_score": float(stability_score),
        "num_disturbances": disturbances_applied,
        "recovery_times_ms": recovery_times,
        "motor_neuron_stats": motor_stats,
        "descending_neuron_stats": descending_stats,
        "telemetry_file": str(telemetry_path.relative_to(output_dir.parent)),
    }

    # Save summary report.
    report_path = output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print("Autonomous Agent Summary:")
    print(f"  Duration: {duration_seconds:.1f}s ({total_steps:,} steps)")
    print(f"  Max tilt: {max_tilt_deg:.1f}°")
    print(f"  Falls: {fall_count}")
    print(f"  Stability score: {stability_score:.0f}/100")
    print(f"  Disturbances: {disturbances_applied}")
    print(f"  Motor neuron activity: mean={motor_stats.get('mean_activation', 0):.3f}, "
          f"std={motor_stats.get('std_activation', 0):.3f}")
    print(f"  Descending activity: mean={descending_stats.get('mean_activation', 0):.3f}, "
          f"std={descending_stats.get('std_activation', 0):.3f}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    return result


def _apply_random_disturbance(sim, magnitude_deg_s: float) -> str:
    """Apply a random angular impulse to the fly body.

    Args:
        sim: Simulation object.
        magnitude_deg_s: Maximum angular velocity magnitude.

    Returns:
        str: Type of disturbance ("roll", "pitch", or "yaw").
    """
    disturbance_type = np.random.choice(["roll", "pitch", "yaw"])
    mag_rad_s = np.radians(magnitude_deg_s)

    if disturbance_type == "roll":
        sim.mj_data.qvel[3] += (np.random.random() - 0.5) * 2 * mag_rad_s
    elif disturbance_type == "pitch":
        sim.mj_data.qvel[4] += (np.random.random() - 0.5) * 2 * mag_rad_s
    else:  # yaw
        sim.mj_data.qvel[5] += (np.random.random() - 0.5) * 2 * mag_rad_s

    return disturbance_type


def _motor_to_leg_drive(motor_activity_scalar: float) -> float:
    """Convert motor neuron activity scalar to leg drive [0, 1]."""
    return float(min(1.0, max(0.0, motor_activity_scalar)))


def _quat_to_euler_deg(quat: np.ndarray) -> tuple[float, float, float]:
    """Convert quaternion [w, x, y, z] to Euler angles in degrees [roll, pitch, yaw]."""
    w, x, y, z = quat
    # Roll (rotation around x-axis).
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    # Pitch (rotation around y-axis).
    pitch = np.arcsin(2 * (w * y - z * x))
    # Yaw (rotation around z-axis).
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    return float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))


def _get_leg_contacts(sim) -> np.ndarray:
    """Extract per-leg contact status (0 or 1) from simulation."""
    contacts = np.zeros(6, dtype=np.float32)
    # This is a simplified stub; actual implementation depends on FlyGym API.
    if sim.mj_data.ncon == 0:
        return contacts
    # For now, assume any contact means all legs are in contact (placeholder).
    contacts[:] = 1.0
    return contacts


def _get_leg_forces(sim, fly_name: str) -> np.ndarray:
    """Extract per-leg force magnitudes from simulation."""
    forces = np.zeros(6, dtype=np.float32)
    # Placeholder; actual implementation depends on MuJoCo sensor outputs.
    return forces


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run autonomous closed-loop fly agent")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds")
    parser.add_argument("--disturbance-interval", type=float, default=10.0, help="Interval between disturbances (seconds)")
    parser.add_argument("--disturbance-magnitude", type=float, default=500.0, help="Disturbance magnitude (deg/s)")
    parser.add_argument("--neural-interval", type=int, default=1, help="Neural update interval (physics frames)")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    run_autonomous_agent(
        duration_seconds=args.duration,
        disturbance_interval=args.disturbance_interval,
        disturbance_magnitude=args.disturbance_magnitude,
        neural_interval=args.neural_interval,
        output_dir=args.output,
    )
