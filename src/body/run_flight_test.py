import numpy as np

from flygym.compose import ActuatorType, FlatGroundWorld, KinematicPosePreset, NeuroMechFly
from flygym.simulation import Simulation
from src.body.aerodynamics import WingAerodynamics
from src.body.flight_controller import FlightController
from src.body.flight_metrics import FlightMetrics
from src.body.flight_safety import FlightSafetyLimiter
from src.body.takeoff_controller import FlightPhase, TakeoffController


class FlyOrientation:
    format = "quat"
    quat = (1.0, 0.0, 0.0, 0.0)

    def as_kwargs(self):
        return {"quat": self.quat}


def run_control_test(steps: int = 100) -> FlightMetrics:
    """Run the takeoff/flight state logic without opening a viewer."""
    takeoff = TakeoffController(ramp_seconds=0.5)
    safety = FlightSafetyLimiter()
    metrics = FlightMetrics()
    contacts = np.ones(6, dtype=np.float32)
    for step in range(steps):
        height = 2.0 if step < steps // 2 else 2.1
        if step >= steps // 2:
            contacts = np.zeros(6, dtype=np.float32)
        drive, _ = takeoff.step(1.0, contacts, height, 0.01)
        drive = safety.limit(drive, 0.0, 0.0)
        metrics.update(height, np.array([1.0, 0.0, 0.0, 0.0]), contacts)
        assert 0.0 <= drive <= 1.0
    assert takeoff.phase is FlightPhase.AIRBORNE
    return metrics


def run_physics_test(steps: int = 2000) -> FlightMetrics:
    """Run the actual MuJoCo body with bounded bilateral wing forces."""
    fly = NeuroMechFly()
    fly.skeleton = fly._get_base_skeleton()
    fly.add_joints(fly.skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)
    joint_dofs = list(fly.skeleton.iter_jointdofs())
    fly.add_actuators(
        joint_dofs,
        ActuatorType.POSITION,
        {joint_dof: 0.0 for joint_dof in joint_dofs},
    )
    world = FlatGroundWorld()
    world.add_fly(
        fly,
        spawn_position=(0.0, 0.0, 2.0),
        spawn_rotation=FlyOrientation(),
    )
    sim = Simulation(world=world)
    sim.reset()
    aerodynamics = WingAerodynamics(
        sim.mj_model,
        f"{fly.name}/l_wing",
        f"{fly.name}/r_wing",
        f"{fly.name}/c_thorax",
    )
    flight = FlightController()
    metrics = FlightMetrics()
    phase = 0.0
    dt = sim.mj_model.opt.timestep
    metrics.update(
        sim.mj_data.qpos[2],
        sim.mj_data.qpos[3:7],
        np.ones(6, dtype=np.float32),
    )
    for _ in range(steps):
        phase = (phase + 2.0 * np.pi * 40.0 * dt) % (2.0 * np.pi)
        roll, pitch, differential = flight.corrections(sim.mj_data.qpos[3:7])
        aerodynamics.stabilize(
            sim.mj_data,
            roll=roll,
            pitch=pitch,
            quaternion=sim.mj_data.qpos[3:7],
            angular_velocity=sim.mj_data.qvel[3:6],
        )
        aerodynamics.apply(
            sim.mj_data,
            phase=phase,
            drive=1.0,
            differential=differential,
        )
        sim.step()
        aerodynamics.clear(sim.mj_data)
        contacts = np.ones(6, dtype=np.float32) if sim.mj_data.ncon else np.zeros(6)
        metrics.update(sim.mj_data.qpos[2], sim.mj_data.qpos[3:7], contacts)
    return metrics


if __name__ == "__main__":
    result = run_control_test()
    physics = run_physics_test()
    print(
        f"max_height={result.max_height:.2f} "
        f"airborne_fraction={result.airborne_fraction:.2f}\n"
        f"physics_max_height={physics.max_height:.2f} "
        f"physics_gain={physics.altitude_gain:.3f} "
        f"physics_roll={physics.max_roll_deg:.1f} "
        f"physics_pitch={physics.max_pitch_deg:.1f} "
        f"controlled_flight={physics.controlled_flight}"
    )
