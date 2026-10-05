import numpy as np
import time
import mujoco
import mujoco.viewer
from gait_controller import TripodGait
from src.body.balance_controller import BalanceController
from src.body.aerodynamics import WingAerodynamics
from src.body.flight_controller import FlightController
from src.body.flight_metrics import FlightMetrics
from src.body.flight_safety import FlightSafetyLimiter
from src.body.locomotion_controller import StanceSwingController
from src.body.muscle_activation import MuscleActivation
from src.body.proprioception import Proprioception
from src.body.recovery_controller import RecoveryController
from src.body.support_balance import SupportBalance
from src.body.takeoff_controller import FlightPhase, TakeoffController
from src.body.telemetry_logger import TelemetryLogger
from src.body.wing_controller import WingbeatController
from src.brain.behavior_controller import BehaviorController
from src.brain.connectome_brain import ConnectomeBrain
from flygym.simulation import Simulation
from flygym.compose import (
    NeuroMechFly,
    FlatGroundWorld,
    ActuatorType,
    KinematicPosePreset,
)
class FlyOrientation:
    def __init__(self, quat=(1.0, 0.0, 0.0, 0.0)):
        self.format = 'quat'
        self.quat = quat
    def as_kwargs(self):
        return {'quat': self.quat}

# 1. Setup Fly & World
fly = NeuroMechFly()
fly.skeleton = fly._get_base_skeleton()
fly.add_joints(
    fly.skeleton,
    neutral_pose=KinematicPosePreset.NEUTRAL,
)

joint_dofs = list(fly.skeleton.iter_jointdofs())
actuator_type = ActuatorType.POSITION
fly.add_actuators(
    joint_dofs,
    actuator_type,
    {jdof: 0.0 for jdof in joint_dofs},
    kp=20.0,
    forcerange=(-30.0, 30.0),
)

world = FlatGroundWorld()
world.add_fly(fly, spawn_position=(0, 0, 0.3), spawn_rotation=FlyOrientation())

# 2. Setup Simulation
sim = Simulation(world=world)
sim.reset()

# 3. Retrieve Joint Order & Limits
joint_order = fly.get_actuated_jointdofs_order(actuator_type)
num_dofs = len(joint_order)

mj_joint_lookup = {}
for j_id in range(sim.mj_model.njnt):
    full_name = sim.mj_model.joint(j_id).name
    mj_joint_lookup[full_name] = j_id
    if '/' in full_name:
        mj_joint_lookup[full_name.split('/')[-1]] = j_id

bio_limits_min = np.zeros(num_dofs)
bio_limits_max = np.zeros(num_dofs)
DEFAULT_PHYSIO_BOUND = 0.8 

for i, jdof in enumerate(joint_order):
    j_id = mj_joint_lookup.get(jdof.name, -1)
    if j_id != -1 and sim.mj_model.jnt_limited[j_id]:
        r = sim.mj_model.jnt_range[j_id]
        bio_limits_min[i], bio_limits_max[i] = r[0], r[1]
    else:
        bio_limits_min[i], bio_limits_max[i] = -DEFAULT_PHYSIO_BOUND, DEFAULT_PHYSIO_BOUND

# Let the neutral body settle onto its feet before starting the controller.
sim.set_actuator_inputs(
    fly.name,
    ActuatorType.POSITION,
    np.full(num_dofs, 0.5),
)
for _ in range(300):
    sim.step()

legs = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
PHYSICS_SUBSTEPS = 50
NEURAL_GAIN = 1.0
STABILIZE_ROOT_FOR_DEBUG = False
ALLOW_FORWARD_TRANSLATION = True
BALANCE_CONTROLLER_ENABLED = True
FLIGHT_TEST_ENABLED = False
WINGBEAT_IN_WALKING = True
WING_WALKING_AMPLITUDE = 0.55
TELEMETRY_PATH = "data/live_telemetry.csv"
leg_dof_map = {
    leg: [
        i for i, jdof in enumerate(joint_order) 
        if f'-{leg.lower()}_' in jdof.name.lower() or f'_{leg.lower()}_' in jdof.name.lower()
    ]
    for leg in legs
}
wing_dof_map = {
    "left": [
        i for i, jdof in enumerate(joint_order)
        if "-l_wing-" in jdof.name.lower()
    ],
    "right": [
        i for i, jdof in enumerate(joint_order)
        if "-r_wing-" in jdof.name.lower()
    ],
}
# Keep the body and legs conservative while giving the wing hinges enough
# position authority to track the neural wingbeat.
for wing_index in wing_dof_map["left"] + wing_dof_map["right"]:
    sim.mj_model.actuator_gainprm[wing_index, 0] = 60.0
    sim.mj_model.actuator_forcerange[wing_index] = (-45.0, 45.0)
wing_qpos_addresses = []
for index in wing_dof_map["left"] + wing_dof_map["right"]:
    joint_name = joint_order[index].name
    joint_id = mj_joint_lookup[joint_name]
    wing_qpos_addresses.append(sim.mj_model.jnt_qposadr[joint_id])
joint_qpos_addresses = np.array([
    sim.mj_model.jnt_qposadr[mj_joint_lookup[jdof.name]]
    for jdof in joint_order
])
joint_qvel_addresses = np.array([
    sim.mj_model.jnt_dofadr[mj_joint_lookup[jdof.name]]
    for jdof in joint_order
])

# 4. Launch Live Interactive Viewer
filtered_joint_targets = np.zeros(num_dofs)
print("Launching live interactive 3D fly simulation...")
with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data) as viewer:
    viewer.cam.lookat[:] = sim.mj_data.qpos[:3]
    viewer.cam.distance = 8.0
    gait = TripodGait()
    gait.frequency = 20.0
    locomotion = StanceSwingController()
    wings = WingbeatController(frequency_hz=12.0)
    aerodynamics = WingAerodynamics(
        sim.mj_model,
        left_body_name=f"{fly.name}/l_wing",
        right_body_name=f"{fly.name}/r_wing",
        root_body_name=f"{fly.name}/c_thorax",
    )
    flight = FlightController()
    takeoff = TakeoffController()
    safety = FlightSafetyLimiter()
    metrics = FlightMetrics()
    balance = BalanceController()
    muscle_activation = MuscleActivation(num_dofs)
    wing_activation = MuscleActivation(
        len(wing_dof_map["left"]) + len(wing_dof_map["right"]),
        activation_rate=80.0,
    )
    proprioception = Proprioception(bio_limits_min, bio_limits_max)
    support_balance = SupportBalance()
    recovery = RecoveryController()
    brain = ConnectomeBrain()
    behavior = BehaviorController()
    mechanosensory_drive = 0.0
    leg_contacts = np.zeros(len(legs), dtype=np.float32)
    contact_forces = np.zeros(len(legs), dtype=np.float32)
    positions = np.zeros((len(legs), 3), dtype=np.float32)
    support_error = np.zeros(2, dtype=np.float32)
    contact_confidence = np.zeros(len(legs), dtype=np.float32)
    debug_root_xy = sim.mj_data.qpos[:2].copy()
    debug_root_y = float(debug_root_xy[1])
    filtered_joint_targets = np.zeros(num_dofs)
    lift_ratio = 0.0
    roll = 0.0
    pitch = 0.0
    wing_differential = 0.0
    wing_drive = 0.0
    previous_motor_drive = 0.0
    proprioceptive_drive = 0.0
    velocity_drive = 0.0
    contact_force_drive = 0.0
    startup_contacts, _, _, _, _, _ = sim.get_ground_contact_info(fly.name)
    print(
        f"STARTUP settled_height={sim.mj_data.qpos[2]:.2f} "
        f"contacts={int(np.count_nonzero(startup_contacts > 0))} "
        f"mode={'flight-test' if FLIGHT_TEST_ENABLED else 'walking'}"
    )
    telemetry = TelemetryLogger(TELEMETRY_PATH)
    
    frame = 0
    while viewer.is_running():
        frame += 1
        if frame == 1:
            contact_found, _, _, _, _, _ = sim.get_ground_contact_info(fly.name)
            leg_contacts = (contact_found > 0.0).astype(np.float32)
            mechanosensory_drive = float(leg_contacts.mean())
        brain_state = brain.step(
            olfactory_drive=2.0,
            mechanosensory_drive=mechanosensory_drive,
            proprioceptive_drive=proprioceptive_drive,
            contact_force_drive=contact_force_drive,
        )
        behavior_command = behavior.step(
            motor_drive=brain_state["motor_drive"],
            leg_contacts=leg_contacts,
        )
        # The flight test starts on the ground. Seed the takeoff controller
        # from the measured contact state instead of waiting for a previous
        # wing command to create contact.
        if FLIGHT_TEST_ENABLED:
            wing_drive, flight_phase = takeoff.step(
                motor_drive=brain_state["motor_drive"],
                contacts=leg_contacts,
                height=float(sim.mj_data.qpos[2]),
                dt=PHYSICS_SUBSTEPS * sim.mj_model.opt.timestep,
            )
        else:
            wing_drive = 0.0
            flight_phase = FlightPhase.GROUNDED
        recovery_scale = recovery.scale(
            support_count=int(np.count_nonzero(leg_contacts > 0.0)),
            roll=roll,
            pitch=pitch,
            root_velocity=sim.mj_data.qvel[:2],
        )
        wing_motion_drive = (
            float(
                np.clip(brain_state["motor_drive"], 0.0, 1.0)
                * WING_WALKING_AMPLITUDE
            )
            if WINGBEAT_IN_WALKING
            else 0.0
        )
        wing_motion_drive *= recovery_scale
        neural_delta = brain_state["motor_drive"] - previous_motor_drive
        previous_motor_drive = brain_state["motor_drive"]
        neural_amplitudes = np.array(
            [
                NEURAL_GAIN
                * behavior_command.locomotion_drive
                * brain_state["leg_drives"][leg]
                for leg in legs
            ]
        )
        neural_amplitudes *= recovery_scale
        gait.frequency = float(
            np.clip(6.0 + 18.0 * brain_state["motor_drive"], 4.0, 24.0)
        )
        gait_amplitudes, locomotion_state = locomotion.step(
            contacts=leg_contacts,
            neural_amplitudes=neural_amplitudes,
            contact_confidence=contact_confidence,
        )
        gait_activations = gait.step(
            dt=PHYSICS_SUBSTEPS * sim.mj_model.opt.timestep,
            amplitudes=gait_amplitudes,
            contacts=leg_contacts,
        )
        # Unmapped joints must remain at the normalized midpoint, not at the
        # lower biological limit.
        raw_normalized = np.full(num_dofs, 0.5)
        
        for leg_idx, leg in enumerate(legs):
            dof_indices = leg_dof_map[leg]
            if len(dof_indices) > 0:
                acts = gait_activations[leg_idx]
                n_assign = min(len(dof_indices), len(acts))
                raw_normalized[dof_indices[:n_assign]] = acts[:n_assign]
        actual_normalized = proprioception.normalized_positions(
            sim.mj_data.qpos,
            joint_qpos_addresses,
        )
        actual_velocity = proprioception.normalized_velocity(
            sim.mj_data.qvel,
            joint_qvel_addresses,
        )
        tracking_scale = proprioception.tracking_scale(
            raw_normalized,
            actual_normalized,
            leg_dof_map,
        )
        proprioceptive_drive = float(
            np.clip(1.0 - np.mean(tracking_scale), 0.0, 1.0)
        )
        velocity_drive = float(np.clip(np.mean(actual_velocity), 0.0, 1.0))
        proprioceptive_drive = float(
            np.clip(0.7 * proprioceptive_drive + 0.3 * velocity_drive, 0.0, 1.0)
        )
        for leg_index, leg in enumerate(legs):
            indices = leg_dof_map[leg]
            raw_normalized[indices] = 0.5 + (
                raw_normalized[indices] - 0.5
            ) * tracking_scale[leg_index]
        raw_normalized, support_error = support_balance.correction(
            raw_normalized,
            leg_dof_map,
            positions,
            contact_forces,
            sim.mj_data.qpos[:3],
            sim.mj_data.qvel[:2],
            sim.mj_data.qpos[3:7],
        )
        wing_targets = wings.step(
            drive=wing_motion_drive,
            dt=PHYSICS_SUBSTEPS * sim.mj_model.opt.timestep,
            left_indices=wing_dof_map["left"],
            right_indices=wing_dof_map["right"],
        )
        raw_normalized[wing_dof_map["left"]] = wing_targets[:3]
        raw_normalized[wing_dof_map["right"]] = wing_targets[3:]
        if BALANCE_CONTROLLER_ENABLED:
            raw_normalized = balance.apply(
                raw_normalized,
                leg_dof_map,
                sim.mj_data.qpos[3:7].copy(),
            )
        
        control_dt = PHYSICS_SUBSTEPS * sim.mj_model.opt.timestep
        normalized_targets = muscle_activation.step(
            raw_normalized,
            control_dt,
        )
        wing_indices = wing_dof_map["left"] + wing_dof_map["right"]
        normalized_targets[wing_indices] = wing_activation.step(
            raw_normalized[wing_indices],
            control_dt,
        )
        filtered_joint_targets = bio_limits_min + normalized_targets * (
            bio_limits_max - bio_limits_min
        )
        filtered_joint_targets = np.clip(
            filtered_joint_targets,
            bio_limits_min,
            bio_limits_max,
        )
        
        sim.set_actuator_inputs(fly.name, actuator_type, filtered_joint_targets)
        for _ in range(PHYSICS_SUBSTEPS):
            roll, pitch, wing_differential = flight.corrections(
                sim.mj_data.qpos[3:7].copy()
            )
            safe_drive = safety.limit(wing_drive, roll, pitch)
            aerodynamics.stabilize(
                sim.mj_data,
                roll=roll,
                pitch=pitch,
                quaternion=sim.mj_data.qpos[3:7].copy(),
                angular_velocity=sim.mj_data.qvel[3:6].copy(),
            )
            lift_ratio = aerodynamics.apply(
                sim.mj_data,
                phase=wings.phase,
                drive=safe_drive,
                differential=wing_differential,
            )
            sim.step()
            aerodynamics.clear(sim.mj_data)
            if STABILIZE_ROOT_FOR_DEBUG:
                # Temporary locomotion aid: suppress lateral drift and tipping
                # while allowing forward translation to be measured.
                if not ALLOW_FORWARD_TRANSLATION:
                    sim.mj_data.qpos[:2] = debug_root_xy
                else:
                    sim.mj_data.qpos[1] = debug_root_y
                sim.mj_data.qvel[1] = 0.0
                sim.mj_data.qpos[2] = max(sim.mj_data.qpos[2], 3.0)
                sim.mj_data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
                sim.mj_data.qvel[3:6] = 0.0
        contact_found, forces, _, positions, _, _ = (
            sim.get_ground_contact_info(fly.name)
        )
        contact_forces = np.linalg.norm(forces, axis=1)
        contact = (contact_found > 0.0).astype(np.float32)
        contact_confidence = np.clip(contact_forces / 10.0, 0.0, 1.0)
        mechanosensory_drive = float(contact.mean())
        contact_force_drive = float(
            np.clip(np.mean(contact_forces) / 10.0, 0.0, 1.0)
        )
        leg_contacts = contact
        metrics.update(
            float(sim.mj_data.qpos[2]),
            sim.mj_data.qpos[3:7].copy(),
            leg_contacts,
        )
        telemetry.write(
            {
                "frame": frame,
                "sim_time": sim.mj_data.time,
                "motor_drive": brain_state["motor_drive"],
                "left_drive": brain_state["left_drive"],
                "right_drive": brain_state["right_drive"],
                "contact_fraction": mechanosensory_drive,
                "force_drive": contact_force_drive,
                "proprioceptive_drive": proprioceptive_drive,
                "velocity_drive": velocity_drive,
                "recovery_scale": recovery_scale,
                "height": sim.mj_data.qpos[2],
                "roll_deg": np.degrees(roll),
                "pitch_deg": np.degrees(pitch),
                "wing_motion": wing_motion_drive,
                "wing_drive": wing_drive,
            }
        )
        if frame <= 20 or frame % 100 == 0:
            airborne = mechanosensory_drive == 0.0 and sim.mj_data.qpos[2] > 1.5
            print(
                f"frame={frame} sim_time={sim.mj_data.time:.3f}s "
                f"motor={brain_state['motor_drive']:.3f} "
                f"neural_delta={neural_delta:+.3f} "
                f"gait_hz={gait.frequency:.1f} "
                f"legs={np.round([brain_state['leg_drives'][leg] for leg in legs], 2).tolist()} "
                f"track={np.round(tracking_scale, 2).tolist()} "
                f"proprio={proprioceptive_drive:.2f} "
                f"velocity={velocity_drive:.2f} "
                f"force_sense={contact_force_drive:.2f} "
                f"activation={normalized_targets.min():.2f}-"
                f"{normalized_targets.max():.2f} "
                f"behavior={behavior_command.name} "
                f"contact={mechanosensory_drive:.3f} "
                f"support={locomotion_state.support_count}/6 "
                f"propulsion={locomotion_state.propulsion_enabled} "
                f"lift={lift_ratio:.2f}xweight "
                f"roll={np.degrees(roll):.1f}deg "
                f"pitch={np.degrees(pitch):.1f}deg "
                f"wing_diff={wing_differential:.2f} "
                f"height={sim.mj_data.qpos[2]:.2f} "
                f"airborne={airborne} "
                f"flight_phase={flight_phase.value} "
                f"wing_drive={wing_drive:.2f} "
                f"wing_motion={wing_motion_drive:.2f} "
                f"wing_target={np.round(filtered_joint_targets[wing_indices], 2).tolist()} "
                f"wing_actual={np.round(sim.mj_data.qpos[wing_qpos_addresses], 2).tolist()} "
                f"max_h={metrics.max_height:.2f} "
                f"air_frac={metrics.airborne_fraction:.2f} "
                f"wing_phase={wings.phase:.2f} "
                f"phases={''.join(p[0].upper() for p in locomotion_state.phases)} "
                f"forces={np.round(contact_forces, 1).tolist()} "
                f"support_conf={np.round(contact_confidence, 2).tolist()} "
                f"support_error={np.round(support_error, 2).tolist()} "
                f"root_vel={np.round(sim.mj_data.qvel[:3], 2).tolist()} "
                f"attitude={np.round(np.degrees([roll, pitch]), 1).tolist()} "
                f"recovery={recovery_scale:.2f} "
                f"foot_z={np.round(positions[:, 2], 2).tolist()}"
            )
        
        # Follow the fly smoothly; FlyGym uses millimetres for world coordinates.
        viewer.cam.lookat[:] = (
            0.95 * viewer.cam.lookat
            + 0.05 * sim.mj_data.qpos[:3]
        )
        viewer.sync()
        time.sleep(0.01)
    telemetry.close()
