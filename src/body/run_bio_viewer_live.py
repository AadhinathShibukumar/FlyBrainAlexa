import numpy as np
import time
import mujoco
import mujoco.viewer
from gait_controller import TripodGait
from src.brain.behavior_controller import BehaviorController
from src.brain.connectome_brain import ConnectomeBrain
from flygym.simulation import Simulation
from flygym.compose import NeuroMechFly, FlatGroundWorld, ActuatorType

class FlyOrientation:
    def __init__(self, quat=(1.0, 0.0, 0.0, 0.0)):
        self.format = 'quat'
        self.quat = quat
    def as_kwargs(self):
        return {'quat': self.quat}

# 1. Setup Fly & World
fly = NeuroMechFly()
fly.skeleton = fly._get_base_skeleton()
fly.add_joints(fly.skeleton)

joint_dofs = list(fly.skeleton.iter_jointdofs())
actuator_type = ActuatorType.POSITION
fly.add_actuators(joint_dofs, actuator_type, {jdof: 0.0 for jdof in joint_dofs})

world = FlatGroundWorld()
world.add_fly(fly, spawn_position=(0, 0, 2.0), spawn_rotation=FlyOrientation())

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

legs = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
PHYSICS_SUBSTEPS = 50
NEURAL_GAIN = 1.0
STABILIZE_ROOT_FOR_DEBUG = True
ALLOW_FORWARD_TRANSLATION = False
leg_dof_map = {
    leg: [
        i for i, jdof in enumerate(joint_order) 
        if f'-{leg.lower()}_' in jdof.name.lower() or f'_{leg.lower()}_' in jdof.name.lower()
    ]
    for leg in legs
}

# 4. Launch Live Interactive Viewer
filtered_joint_targets = np.zeros(num_dofs)
alpha = 0.15

print("Launching live interactive 3D fly simulation...")
with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data) as viewer:
    viewer.cam.lookat[:] = sim.mj_data.qpos[:3]
    viewer.cam.distance = 8.0
    gait = TripodGait()
    brain = ConnectomeBrain()
    behavior = BehaviorController()
    mechanosensory_drive = 0.0
    leg_contacts = np.zeros(len(legs), dtype=np.float32)
    contact_forces = np.zeros(len(legs), dtype=np.float32)
    debug_root_xy = sim.mj_data.qpos[:2].copy()
    debug_root_y = float(debug_root_xy[1])
    filtered_joint_targets = np.zeros(num_dofs)
    
    frame = 0
    while viewer.is_running():
        frame += 1
        brain_state = brain.step(
            olfactory_drive=2.0,
            mechanosensory_drive=mechanosensory_drive,
        )
        behavior_command = behavior.step(
            motor_drive=brain_state["motor_drive"],
            leg_contacts=leg_contacts,
        )
        gait_activations = gait.step(
            amplitudes=[
                NEURAL_GAIN
                * behavior_command.locomotion_drive
                * brain_state["leg_drives"][leg]
                for leg in legs
            ],
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
        
        unclamped_targets = bio_limits_min + raw_normalized * (bio_limits_max - bio_limits_min)
        clamped_targets = np.clip(unclamped_targets, bio_limits_min, bio_limits_max)
        filtered_joint_targets = (
            alpha * clamped_targets
            + (1 - alpha) * filtered_joint_targets
        )
        
        sim.set_actuator_inputs(fly.name, actuator_type, filtered_joint_targets)
        for _ in range(PHYSICS_SUBSTEPS):
            sim.step()
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
        sensory_state = {
            "leg_contact": np.zeros(len(legs), dtype=np.float32),
            "contact_force": np.zeros(len(legs), dtype=np.float32),
        }
        for index, leg in enumerate(legs):
            sensor_id = sim.mj_model.sensor(
                f"ground_contact_{leg.lower()}_leg"
            ).id
            start = sim.mj_model.sensor_adr[sensor_id]
            size = sim.mj_model.sensor_dim[sensor_id]
            force_norm = np.linalg.norm(
                sim.mj_data.sensordata[start:start + size]
            )
            sensory_state["contact_force"][index] = force_norm
            sensory_state["leg_contact"][index] = float(force_norm > 2.0)
        contact = sensory_state["leg_contact"]
        contact_forces = sensory_state["contact_force"]
        mechanosensory_drive = float(contact.mean())
        leg_contacts = contact
        if frame % 100 == 0:
            print(
                f"frame={frame} sim_time={sim.mj_data.time:.3f}s "
                f"motor={brain_state['motor_drive']:.3f} "
                f"behavior={behavior_command.name} "
                f"contact={mechanosensory_drive:.3f} "
                f"support={int(np.count_nonzero(leg_contacts > 0.05))}/6 "
                f"forces={np.round(contact_forces, 1).tolist()}"
            )
        
        # Follow the fly smoothly; FlyGym uses millimetres for world coordinates.
        viewer.cam.lookat[:] = (
            0.95 * viewer.cam.lookat
            + 0.05 * sim.mj_data.qpos[:3]
        )
        viewer.sync()
        time.sleep(0.01)
