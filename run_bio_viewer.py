import torch
import numpy as np
import mujoco
import mujoco.viewer
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
world.add_fly(fly, spawn_position=(0, 0, 0.6), spawn_rotation=FlyOrientation())

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
leg_dof_map = {
    leg: [
        i for i, jdof in enumerate(joint_order) 
        if f'-{leg.lower()}_' in jdof.name.lower() or f'_{leg.lower()}_' in jdof.name.lower()
    ]
    for leg in legs
}

# 4. Step Simulation with Native MuJoCo Interactive Viewer
filtered_joint_targets = np.zeros(num_dofs)
alpha = 0.15

with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data) as viewer:
    for step in range(500):
        if not viewer.is_running():
            break
            
        snn_activations = torch.rand(6, 24)
        raw_normalized = np.zeros(num_dofs)
        
        for leg_idx, leg in enumerate(legs):
            dof_indices = leg_dof_map[leg]
            if len(dof_indices) > 0:
                acts = snn_activations[leg_idx].detach().cpu().numpy()
                n_assign = min(len(dof_indices), len(acts))
                raw_normalized[dof_indices[:n_assign]] = acts[:n_assign]
        
        unclamped_targets = bio_limits_min + raw_normalized * (bio_limits_max - bio_limits_min)
        clamped_targets = np.clip(unclamped_targets, bio_limits_min, bio_limits_max)
        filtered_joint_targets = alpha * clamped_targets + (1 - alpha) * filtered_joint_targets
        
        sim.set_actuator_inputs(fly.name, actuator_type, filtered_joint_targets)
        sim.step()
        
        viewer.sync()
