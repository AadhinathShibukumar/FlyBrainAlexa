import numpy as np
from flygym.simulation import Simulation
from flygym.compose import NeuroMechFly, FlatGroundWorld, ActuatorType

class FlyOrientation:
    def __init__(self, quat=(1.0, 0.0, 0.0, 0.0)):
        self.format = "quat"
        self.quat = quat

    def as_kwargs(self):
        return {"quat": self.quat}

# 1. Instantiate fly and initialize base skeleton
fly = NeuroMechFly()
fly.skeleton = fly._get_base_skeleton()

# 2. Add joints to the MJCF tree passing the skeleton explicitly
fly.add_joints(fly.skeleton)

# 3. Add position actuators across registered joint DOFs
joint_dofs = list(fly.skeleton.iter_jointdofs())
neutral_inputs = {jdof: 0.0 for jdof in joint_dofs}
fly.add_actuators(
    jointdofs=joint_dofs,
    actuator_type=ActuatorType.POSITION,
    neutral_input=neutral_inputs
)

# 4. Instantiate world and bypass legacy ground contact patch
world = FlatGroundWorld()
world._add_ground_contact_sensors = lambda *args, **kwargs: None
world._set_ground_contact = lambda *args, **kwargs: None

# 5. Attach fly to arena
world.add_fly(fly, spawn_position=(0, 0, 1.3), spawn_rotation=FlyOrientation())

# 6. Build and reset simulation
sim = Simulation(world=world)
sim.reset()

# 7. Drive actuation inputs and step simulation
pos_key = ActuatorType.POSITION
joint_order = fly.get_actuated_jointdofs_order(pos_key)
num_joints = len(joint_order)

neutral_action_map = fly.jointdof_to_neutralaction_by_type[pos_key]
neutral_action = np.array([neutral_action_map[jdof] for jdof in joint_order])

sim.set_actuator_inputs(fly.name, pos_key, neutral_action)
sim.step()

# 8. Retrieve joint angles by specifying fly.name
angles = sim.get_joint_angles(fly.name)

print(f"\n=== FlyGym v2 Closed-Loop Simulation Test Passed! ===")
print(f"Fly Name: {fly.name}")
print(f"Actuator Type Registered: {pos_key}")
print(f"Actuated Joint DOFs Count: {num_joints}")
print(f"Sample Joint Order: {joint_order[:5]}")
print(f"Joint Angles Readout Shape: {angles.shape}")
print(f"First 5 Joint Angle Readouts (rad): {angles[:5]}")
