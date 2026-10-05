import numpy as np
import torch

from flygym.simulation import Simulation
from flygym.compose import (
    NeuroMechFly,
    FlatGroundWorld,
    ActuatorType,
)

DEFAULT_PHYSIO_BOUND = 0.8


class FlyOrientation:
    def __init__(self, quat=(1.0, 0.0, 0.0, 0.0)):
        self.format = "quat"
        self.quat = quat

    def as_kwargs(self):
        return {"quat": self.quat}


class FlyBody:

    def __init__(self):

        self.actuator_type = ActuatorType.POSITION

        self.fly = NeuroMechFly()
        self.fly.skeleton = self.fly._get_base_skeleton()
        self.fly.add_joints(self.fly.skeleton)

        joint_dofs = list(self.fly.skeleton.iter_jointdofs())

        self.fly.add_actuators(
            joint_dofs,
            self.actuator_type,
            {jdof: 0.0 for jdof in joint_dofs},
        )

        world = FlatGroundWorld()

        world.add_fly(
            self.fly,
            spawn_position=(0, 0, 0.2),
            spawn_rotation=FlyOrientation(),
        )

        self.sim = Simulation(world=world)

        self.sim.reset()

        self.joint_order = self.fly.get_actuated_jointdofs_order(
            self.actuator_type
        )

        self.num_dofs = len(self.joint_order)

        self.filtered_joint_targets = np.zeros(self.num_dofs)

        self.alpha = 0.15

    def step(self, joint_targets):

        self.filtered_joint_targets = (
            self.alpha * joint_targets
            + (1 - self.alpha) * self.filtered_joint_targets
        )

        self.sim.set_actuator_inputs(
            self.fly.name,
            self.actuator_type,
            self.filtered_joint_targets,
        )

        self.sim.step()

    def get_sim(self):
        return self.sim

    def get_fly(self):
        return self.fly
