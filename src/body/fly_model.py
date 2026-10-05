import numpy as np
import torch

from flygym.simulation import Simulation
from flygym.compose import (
    NeuroMechFly,
    FlatGroundWorld,
    ActuatorType,
    KinematicPosePreset,
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
        self.fly.add_joints(
            self.fly.skeleton,
            neutral_pose=KinematicPosePreset.NEUTRAL,
        )

        joint_dofs = list(self.fly.skeleton.iter_jointdofs())

        self.fly.add_actuators(
            joint_dofs,
            self.actuator_type,
            {jdof: 0.0 for jdof in joint_dofs},
        )

        world = FlatGroundWorld()

        world.add_fly(
            self.fly,
            spawn_position=(0, 0, 2.0),
            spawn_rotation=FlyOrientation(),
        )

        self.sim = Simulation(world=world)

        self.sim.reset()

        self.joint_order = self.fly.get_actuated_jointdofs_order(
            self.actuator_type
        )

        self.num_dofs = len(self.joint_order)
        self.mj_joint_lookup = {}

        for j_id in range(self.sim.mj_model.njnt):
            full_name = self.sim.mj_model.joint(j_id).name

            self.mj_joint_lookup[full_name] = j_id

            if "/" in full_name:
                self.mj_joint_lookup[
                    full_name.split("/")[-1]
                ] = j_id

        self.bio_limits_min = np.zeros(self.num_dofs)
        self.bio_limits_max = np.zeros(self.num_dofs)

        for i, jdof in enumerate(self.joint_order):

            j_id = self.mj_joint_lookup.get(
                jdof.name,
                -1,
            )

            if (
                j_id != -1
                and self.sim.mj_model.jnt_limited[j_id]
            ):
                r = self.sim.mj_model.jnt_range[j_id]

                self.bio_limits_min[i] = r[0]
                self.bio_limits_max[i] = r[1]

            else:
                self.bio_limits_min[i] = -DEFAULT_PHYSIO_BOUND
                self.bio_limits_max[i] = DEFAULT_PHYSIO_BOUND

        self.legs = [
            "LF",
            "LM",
            "LH",
            "RF",
            "RM",
            "RH",
        ]

        self.leg_dof_map = {
            leg: [
                i
                for i, jdof in enumerate(
                    self.joint_order
                )
                if (
                    f'-{leg.lower()}_' in jdof.name.lower()
                    or f'_{leg.lower()}_' in jdof.name.lower()
                )
            ]
            for leg in self.legs
        }

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

    def get_sensory_state(self):
        """Return per-leg contact force and normalized support feedback."""
        found, forces, _, positions, normals, tangents = (
            self.sim.get_ground_contact_info(self.fly.name)
        )
        contact_force = np.linalg.norm(forces, axis=1).astype(np.float32)
        contact = (found > 0.0).astype(np.float32)
        return {
            "leg_contact": contact,
            "contact_force": contact_force,
            "contact_position": positions.astype(np.float32),
            "contact_normal": normals.astype(np.float32),
            "contact_tangent": tangents.astype(np.float32),
            "contact_found": found.astype(np.float32),
            "mechanosensory_drive": float(contact.mean()),
        }
