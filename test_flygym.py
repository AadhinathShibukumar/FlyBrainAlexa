import numpy as np
from flygym.simulation import Simulation
from flygym.compose import NeuroMechFly, FlatGroundWorld

class FlyOrientation:
    def __init__(self, quat=(1.0, 0.0, 0.0, 0.0)):
        self.format = "quat"
        self.quat = quat

    def as_kwargs(self):
        return {"quat": self.quat}

# 1. Initialize fly with contact sensors disabled to prevent MJCF body lookup errors
fly = NeuroMechFly(contact_sensor_placements=())
world = FlatGroundWorld()

# 2. Attach fly to arena
spawn_pos = (0, 0, 1.3)
spawn_rot = FlyOrientation()

world.add_fly(fly, spawn_position=spawn_pos, spawn_rotation=spawn_rot)

# 3. Build and reset simulation
sim = Simulation(world=world)
obs = sim.reset()

print(f"=== FlyGym loaded successfully! Actuators active: {len(fly.actuator_lookup)} ===")
