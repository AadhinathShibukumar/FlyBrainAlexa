import numpy as np
from flygym.simulation import Simulation
from flygym.compose import NeuroMechFly, FlatGroundWorld

# 1. Orientation object satisfying FlyGym's internal 'format' requirement
class FlyOrientation:
    def __init__(self, quat=(1.0, 0.0, 0.0, 0.0)):
        self.format = "quat"
        self.quat = quat

    def as_kwargs(self):
        return {"quat": self.quat}

# 2. Instantiate fly and world
fly = NeuroMechFly()
world = FlatGroundWorld()

# 3. Bypass hardcoded ground contact sensor generation on world
world._add_ground_contact_sensors = lambda *args, **kwargs: None
world._set_ground_contact = lambda *args, **kwargs: None

# 4. Attach fly to arena
spawn_pos = (0, 0, 1.3)
spawn_rot = FlyOrientation()

world.add_fly(fly, spawn_position=spawn_pos, spawn_rotation=spawn_rot)

# 5. Build and reset simulation
sim = Simulation(world=world)
obs = sim.reset()

print(f"=== FlyGym loaded successfully! Actuators active: {len(fly.actuator_lookup)} ===")
