from flygym.simulation import Simulation
from flygym.compose import NeuroMechFly, FlatGroundWorld

class FlyOrientation:
    def __init__(self, quat=(1.0, 0.0, 0.0, 0.0)):
        self.format = "quat"
        self.quat = quat

    def as_kwargs(self):
        return {"quat": self.quat}

fly = NeuroMechFly()
world = FlatGroundWorld()

world._add_ground_contact_sensors = lambda *args, **kwargs: None
world._set_ground_contact = lambda *args, **kwargs: None

world.add_fly(fly, spawn_position=(0, 0, 1.3), spawn_rotation=FlyOrientation())
sim = Simulation(world=world)

# 1. Inspect simulation attributes
print("--- Simulation Reset Return ---")
res = sim.reset()
print("Reset Return:", res)

print("\n--- Simulation Object Attributes ---")
print([a for a in dir(sim) if not a.startswith("_")])

print("\n--- Fly Object Attributes ---")
print([a for a in dir(fly) if "act" in a.lower() or "joint" in a.lower() or "control" in a.lower()])

if hasattr(fly, "actuators"):
    print("\nFly .actuators type:", type(fly.actuators))
    if hasattr(fly.actuators, "__len__"):
        print("Fly .actuators count:", len(fly.actuators))

