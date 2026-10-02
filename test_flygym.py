from flygym.simulation import Simulation
from flygym.compose import NeuroMechFly, FlatGroundWorld

# 1. Initialize the official 3D NeuroMechFly body and ground world
fly = NeuroMechFly()
world = FlatGroundWorld(fly=fly)

# 2. Build and reset the simulation
sim = Simulation(world=world)
obs = sim.reset()

print(f"=== FlyGym loaded successfully! Actuators active: {len(fly.actuator_lookup)} ===")
