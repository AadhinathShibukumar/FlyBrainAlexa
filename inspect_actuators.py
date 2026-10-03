import inspect
from flygym.simulation import Simulation
from flygym.compose import NeuroMechFly, FlatGroundWorld, ActuatorType

fly = NeuroMechFly()

print("--- add_actuators Signature ---")
print(inspect.signature(fly.add_actuators))

# Add position actuators for all leg joints
fly.add_actuators(actuator_type=ActuatorType.POSITION)

world = FlatGroundWorld()
world._add_ground_contact_sensors = lambda *args, **kwargs: None
world._set_ground_contact = lambda *args, **kwargs: None

world.add_fly(fly, spawn_position=(0, 0, 1.3))
sim = Simulation(world=world)
sim.reset()

print("\n--- Registered Actuator Types in Simulation ---")
registered_types = list(sim._intern_actuatorids_by_type_by_fly.keys())
print(registered_types)

for act_type in registered_types:
    flies = list(sim._intern_actuatorids_by_type_by_fly[act_type].keys())
    print(f"Type: {act_type} -> Flies: {flies}")
