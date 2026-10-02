import inspect
from flygym.compose import NeuroMechFly, FlatGroundWorld

world = FlatGroundWorld()
fly = NeuroMechFly()

print("--- FlatGroundWorld __init__ Signature ---")
print(inspect.signature(FlatGroundWorld.__init__))

print("\n--- World Attributes ---")
print([attr for attr in dir(world) if "contact" in attr or "sensor" in attr or "leg" in attr])

print("\n--- Fly Attributes ---")
print([attr for attr in dir(fly) if "contact" in attr or "sensor" in attr or "leg" in attr])
