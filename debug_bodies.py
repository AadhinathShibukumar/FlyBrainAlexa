from flygym.compose import NeuroMechFly

fly = NeuroMechFly()

# Check contact_sensor_placements default value
print("Default contact_sensor_placements:", getattr(fly, "contact_sensor_placements", "Not set"))

# Print body names present in the model spec
if hasattr(fly, "mjcf_root"):
    # If MjSpec
    if hasattr(fly.mjcf_root, "bodies"):
        body_names = [b.name for b in fly.mjcf_root.bodies]
        print("\nFirst 15 Body Names in MjSpec:")
        print(body_names[:15])
    # If PyMJCF
    elif hasattr(fly.mjcf_root, "find_all"):
        bodies = fly.mjcf_root.find_all("body")
        print("\nFirst 15 Body Names in PyMJCF:")
        print([b.name for b in bodies[:15]])
