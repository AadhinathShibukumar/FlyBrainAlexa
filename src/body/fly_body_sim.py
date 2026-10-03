import time
import numpy as np
import torch
import pyvista as pv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== Initializing Drosophila Embodied Simulator on {device} ===")

# 1. Load Brain Connectome
adj_matrix = torch.load("data/fly_adjacency_matrix.pt", map_location=device)
cell_groups = torch.load("data/cell_groups.pt")
coords = np.load("data/neuron_coords.npy")
num_neurons = adj_matrix.shape[0]

olf_idx = cell_groups.get("olfactory", torch.tensor([])).numpy()
mb_idx = cell_groups.get("mushroom_body", torch.tensor([])).numpy()
motor_idx = cell_groups.get("motor", torch.tensor([])).numpy()
visual_idx = cell_groups.get("visual", torch.tensor([])).numpy()

if len(visual_idx) == 0:
    visual_idx = np.arange(0, min(5000, num_neurons))

# 2. Build 3D Physical Fly Body Parts
thorax = pv.Sphere(radius=0.3, center=(0, 0, 0.6))
head = pv.Sphere(radius=0.18, center=(0, 0.35, 0.65))
abdomen = pv.Cylinder(radius=0.2, height=0.6, center=(0, -0.45, 0.55), direction=(0, 1, -0.2))

wing_l_base = pv.Cube(center=(-0.35, 0.0, 0.85), x_length=0.6, y_length=0.25, z_length=0.02)
wing_r_base = pv.Cube(center=(0.35, 0.0, 0.85), x_length=0.6, y_length=0.25, z_length=0.02)

legs_base = []
leg_offsets = [(-0.25, 0.15), (0.25, 0.15), (-0.3, 0.0), (0.3, 0.0), (-0.25, -0.2), (0.25, -0.2)]
for x, y in leg_offsets:
    legs_base.append(pv.Cylinder(radius=0.03, height=0.4, center=(x, y, 0.3), direction=(x*0.5, 0, -1)))

# 3. Setup Viewer
plotter = pv.Plotter(title="Fly Physical Embodiment - Neural Drive (Clamped Control)")
plotter.set_background("#020205")

plotter.add_mesh(thorax, color="#333344")
head_actor = plotter.add_mesh(head, color="#cc2222") # Red Compound Eyes
plotter.add_mesh(abdomen, color="#111118")

wing_l_actor = plotter.add_mesh(wing_l_base, color="#88aaff", opacity=0.6)
wing_r_actor = plotter.add_mesh(wing_r_base, color="#88aaff", opacity=0.6)

leg_actors = []
for leg_mesh in legs_base:
    leg_actors.append(plotter.add_mesh(leg_mesh, color="#555566"))

plotter.add_text(
    "Keys: [1] Olfactory (Wings) | [2] Visual (Head) | [3] Motor (Legs) | [0] Clear Pulse",
    position="upper_left",
    font_size=9
)

plotter.set_focus((0, 0, 0.5))
plotter.camera_position = [(1.5, 1.5, 1.2), (0, 0, 0.5), (0, 0, 1)]

# 4. Neural Parameters (Dampened Synapse Weight)
decay, v_threshold, v_reset = 0.70, 1.0, 0.0
synapse_weight = 0.001  # Lowered weight prevents runaway network explosion

v_membrane = torch.zeros(num_neurons, device=device)
spikes = torch.zeros(num_neurons, device=device)

step = 0
current_stimulus = None

def trigger_olf():
    global current_stimulus; current_stimulus = 'olfactory'; print("--> [PULSE 1] Olfactory Circuit")
def trigger_vis():
    global current_stimulus; current_stimulus = 'visual'; print("--> [PULSE 2] Visual Circuit")
def trigger_motor():
    global current_stimulus; current_stimulus = 'motor'; print("--> [PULSE 3] Motor Circuit")
def trigger_clear():
    global v_membrane, spikes, current_stimulus
    v_membrane.zero_(); spikes.zero_(); current_stimulus = None
    print("--> [RESET] Cleared membrane potentials")

plotter.add_key_event("1", trigger_olf)
plotter.add_key_event("2", trigger_vis)
plotter.add_key_event("3", trigger_motor)
plotter.add_key_event("0", trigger_clear)

# 5. Physics Simulation Loop Callback
def timer_callback(step_id):
    global step, v_membrane, spikes, current_stimulus
    step += 1

    external_input = torch.zeros(num_neurons, device=device)
    if current_stimulus == 'olfactory' and len(olf_idx) > 0:
        external_input[cell_groups["olfactory"].to(device)] = 1.2
    elif current_stimulus == 'visual' and len(visual_idx) > 0:
        external_input[torch.tensor(visual_idx, device=device)] = 1.2
    elif current_stimulus == 'motor' and len(motor_idx) > 0:
        external_input[cell_groups["motor"].to(device)] = 1.2

    current_stimulus = None

    # Compute SNN Dynamics
    synaptic_current = torch.sparse.mm(adj_matrix.t(), spikes.unsqueeze(1)).squeeze(1) * synapse_weight
    v_membrane = (v_membrane * decay) + synaptic_current + external_input

    spikes = (v_membrane >= v_threshold).float()
    v_membrane = torch.where(spikes > 0, torch.tensor(v_reset, device=device), v_membrane)

    spikes_np = spikes.cpu().numpy()

    # Read Neural Outputs
    raw_motor = spikes_np[motor_idx].sum() if len(motor_idx) > 0 else 0
    raw_vis = spikes_np[visual_idx].sum() if len(visual_idx) > 0 else 0
    raw_olf = spikes_np[olf_idx].sum() if len(olf_idx) > 0 else 0

    # SAFETY CLAMP: Restrict spike influence to normalized [0.0, 1.0] range
    motor_act = np.clip(raw_motor / 500.0, 0.0, 1.0)
    vis_act = np.clip(raw_vis / 500.0, 0.0, 1.0)
    olf_act = np.clip(raw_olf / 500.0, 0.0, 1.0)

    # Drive Clean Physical Kinematics
    wing_angle = np.sin(step * 0.8) * olf_act * 0.4
    leg_shift = np.sin(step * 0.3) * motor_act * 0.08

    # Animate Geometry cleanly without distorting scene viewport
    wing_l_actor.SetOrientation((0, 0, wing_angle * 35))
    wing_r_actor.SetOrientation((0, 0, -wing_angle * 35))

    for i, leg_actor in enumerate(leg_actors):
        phase = 1 if i % 2 == 0 else -1
        leg_actor.SetPosition((0, leg_shift * phase, 0))

plotter.add_timer_event(max_steps=2000, duration=20, callback=timer_callback)

print("\nFixed script ready! Press 1, 2, 3, or 0 on the viewer.")
plotter.show()
