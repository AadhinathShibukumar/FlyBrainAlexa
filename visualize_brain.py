import time
import select
import sys
import numpy as np
import torch
import pyvista as pv
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== Launching Balanced Multi-Sensory Simulator on {device} ===")

# 1. Load Data
adj_matrix = torch.load("data/fly_adjacency_matrix.pt", map_location=device)
cell_groups = torch.load("data/cell_groups.pt")
coords = np.load("data/neuron_coords.npy")
num_neurons = adj_matrix.shape[0]

# 2. Extract Top Synaptic Edges
coo = adj_matrix.coalesce()
indices = coo.indices().cpu().numpy()
values = coo.values().cpu().numpy()

top_edge_idx = np.argsort(values)[-1500:]
src_indices = indices[0, top_edge_idx]
dst_indices = indices[1, top_edge_idx]

num_edges = len(top_edge_idx)
lines = np.empty((num_edges, 3), dtype=np.int64)
lines[:, 0] = 2
lines[:, 1] = src_indices
lines[:, 2] = dst_indices

edge_mesh = pv.PolyData(coords, lines.ravel())
edge_mesh["active"] = np.zeros(num_edges)

# 3. Setup Distinct Circuit Base Colors (Dim Resting Palette)
rgb_colors = np.full((num_neurons, 3), [0.03, 0.04, 0.07], dtype=np.float32)

olf_idx = cell_groups.get("olfactory", torch.tensor([])).numpy()
mb_idx = cell_groups.get("mushroom_body", torch.tensor([])).numpy()
motor_idx = cell_groups.get("motor", torch.tensor([])).numpy()
visual_idx = cell_groups.get("visual", torch.tensor([])).numpy()

if len(visual_idx) == 0:
    visual_idx = np.arange(0, min(3000, num_neurons))

# Vibrant Base Colors for Resting Circuits
if len(olf_idx) > 0:
    rgb_colors[olf_idx] = [0.0, 0.4, 0.7]     # Cyan/Blue: Olfactory
if len(visual_idx) > 0:
    rgb_colors[visual_idx] = [0.7, 0.3, 0.0]  # Orange: Visual
if len(mb_idx) > 0:
    rgb_colors[mb_idx] = [0.0, 0.5, 0.15]    # Green: Mushroom Body
if len(motor_idx) > 0:
    rgb_colors[motor_idx] = [0.6, 0.0, 0.5]   # Magenta: Motor

point_cloud = pv.PolyData(coords)
point_cloud["colors"] = rgb_colors.copy()

# 4. BALANCED LIF PARAMETERS (Prevents Runaway Saturation)
decay = 0.70             # Faster leak recovery
v_threshold = 1.0        # Firing threshold
v_reset = 0.0
synapse_weight = 0.005   # Scaled down to prevent full-brain cascade

v_membrane = torch.zeros(num_neurons, device=device)
spikes = torch.zeros(num_neurons, device=device)

# 5. Setup Live Analytics Plot
plt.ion()
fig, ax1 = plt.subplots(figsize=(6, 4), facecolor='#0a0a12')
plt.subplots_adjust(left=0.18, right=0.92, top=0.88, bottom=0.15)

ax1.set_facecolor('#12121c')
ax1.tick_params(colors='white', labelsize=8)
for spine in ax1.spines.values():
    spine.set_color('#333344')

history_steps = []
history_olf = []
history_vis = []
history_motor = []

l_olf, = ax1.plot([], [], label="Olfactory", color="#00ccff", lw=1.8)
l_vis, = ax1.plot([], [], label="Visual", color="#ff9900", lw=1.8)
l_motor, = ax1.plot([], [], label="Motor", color="#e633cc", lw=1.8)
ax1.set_title("Active Spike Propagation by Region", color="white", fontsize=10, pad=6)
ax1.set_xlim(0, 200)
ax1.set_ylim(0, 300)
ax1.legend(loc="upper right", facecolor="#12121c", edgecolor="#333344", labelcolor="white", fontsize=8)
ax1.grid(True, color="#222233", linestyle="--", alpha=0.5)

fig.canvas.draw()
fig.canvas.flush_events()

# 6. Setup 3D PyVista Plotter
plotter = pv.Plotter(title="Drosophila Brain - Multi-Sensory Interactive Simulation")
plotter.set_background("#030306")

plotter.add_mesh(
    point_cloud,
    scalars="colors",
    rgb=True,
    point_size=2.2,
    render_points_as_spheres=True
)

plotter.add_mesh(
    edge_mesh,
    scalars="active",
    cmap=["#030306", "#FFD700"],
    line_width=1.2,
    opacity=0.5,
    show_scalar_bar=False,
    clim=[0, 1]
)

plotter.add_text(
    "Cyan: Olfactory | Orange: Visual | Green: Mushroom Body | Magenta: Motor\nBright Yellow/White = Active Firing Neurons",
    position="upper_left",
    font_size=9
)

centroid = coords.mean(axis=0)
plotter.set_focus(centroid)
radius = 450.0

plotter.show(interactive_update=True)

def get_stimulus_pulse():
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline().strip()
        if line == '1': return 'olfactory'
        if line == '2': return 'visual'
        if line == '3': return 'motor'
        if line == '4': return 'dual'
    return None

print("\nReady! Enter 1 (Olfactory), 2 (Visual), 3 (Motor), or 4 (Dual) + Enter to trigger localized pulses:\n")

current_stimulus = None

for step in range(1, 201):
    user_input = get_stimulus_pulse()
    if user_input:
        current_stimulus = user_input
        print(f"--> [Step {step}] Pulse Triggered: {current_stimulus.upper()}")

    external_input = torch.zeros(num_neurons, device=device)
    if current_stimulus == 'olfactory' and len(olf_idx) > 0:
        external_input[cell_groups["olfactory"].to(device)] = 1.5
    elif current_stimulus == 'visual' and len(visual_idx) > 0:
        external_input[torch.tensor(visual_idx, device=device)] = 1.5
    elif current_stimulus == 'motor' and len(motor_idx) > 0:
        external_input[cell_groups["motor"].to(device)] = 1.5
    elif current_stimulus == 'dual':
        if len(olf_idx) > 0: external_input[cell_groups["olfactory"].to(device)] = 1.2
        if len(visual_idx) > 0: external_input[torch.tensor(visual_idx, device=device)] = 1.2

    current_stimulus = None

    # Compute LIF Dynamics
    synaptic_current = torch.sparse.mm(adj_matrix.t(), spikes.unsqueeze(1)).squeeze(1) * synapse_weight
    v_membrane = (v_membrane * decay) + synaptic_current + external_input

    spikes = (v_membrane >= v_threshold).float()
    v_membrane = torch.where(spikes > 0, torch.tensor(v_reset, device=device), v_membrane)

    spikes_np = spikes.cpu().numpy()
    active_mask = spikes_np.astype(bool)

    # Highlight ONLY currently active firing neurons in bright yellow
    frame_colors = rgb_colors.copy()
    frame_colors[active_mask] = [1.0, 1.0, 0.3]  # Bright yellow glow
    point_cloud["colors"] = frame_colors

    src_firing = active_mask[src_indices]
    edge_mesh["active"] = src_firing.astype(float)

    # Orbit Camera
    angle = np.radians(step * 1.8)
    cam_x = centroid[0] + radius * np.sin(angle)
    cam_z = centroid[2] + radius * np.cos(angle)
    plotter.camera_position = [(cam_x, centroid[1] + 30, cam_z), centroid, (0, 1, 0)]
    plotter.render()

    # Collect Analytics Metrics
    olf_spikes = int(spikes_np[olf_idx].sum()) if len(olf_idx) > 0 else 0
    vis_spikes = int(spikes_np[visual_idx].sum()) if len(visual_idx) > 0 else 0
    motor_spikes = int(spikes_np[motor_idx].sum()) if len(motor_idx) > 0 else 0

    history_steps.append(step)
    history_olf.append(olf_spikes)
    history_vis.append(vis_spikes)
    history_motor.append(motor_spikes)

    l_olf.set_data(history_steps, history_olf)
    l_vis.set_data(history_steps, history_vis)
    l_motor.set_data(history_steps, history_motor)

    max_spikes = max(max(history_olf + [1]), max(history_vis + [1]), max(history_motor + [1]))
    ax1.set_ylim(0, max(max_spikes * 1.3, 20))

    fig.canvas.draw()
    fig.canvas.flush_events()

    time.sleep(0.02)

plt.ioff()
plotter.show(interactive=True)
