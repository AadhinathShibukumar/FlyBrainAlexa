import time
import numpy as np
import torch
import pyvista as pv
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== Launching Connectome Analytics & 3D Fly Brain Simulator on {device} ===")

# 1. Load Data
adj_matrix = torch.load("data/fly_adjacency_matrix.pt", map_location=device)
cell_groups = torch.load("data/cell_groups.pt")
coords = np.load("data/neuron_coords.npy")
num_neurons = adj_matrix.shape[0]

# 2. Extract Top Active Synaptic Edges
coo = adj_matrix.coalesce()
indices = coo.indices().cpu().numpy()
values = coo.values().cpu().numpy()

top_edge_idx = np.argsort(values)[-2000:]
src_indices = indices[0, top_edge_idx]
dst_indices = indices[1, top_edge_idx]

num_edges = len(top_edge_idx)
lines = np.empty((num_edges, 3), dtype=np.int64)
lines[:, 0] = 2
lines[:, 1] = src_indices
lines[:, 2] = dst_indices

edge_mesh = pv.PolyData(coords, lines.ravel())
edge_mesh["active"] = np.zeros(num_edges)

# 3. Setup Region Base Colors (RGB)
rgb_colors = np.full((num_neurons, 3), [0.1, 0.12, 0.17], dtype=np.float32)

olf_idx = cell_groups.get("olfactory", torch.tensor([])).numpy()
mb_idx = cell_groups.get("mushroom_body", torch.tensor([])).numpy()
motor_idx = cell_groups.get("motor", torch.tensor([])).numpy()

if len(olf_idx) > 0:
    rgb_colors[olf_idx] = [0.0, 0.8, 1.0]     # Cyan
if len(mb_idx) > 0:
    rgb_colors[mb_idx] = [0.1, 0.9, 0.3]      # Green
if len(motor_idx) > 0:
    rgb_colors[motor_idx] = [0.9, 0.2, 0.8]   # Magenta

point_cloud = pv.PolyData(coords)
point_cloud["colors"] = rgb_colors.copy()

# 4. Simulation State Variables
decay = 0.90
v_threshold = 1.0
v_reset = 0.0
synapse_weight = 0.04

v_membrane = torch.zeros(num_neurons, device=device)
spikes = torch.zeros(num_neurons, device=device)

external_input = torch.zeros(num_neurons, device=device)
if len(olf_idx) > 0:
    external_input[cell_groups["olfactory"].to(device)] = 2.0

# 5. Set up 2D Matplotlib Live Analytics Dashboard
plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 8), facecolor='#0a0a12')
plt.subplots_adjust(hspace=0.45, left=0.18, right=0.92, top=0.92, bottom=0.08)

for ax in (ax1, ax2, ax3):
    ax.set_facecolor('#12121c')
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#333344')

# Analytics Metric Arrays
history_steps = []
history_olf = []
history_mb = []
history_motor = []
history_v_mean = []
history_syn_throughput = []

# Ax1: Population Firing Rates
l_olf, = ax1.plot([], [], label="Olfactory", color="#00ccff", lw=1.8)
l_mb, = ax1.plot([], [], label="Mushroom Body", color="#1aff53", lw=1.8)
l_motor, = ax1.plot([], [], label="Motor", color="#e633cc", lw=1.8)
ax1.set_title("Population Firing Rates (Spikes / Step)", color="white", fontsize=10, pad=6)
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 1500)
ax1.legend(loc="upper right", facecolor="#12121c", edgecolor="#333344", labelcolor="white", fontsize=7)
ax1.grid(True, color="#222233", linestyle="--", alpha=0.5)

# Ax2: Circuit Membrane Voltage Trajectory
l_vmean, = ax2.plot([], [], color="#ffaa00", lw=1.8)
ax2.set_title("Mean Circuit Membrane Potential (mV)", color="white", fontsize=10, pad=6)
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 1.2)
ax2.grid(True, color="#222233", linestyle="--", alpha=0.5)

# Ax3: Active Synaptic Transmission Volume
l_syn, = ax3.plot([], [], color="#ff3366", lw=1.8)
ax3.set_title("Active Synaptic Signal Throughput", color="white", fontsize=10, pad=6)
ax3.set_xlim(0, 100)
ax3.set_ylim(0, 2000)
ax3.set_xlabel("Simulation Step", color="white", fontsize=8)
ax3.grid(True, color="#222233", linestyle="--", alpha=0.5)

fig.canvas.draw()
fig.canvas.flush_events()

# 6. Setup 3D PyVista Plotter
plotter = pv.Plotter(title="Drosophila Brain - Live Connectome Analytics Simulator")
plotter.set_background("#05050A")

plotter.add_mesh(
    point_cloud,
    scalars="colors",
    rgb=True,
    point_size=3.0,
    render_points_as_spheres=True
)

plotter.add_mesh(
    edge_mesh,
    scalars="active",
    cmap=["#05050A", "#FFD700"],
    line_width=1.5,
    opacity=0.6,
    show_scalar_bar=False,
    clim=[0, 1]
)

plotter.add_text(
    "3D Drosophila Neural Dynamics\nCyan: Olfactory | Green: Mushroom Body | Magenta: Motor",
    position="upper_left",
    font_size=10
)

centroid = coords.mean(axis=0)
plotter.set_focus(centroid)
radius = 450.0

plotter.show(interactive_update=True)

# 7. Live Simulation & Analytics Execution Loop
print("\nExecuting live simulation and real-time analytics streaming...")

for step in range(1, 101):
    synaptic_current = torch.sparse.mm(adj_matrix.t(), spikes.unsqueeze(1)).squeeze(1) * synapse_weight
    current_input = external_input if step == 1 else 0.0
    v_membrane = (v_membrane * decay) + synaptic_current + current_input

    spikes = (v_membrane >= v_threshold).float()
    v_membrane = torch.where(spikes > 0, torch.tensor(v_reset, device=device), v_membrane)

    spikes_np = spikes.cpu().numpy()
    active_mask = spikes_np.astype(bool)

    # Visual updates
    frame_colors = rgb_colors.copy()
    frame_colors[active_mask] = [1.0, 0.95, 0.4]
    point_cloud["colors"] = frame_colors

    src_firing = active_mask[src_indices]
    edge_mesh["active"] = src_firing.astype(float)

    # Smooth Orbit Camera
    angle = np.radians(step * 3.6)
    cam_x = centroid[0] + radius * np.sin(angle)
    cam_z = centroid[2] + radius * np.cos(angle)
    plotter.camera_position = [(cam_x, centroid[1] + 50, cam_z), centroid, (0, 1, 0)]
    plotter.render()

    # Collect Analytics Metrics
    olf_spikes = int(spikes_np[olf_idx].sum()) if len(olf_idx) > 0 else 0
    mb_spikes = int(spikes_np[mb_idx].sum()) if len(mb_idx) > 0 else 0
    motor_spikes = int(spikes_np[motor_idx].sum()) if len(motor_idx) > 0 else 0
    v_mean_val = float(v_membrane.mean().item())
    syn_throughput_val = int(src_firing.sum())

    history_steps.append(step)
    history_olf.append(olf_spikes)
    history_mb.append(mb_spikes)
    history_motor.append(motor_spikes)
    history_v_mean.append(v_mean_val)
    history_syn_throughput.append(syn_throughput_val)

    # Update Analytics Plots
    l_olf.set_data(history_steps, history_olf)
    l_mb.set_data(history_steps, history_mb)
    l_motor.set_data(history_steps, history_motor)
    l_vmean.set_data(history_steps, history_v_mean)
    l_syn.set_data(history_steps, history_syn_throughput)

    # Adjust dynamic plot limits if needed
    max_spikes = max(max(history_olf), max(history_mb + [1]), max(history_motor + [1]))
    ax1.set_ylim(0, max(max_spikes * 1.2, 50))
    ax3.set_ylim(0, max(max(history_syn_throughput + [1]) * 1.2, 50))

    fig.canvas.draw()
    fig.canvas.flush_events()

    print(f"Step {step:3d}/100 | Olfactory: {olf_spikes:4d} | MB: {mb_spikes:4d} | Motor: {motor_spikes:4d} | V_mean: {v_mean_val:.3f}")
    time.sleep(0.02)

plt.ioff()
print("\nSimulation & Analytics complete! Close figures to terminate.")
plotter.show(interactive=True)
