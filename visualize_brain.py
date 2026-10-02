import numpy as np
import torch
import pyvista as pv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== Launching Async 3D Brain Simulator on {device} ===")

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

# 3. Setup Vivid Region Palette (High Contrast)
dim_rgb = np.full((num_neurons, 3), [0.05, 0.08, 0.15], dtype=np.float32)  # Deep Navy Blue
bright_rgb = np.full((num_neurons, 3), [0.4, 0.4, 0.5], dtype=np.float32)

olf_idx = cell_groups.get("olfactory", torch.tensor([])).numpy()
mb_idx = cell_groups.get("mushroom_body", torch.tensor([])).numpy()
motor_idx = cell_groups.get("motor", torch.tensor([])).numpy()
visual_idx = cell_groups.get("visual", torch.tensor([])).numpy()

if len(visual_idx) == 0:
    visual_idx = np.arange(0, min(5000, num_neurons))

# Distinct High-Contrast Colors
if len(olf_idx) > 0:
    dim_rgb[olf_idx] = [0.0, 0.5, 0.9]       # Cyan/Blue
    bright_rgb[olf_idx] = [0.0, 1.0, 1.0]

if len(visual_idx) > 0:
    dim_rgb[visual_idx] = [0.9, 0.4, 0.0]    # Vibrant Orange
    bright_rgb[visual_idx] = [1.0, 0.7, 0.0]

if len(mb_idx) > 0:
    dim_rgb[mb_idx] = [0.0, 0.8, 0.2]        # Vibrant Green
    bright_rgb[mb_idx] = [0.3, 1.0, 0.4]

if len(motor_idx) > 0:
    dim_rgb[motor_idx] = [0.9, 0.0, 0.6]     # Vivid Magenta
    bright_rgb[motor_idx] = [1.0, 0.3, 0.8]

point_cloud = pv.PolyData(coords)
point_cloud["colors"] = dim_rgb.copy()

# 4. LIF Parameters
decay = 0.70
v_threshold = 1.0
v_reset = 0.0
synapse_weight = 0.003

v_membrane = torch.zeros(num_neurons, device=device)
spikes = torch.zeros(num_neurons, device=device)
step = 0
current_stimulus = None

# 5. Setup PyVista Viewer
plotter = pv.Plotter(title="Drosophila Brain - Non-Blocking Realtime Visualizer")
plotter.set_background("#020205")

plotter.add_mesh(
    point_cloud,
    scalars="colors",
    rgb=True,
    point_size=3.5,
    render_points_as_spheres=True
)

plotter.add_mesh(
    edge_mesh,
    scalars="active",
    cmap=["#020205", "#FFD700"],
    line_width=1.5,
    opacity=0.6,
    show_scalar_bar=False,
    clim=[0, 1]
)

plotter.add_text(
    "Cyan: Olfactory | Orange: Visual | Green: Mushroom Body | Magenta: Motor\nPress Key: [1] Olfactory | [2] Visual | [3] Motor | [0] Clear Pulse",
    position="upper_left",
    font_size=9
)

centroid = coords.mean(axis=0)
plotter.set_focus(centroid)
radius = 420.0

# Keypress Trigger Callbacks directly inside PyVista window
def trigger_olf():
    global current_stimulus; current_stimulus = 'olfactory'; print("--> [PULSE] Olfactory Circuit")
def trigger_vis():
    global current_stimulus; current_stimulus = 'visual'; print("--> [PULSE] Visual Circuit")
def trigger_motor():
    global current_stimulus; current_stimulus = 'motor'; print("--> [PULSE] Motor Circuit")
def trigger_clear():
    global v_membrane, spikes, current_stimulus
    v_membrane.zero_(); spikes.zero_(); current_stimulus = None
    print("--> [RESET] Cleared all signals")

plotter.add_key_event("1", trigger_olf)
plotter.add_key_event("2", trigger_vis)
plotter.add_key_event("3", trigger_motor)
plotter.add_key_event("0", trigger_clear)

# Non-blocking async step loop callback
def timer_callback(step_id):
    global step, v_membrane, spikes, current_stimulus
    step += 1

    external_input = torch.zeros(num_neurons, device=device)
    if current_stimulus == 'olfactory' and len(olf_idx) > 0:
        external_input[cell_groups["olfactory"].to(device)] = 1.8
    elif current_stimulus == 'visual' and len(visual_idx) > 0:
        external_input[torch.tensor(visual_idx, device=device)] = 1.8
    elif current_stimulus == 'motor' and len(motor_idx) > 0:
        external_input[cell_groups["motor"].to(device)] = 1.8

    current_stimulus = None  # single pulse per trigger

    synaptic_current = torch.sparse.mm(adj_matrix.t(), spikes.unsqueeze(1)).squeeze(1) * synapse_weight
    v_membrane = (v_membrane * decay) + synaptic_current + external_input

    spikes = (v_membrane >= v_threshold).float()
    v_membrane = torch.where(spikes > 0, torch.tensor(v_reset, device=device), v_membrane)

    spikes_np = spikes.cpu().numpy()
    active_mask = spikes_np.astype(bool)

    frame_colors = np.where(active_mask[:, None], bright_rgb, dim_rgb)
    point_cloud["colors"] = frame_colors

    src_firing = active_mask[src_indices]
    edge_mesh["active"] = src_firing.astype(float)

    # Smooth Camera Orbit
    angle = np.radians(step * 0.8)
    cam_x = centroid[0] + radius * np.sin(angle)
    cam_z = centroid[2] + radius * np.cos(angle)
    plotter.camera_position = [(cam_x, centroid[1] + 20, cam_z), centroid, (0, 1, 0)]

# Add 20ms timer for continuous smooth rendering
plotter.add_timer_event(max_steps=1000, duration=20, callback=timer_callback)

print("\nWindow loaded! Press keys 1, 2, 3, or 0 DIRECTLY on the 3D PyVista window.")
plotter.show()
