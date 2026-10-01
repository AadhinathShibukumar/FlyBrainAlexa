import time
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== Targeted Drosophila SNN Simulation on {device} ===")

# 1. Load Matrix & Cell Groups
adj_matrix = torch.load("data/fly_adjacency_matrix.pt", map_location=device)
cell_groups = torch.load("data/cell_groups.pt")
num_neurons = adj_matrix.shape[0]

# 2. Simulation Physics Hyperparameters
decay = 0.95
v_threshold = 1.0
v_reset = 0.0
synapse_weight = 0.08

v_membrane = torch.zeros(num_neurons, device=device)
spikes = torch.zeros(num_neurons, device=device)

# 3. Stimulate Olfactory (Smell) Circuit
external_input = torch.zeros(num_neurons, device=device)
olfactory_idx = cell_groups["olfactory"].to(device)
mb_idx = cell_groups["mushroom_body"].to(device)
motor_idx = cell_groups["descending_motor"].to(device)

external_input[olfactory_idx] = 2.0  # Drive sensory neurons past firing threshold
print(f"Injecting sensory signal into {len(olfactory_idx)} Olfactory neurons...\n")

start_time = time.time()
total_spikes = 0

# 4. 50-step (50ms) Simulation Loop
for step in range(1, 51):
    synaptic_current = torch.sparse.mm(adj_matrix.t(), spikes.unsqueeze(1)).squeeze(1) * synapse_weight
    current_input = external_input if step == 1 else 0.0
    v_membrane = (v_membrane * decay) + synaptic_current + current_input

    spikes = (v_membrane >= v_threshold).float()
    v_membrane = torch.where(spikes > 0, torch.tensor(v_reset, device=device), v_membrane)

    step_spikes = int(spikes.sum().item())
    total_spikes += step_spikes

    mb_spikes = int(spikes[mb_idx].sum().item())
    motor_spikes = int(spikes[motor_idx].sum().item())

    print(f"Step {step:2d}/50 | Total Spikes: {step_spikes:5d} | Mushroom Body: {mb_spikes:2d} | Motor Output: {motor_spikes:2d}")

elapsed = time.time() - start_time
print(f"\nSimulation Complete: {total_spikes:,} total spikes processed in {elapsed:.3f}s on GPU.")
