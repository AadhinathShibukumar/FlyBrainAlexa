import time
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== FlyBrain Spiking Neural Network on {device} ===")

# 1. Load Pre-built Matrix (~0.5s load time)
print("Loading saved adjacency matrix...")
adj_matrix = torch.load("data/fly_adjacency_matrix.pt", map_location=device)
num_neurons = adj_matrix.shape[0]

# 2. Simulation Parameters
decay = 0.95              # Membrane decay coefficient
v_threshold = 1.0         # Spiking threshold
v_reset = 0.0             # Reset potential after spike
synapse_weight = 0.05     # Synaptic strength multiplier

# State Variables on GPU
v_membrane = torch.zeros(num_neurons, device=device)
spikes = torch.zeros(num_neurons, device=device)

# 3. Inject Initial Stimulus (Stimulate first 500 sensory neurons)
print("Injecting initial current into 500 sensory neurons...")
external_input = torch.zeros(num_neurons, device=device)
external_input[:500] = 1.5

# 4. Run Simulation Steps
time_steps = 50
print(f"\nRunning {time_steps} timestep GPU simulation...")
start_time = time.time()

total_spikes = 0
for step in range(1, time_steps + 1):
    # Synaptic current received from connected neurons that spiked last step
    synaptic_current = torch.sparse.mm(adj_matrix.t(), spikes.unsqueeze(1)).squeeze(1) * synapse_weight
    
    # Update membrane potential: Decay + Synaptic Input + External Input (only on step 1)
    current_input = external_input if step == 1 else 0.0
    v_membrane = (v_membrane * decay) + synaptic_current + current_input
    
    # Determine which neurons spike this turn
    spikes = (v_membrane >= v_threshold).float()
    num_spikes_this_step = int(spikes.sum().item())
    total_spikes += num_spikes_this_step
    
    # Reset potential of neurons that spiked
    v_membrane = torch.where(spikes > 0, torch.tensor(v_reset, device=device), v_membrane)
    
    if step % 10 == 0 or num_spikes_this_step > 0:
        print(f"  Step {step:2d}/{time_steps} | Active Spikes: {num_spikes_this_step:,}")

elapsed = time.time() - start_time
print(f"\nSimulation Complete in {elapsed * 1000:.2f} ms!")
print(f"Total Spikes Triggered Across Fly Brain: {total_spikes:,}")
print(f"VRAM Usage: {torch.cuda.memory_allocated(0) / (1024 ** 2):.2f} MB")