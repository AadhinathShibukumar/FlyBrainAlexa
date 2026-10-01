import time
import numpy as np
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== Building & Saving FlyWire Sparse Matrix on {device} ===")

start_time = time.time()

# 1. Load Proofread Neuron IDs and create index mapping
neuron_ids = np.load("data/proofread_root_ids_783.npy")
num_neurons = len(neuron_ids)
id_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}
print(f"Indexed {num_neurons:,} neurons.")

# 2. Load Synapse Connections
print("Reading raw synapse connections...")
df = pd.read_feather(
    "data/flywire_synapses_783.feather",
    columns=["pre_pt_root_id", "post_pt_root_id"]
)

# 3. Map IDs to matrix coordinates
print("Mapping 64-bit IDs to matrix indices...")
df["pre_idx"] = df["pre_pt_root_id"].map(id_to_idx)
df["post_idx"] = df["post_pt_root_id"].map(id_to_idx)

# Drop non-proofread references
df = df.dropna(subset=["pre_idx", "post_idx"])
df["pre_idx"] = df["pre_idx"].astype(np.int64)
df["post_idx"] = df["post_idx"].astype(np.int64)

# 4. Aggregate multiple synapses into weights
print("Aggregating synaptic weights...")
edges = df.groupby(["pre_idx", "post_idx"]).size().reset_index(name="weight")

# 5. Build PyTorch Sparse Matrix efficiently (Warning-Free)
print("Constructing PyTorch Sparse Tensor...")
indices_np = np.stack([edges["pre_idx"].values, edges["post_idx"].values])
indices = torch.from_numpy(indices_np).to(device)
values = torch.from_numpy(edges["weight"].values).float().to(device)

adj_matrix = torch.sparse_coo_tensor(
    indices, 
    values, 
    size=(num_neurons, num_neurons), 
    device=device,
    check_invariants=False
).coalesce()

# 6. Save Matrix to Disk for Instant Loading
save_path = "data/fly_adjacency_matrix.pt"
torch.save(adj_matrix, save_path)

elapsed = time.time() - start_time
print(f"\nSuccess! Brain Adjacency Matrix saved to '{save_path}' in {elapsed:.2f}s")
print(f"Matrix Dimensions: {adj_matrix.shape[0]:,} x {adj_matrix.shape[1]:,}")
print(f"Unique Connectome Edges: {adj_matrix._nnz():,}")
print(f"VRAM Allocated: {torch.cuda.memory_allocated(0) / (1024 ** 2):.2f} MB")