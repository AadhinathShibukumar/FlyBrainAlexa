import os
import numpy as np
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== FlyBrain Core Initializing on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) ===")

# 1. Load Proofread Root IDs
ids_path = "data/proofread_root_ids_783.npy"
if os.path.exists(ids_path):
    neuron_ids = np.load(ids_path)
    print(f"Loaded {len(neuron_ids):,} proofread FlyWire neurons.")

# 2. Load Synapse Summaries
post_path = "data/per_neuron_neuropil_count_post_783.feather"
if os.path.exists(post_path):
    df_post = pd.read_feather(post_path)
    print(f"Loaded {len(df_post):,} postsynaptic neuropil records.")

# 3. Load Raw Synapse Dataset (Using verified column names)
synapse_path = "data/flywire_synapses_783.feather"
if os.path.exists(synapse_path):
    print("\nLoading 130M+ raw synapse connections...")
    df_synapses = pd.read_feather(
        synapse_path, 
        columns=["pre_pt_root_id", "post_pt_root_id"]
    )
    print(f"Loaded {len(df_synapses):,} raw synaptic links!")
    print("\nSample Synapse Data:\n", df_synapses.head())

# 4. Allocate GPU Tensor
if torch.cuda.is_available():
    id_tensor = torch.tensor(neuron_ids, device=device, dtype=torch.int64)
    print(f"\nAllocated neuron ID tensor on GPU: {id_tensor.shape}")