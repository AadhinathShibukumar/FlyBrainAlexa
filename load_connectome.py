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
else:
    print(f"Error: Could not find {ids_path}")

# 2. Load Synapse Summaries
post_path = "data/per_neuron_neuropil_count_post_783.feather"
if os.path.exists(post_path):
    df_post = pd.read_feather(post_path)
    print(f"Loaded {len(df_post):,} postsynaptic neuropil records.")
else:
    print(f"Error: Could not find {post_path}")

# 3. Convert array to PyTorch Tensor on GPU
if torch.cuda.is_available():
    id_tensor = torch.tensor(neuron_ids, device=device, dtype=torch.int64)
    print(f"Successfully allocated neuron ID tensor on GPU: {id_tensor.shape} ({id_tensor.dtype})")