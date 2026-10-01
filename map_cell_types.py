import numpy as np
import pandas as pd
import torch

print("=== Categorizing FlyWire Neurons by Functional Cell Type ===")

# 1. Load proofread IDs and build O(1) index map
neuron_ids = np.load("data/proofread_root_ids_783.npy")
id_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}

# 2. Load annotations dataframe
ann_path = "data/flywire_annotations_783.feather"
df_ann = pd.read_feather(ann_path)

print(f"Loaded {len(df_ann):,} annotation entries.")

# 3. Initialize functional categories
groups = {
    "olfactory": [],
    "visual": [],
    "mechanosensory": [],
    "mushroom_body": [],
    "descending_motor": []
}

# 4. Filter and assign indices based on FlyWire metadata
for _, row in df_ann.iterrows():
    nid = row.get("pt_root_id", row.get("root_id"))
    if nid not in id_to_idx:
        continue
    
    idx = id_to_idx[nid]
    super_class = str(row.get("super_class", "")).lower()
    cell_type = str(row.get("cell_type", "")).lower()
    group_label = str(row.get("group", "")).lower()

    if "olfactory" in super_class or "antennae" in group_label or "orn" in cell_type or "pn" in cell_type:
        groups["olfactory"].append(idx)
    elif "visual" in super_class or "optic" in group_label or "retina" in cell_type:
        groups["visual"].append(idx)
    elif "mechanosensory" in super_class or "auditory" in group_label:
        groups["mechanosensory"].append(idx)
    elif "mbon" in cell_type or "kenyon" in cell_type or "kc" in cell_type:
        groups["mushroom_body"].append(idx)
    elif "descending" in super_class or "dn" in cell_type:
        groups["descending_motor"].append(idx)

# 5. Convert lists to unique PyTorch Tensors
group_indices = {}
print("\n--- Functional Groups Mapped ---")
for name, idx_list in groups.items():
    unique_indices = torch.tensor(list(set(idx_list)), dtype=torch.long)
    group_indices[name] = unique_indices
    print(f"  • {name.title()}: {len(unique_indices):,} mapped neurons")

# 6. Save dictionary mapping to disk
save_path = "data/cell_groups.pt"
torch.save(group_indices, save_path)
print(f"\nSaved functional mapping to '{save_path}'!")
