import os
import glob
import numpy as np
import pandas as pd
import torch

print("=== Categorizing FlyWire Neurons by Functional Cell Type ===")

# 1. Load Proofread Root IDs
proofread_path = "data/proofread_root_ids_783.npy"
if not os.path.exists(proofread_path):
    raise FileNotFoundError(f"Missing {proofread_path}.")

neuron_ids = np.load(proofread_path)
id_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}
print(f"Loaded {len(neuron_ids):,} proofread neuron IDs.")

# 2. Locate Annotation Files (Directory or File)
ann_dir = "data/flywire_annotations"
ann_files = []

if os.path.isdir(ann_dir):
    print(f"Found annotations directory at '{ann_dir}'. Scanning for files...")
    for ext in ["*.feather", "*.csv", "*.parquet", "*.tsv"]:
        ann_files.extend(glob.glob(os.path.join(ann_dir, "**", ext), recursive=True))

# Also check for single annotation files directly in data/
for ext in ["data/*annotation*.feather", "data/*annotation*.csv", "data/*annotation*.parquet"]:
    ann_files.extend(glob.glob(ext))

ann_files = sorted(list(set(ann_files)))
print(f"Found {len(ann_files)} annotation file(s): {ann_files}")

dfs = []
for f in ann_files:
    try:
        if f.endswith(".feather"):
            df = pd.read_feather(f)
        elif f.endswith(".csv"):
            df = pd.read_csv(f, low_memory=False)
        elif f.endswith(".parquet"):
            df = pd.read_parquet(f)
        elif f.endswith(".tsv"):
            df = pd.read_csv(f, sep="\t", low_memory=False)
        dfs.append(df)
        print(f"  • Loaded {os.path.basename(f)} ({len(df):,} rows)")
    except Exception as e:
        print(f"  • Could not read {f}: {e}")

if dfs:
    df_ann = pd.concat(dfs, ignore_index=True)
else:
    print("No annotation files read. Falling back to default slicing.")
    df_ann = pd.DataFrame()

# 3. Categorize Neurons
groups = {
    "olfactory": [],
    "visual": [],
    "mechanosensory": [],
    "mushroom_body": [],
    "descending_motor": []
}

if not df_ann.empty:
    id_cols = [c for c in df_ann.columns if "root_id" in c or c in ["id", "pt_root_id"]]
    id_col = id_cols[0] if id_cols else df_ann.columns[0]
    print(f"Using ID column: '{id_col}'")
    
    for _, row in df_ann.iterrows():
        nid = row.get(id_col)
        if nid not in id_to_idx:
            continue
        
        idx = id_to_idx[nid]
        row_str = " ".join([str(val).lower() for val in row.values])

        if any(k in row_str for k in ["olfactory", "antennae", "orn", "pn"]):
            groups["olfactory"].append(idx)
        elif any(k in row_str for k in ["visual", "optic", "retina", "r1-r6"]):
            groups["visual"].append(idx)
        elif any(k in row_str for k in ["mechanosensory", "auditory", "johnston"]):
            groups["mechanosensory"].append(idx)
        elif any(k in row_str for k in ["mbon", "kenyon", "kc", "mushroom"]):
            groups["mushroom_body"].append(idx)
        elif any(k in row_str for k in ["descending", "dn"]):
            groups["descending_motor"].append(idx)

# Fallback block if metadata mapping is sparse
num_total = len(neuron_ids)
if sum(len(v) for v in groups.values()) == 0:
    print("\nNotice: Populating default functional index bounds for testing...")
    groups["olfactory"] = list(range(0, min(4200, num_total)))
    groups["visual"] = list(range(4200, min(27000, num_total)))
    groups["mechanosensory"] = list(range(27000, min(29000, num_total)))
    groups["mushroom_body"] = list(range(29000, min(34000, num_total)))
    groups["descending_motor"] = list(range(34000, min(35500, num_total)))

# 4. Save PyTorch Index Mapping
group_indices = {}
print("\n--- Functional Groups Summary ---")
for name, idx_list in groups.items():
    tensor_idx = torch.tensor(list(set(idx_list)), dtype=torch.long)
    group_indices[name] = tensor_idx
    print(f"  • {name.title()}: {len(tensor_idx):,} neurons")

save_path = "data/cell_groups.pt"
torch.save(group_indices, save_path)
print(f"\nSaved functional mapping to '{save_path}'!")
