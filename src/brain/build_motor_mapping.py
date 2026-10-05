"""Build a verified motor-neuron mapping from FlyWire annotations.

The current annotation snapshot identifies descending neurons and their side,
but does not identify front/middle/hind leg targets. The output therefore
contains side-specific descending populations and leaves leg identity explicit.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch


ANNOTATIONS = Path("data/flywire_annotations/supplemental_files/Supplemental_file1_neuron_annotations.tsv")
ROOT_IDS = Path("data/proofread_root_ids_783.npy")
OUTPUT = Path("data/motor_mapping.pt")


def main() -> None:
    neuron_ids = np.load(ROOT_IDS)
    id_to_index = {int(root_id): index for index, root_id in enumerate(neuron_ids)}
    annotations = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)

    descending = annotations[annotations["super_class"].eq("descending")].copy()
    descending["index"] = descending["root_id"].map(id_to_index)
    descending = descending.dropna(subset=["index"])
    descending["index"] = descending["index"].astype(np.int64)

    mapping = {
        "descending_all": torch.tensor(
            sorted(descending["index"].unique()), dtype=torch.long
        ),
        "descending_left": torch.tensor(
            sorted(
                descending.loc[descending["side"].eq("left"), "index"].unique()
            ),
            dtype=torch.long,
        ),
        "descending_right": torch.tensor(
            sorted(
                descending.loc[descending["side"].eq("right"), "index"].unique()
            ),
            dtype=torch.long,
        ),
        "leg_targets_verified": False,
    }
    torch.save(mapping, OUTPUT)

    print(f"Saved {OUTPUT}")
    print(f"Descending neurons: {len(mapping['descending_all']):,}")
    print(f"Left descending neurons: {len(mapping['descending_left']):,}")
    print(f"Right descending neurons: {len(mapping['descending_right']):,}")
    print("Front/middle/hind leg targets: not present in this annotation snapshot")


if __name__ == "__main__":
    main()
