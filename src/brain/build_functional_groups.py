"""Build connectome input groups from the published FlyWire annotations."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch


ANNOTATIONS = Path(
    "data/flywire_annotations/supplemental_files/"
    "Supplemental_file1_neuron_annotations.tsv"
)
ROOT_IDS = Path("data/proofread_root_ids_783.npy")
OUTPUT = Path("data/cell_groups.pt")


def build_groups(
    annotations: pd.DataFrame,
    root_ids: np.ndarray,
) -> dict[str, torch.Tensor]:
    required = {"root_id", "super_class", "cell_class", "flow", "side"}
    missing = required.difference(annotations.columns)
    if missing:
        raise ValueError(f"annotation columns missing: {sorted(missing)}")

    root_to_index = {int(root_id): index for index, root_id in enumerate(root_ids)}
    cell_class = annotations["cell_class"].astype("string")
    flow = annotations["flow"].astype("string")

    masks = {
        "olfactory": cell_class.str.contains("olfactory", case=False, na=False),
        "mechanosensory": cell_class.str.contains(
            "mechanosensory", case=False, na=False
        ),
        "visual": (
            annotations["flow"].eq("afferent")
            & cell_class.str.contains(
                "visual|ME|optic|retina", case=False, na=False, regex=True
            )
        ),
        "mushroom_body": cell_class.str.contains(
            "Kenyon_Cell|mushroom", case=False, na=False, regex=True
        ),
        "descending_motor": annotations["super_class"].eq("descending"),
        "motor": annotations["super_class"].eq("motor"),
    }
    groups: dict[str, torch.Tensor] = {}
    for name, mask in masks.items():
        indices = sorted(
            {
                root_to_index[int(root_id)]
                for root_id in annotations.loc[mask, "root_id"]
                if int(root_id) in root_to_index
            }
        )
        if not indices:
            raise ValueError(f"annotation-derived group is empty: {name}")
        groups[name] = torch.tensor(indices, dtype=torch.long)
    return groups


def main() -> None:
    annotations = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    root_ids = np.load(ROOT_IDS)
    groups = build_groups(annotations, root_ids)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(groups, OUTPUT)
    print(f"Saved {OUTPUT}")
    for name, indices in groups.items():
        print(f"{name}: {len(indices):,}")


if __name__ == "__main__":
    main()
