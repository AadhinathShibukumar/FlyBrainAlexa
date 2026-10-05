"""Build descending and efferent motor groups from FlyWire annotations."""

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
    motor = annotations[
        annotations["super_class"].eq("motor")
        & annotations["flow"].eq("efferent")
    ].copy()
    motor["index"] = motor["root_id"].map(id_to_index)
    motor = motor.dropna(subset=["index"])
    motor["index"] = motor["index"].astype(np.int64)

    def indices(frame: pd.DataFrame) -> torch.Tensor:
        return torch.tensor(sorted(frame["index"].unique()), dtype=torch.long)

    mapping = {
        "descending_all": indices(descending),
        "descending_left": indices(descending.loc[descending["side"].eq("left")]),
        "descending_right": indices(descending.loc[descending["side"].eq("right")]),
        "motor_all": indices(motor),
        "motor_left": indices(motor.loc[motor["side"].eq("left")]),
        "motor_right": indices(motor.loc[motor["side"].eq("right")]),
        "motor_brain": indices(
            motor.loc[motor["cell_class"].eq("brain_motor_neuron")]
        ),
        "motor_pharyngeal": indices(motor.loc[motor["nerve"].eq("PhN")]),
        "motor_antennal": indices(motor.loc[motor["nerve"].eq("AN")]),
        "motor_ocellar": indices(motor.loc[motor["nerve"].eq("ON")]),
        "motor_maxillary_labial": indices(motor.loc[motor["nerve"].eq("MxLbN")]),
        "motor_cervical": indices(motor.loc[motor["nerve"].eq("CV")]),
        "leg_targets_verified": False,
    }
    torch.save(mapping, OUTPUT)

    print(f"Saved {OUTPUT}")
    print(f"Descending neurons: {len(mapping['descending_all']):,}")
    print(f"Left descending neurons: {len(mapping['descending_left']):,}")
    print(f"Right descending neurons: {len(mapping['descending_right']):,}")
    print(f"Efferent motor neurons: {len(mapping['motor_all']):,}")
    print("Front/middle/hind leg targets: not present in this annotation snapshot")


if __name__ == "__main__":
    main()
