"""Build synapse-weighted motor mapping from FlyWire synapse table.

This script:
1. Reads the 8.9 GB FlyWire synapse table (feather format).
2. Filters for descending→motor connections.
3. Aggregates synapses by (pre_root_id, post_root_id) to build weighted edges.
4. Saves a connectivity-weighted motor mapping suitable for integration into ConnectomeBrain.

The synapse table has columns: pre_root_id, post_root_id, size (synapse count).
Output is a PyTorch dict mapping descending neuron indices to weighted motor drives.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import torch


SYNAPSE_TABLE = Path("data/flywire_synapses_783.feather")
ANNOTATIONS = Path("data/flywire_annotations/supplemental_files/Supplemental_file1_neuron_annotations.tsv")
ROOT_IDS = Path("data/proofread_root_ids_783.npy")
OUTPUT = Path("data/synapse_weighted_motor_mapping.pt")
CHUNK_SIZE = 50_000  # Number of rows to read per chunk


def main() -> None:
    """Main entry point."""
    print("Loading neuron annotations...")
    neuron_ids = np.load(ROOT_IDS)
    id_to_index = {int(root_id): index for index, root_id in enumerate(neuron_ids)}
    annotations = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)

    # Extract descending and motor neuron root IDs.
    descending_neurons = annotations[
        annotations["super_class"].eq("descending")
    ]["root_id"].unique()
    motor_neurons = annotations[
        (annotations["super_class"].eq("motor"))
        & (annotations["flow"].eq("efferent"))
    ]["root_id"].unique()

    descending_set = set(int(rid) for rid in descending_neurons)
    motor_set = set(int(rid) for rid in motor_neurons)

    print(f"Descending neurons: {len(descending_set):,}")
    print(f"Motor neurons: {len(motor_set):,}")

    # Read synapse table in chunks and aggregate connections.
    print(f"Reading synapse table from {SYNAPSE_TABLE}...")
    print("(This may take 1-2 minutes for the full 8.9 GB file)")

    weighted_connections: Dict[int, Dict[int, int]] = {}
    rows_processed = 0

    try:
        # Attempt to read with pyarrow. If it's too slow, this will time out.
        table = feather.read_table(SYNAPSE_TABLE)
        df = table.to_pandas()

        # Filter for descending→motor connections.
        descending_mask = df["pre_root_id"].isin(descending_set)
        motor_mask = df["post_root_id"].isin(motor_set)
        relevant_synapses = df[descending_mask & motor_mask]

        print(f"Found {len(relevant_synapses):,} descending→motor synapses")

        # Aggregate synapses by (pre, post) pair.
        for _, row in relevant_synapses.iterrows():
            pre_id = int(row["pre_root_id"])
            post_id = int(row["post_root_id"])
            size = int(row.get("size", 1))

            if pre_id not in weighted_connections:
                weighted_connections[pre_id] = {}
            weighted_connections[pre_id][post_id] = (
                weighted_connections[pre_id].get(post_id, 0) + size
            )

    except Exception as e:
        print(f"Warning: Failed to read full synapse table: {e}")
        print("Attempting chunked read (may be slow)...")
        weighted_connections = _read_synapses_chunked(
            descending_set, motor_set, CHUNK_SIZE
        )

    print(f"Aggregated {len(weighted_connections):,} descending neurons with motor targets")

    # Build index-based mapping.
    print("Building index-based mapping...")
    mapping = _build_index_mapping(
        weighted_connections, id_to_index, annotations
    )

    torch.save(mapping, OUTPUT)
    print(f"Saved {OUTPUT}")
    print(f"Mapping keys: {list(mapping.keys())}")


def _read_synapses_chunked(
    descending_set: set,
    motor_set: set,
    chunk_size: int,
) -> Dict[int, Dict[int, int]]:
    """Read synapse table in chunks to avoid memory overload."""
    weighted_connections: Dict[int, Dict[int, int]] = {}
    rows_processed = 0

    # Use pyarrow to stream chunks.
    reader = feather.read_table(SYNAPSE_TABLE)

    # Process in batches.
    for i in range(0, reader.num_rows, chunk_size):
        batch = reader.slice(i, min(chunk_size, reader.num_rows - i))
        df_batch = batch.to_pandas()

        # Filter for descending→motor.
        descending_mask = df_batch["pre_root_id"].isin(descending_set)
        motor_mask = df_batch["post_root_id"].isin(motor_set)
        relevant = df_batch[descending_mask & motor_mask]

        # Aggregate.
        for _, row in relevant.iterrows():
            pre_id = int(row["pre_root_id"])
            post_id = int(row["post_root_id"])
            size = int(row.get("size", 1))

            if pre_id not in weighted_connections:
                weighted_connections[pre_id] = {}
            weighted_connections[pre_id][post_id] = (
                weighted_connections[pre_id].get(post_id, 0) + size
            )

        rows_processed += len(df_batch)
        print(f"  Processed {rows_processed:,} rows...")

    return weighted_connections


def _build_index_mapping(
    weighted_connections: Dict[int, Dict[int, int]],
    id_to_index: Dict[int, int],
    annotations: pd.DataFrame,
) -> dict:
    """Convert root-ID-based connectivity to index-based tensors."""
    mapping = {}

    # Convert each descending neuron's motor targets to indices with weights.
    motor_neuron_weights = {}  # (desc_idx, motor_idx) -> weight

    for pre_root_id, motor_targets in weighted_connections.items():
        if pre_root_id not in id_to_index:
            continue
        desc_idx = id_to_index[pre_root_id]

        for post_root_id, synapse_count in motor_targets.items():
            if post_root_id not in id_to_index:
                continue
            motor_idx = id_to_index[post_root_id]
            motor_neuron_weights[(desc_idx, motor_idx)] = synapse_count

    print(
        f"Converted {len(motor_neuron_weights):,} weighted descending→motor connections"
    )

    # Reconstruct the base mapping with synaptic weights.
    neuron_ids = np.load("data/proofread_root_ids_783.npy")
    id_to_index = {int(root_id): index for index, root_id in enumerate(neuron_ids)}
    annotations = pd.read_csv(
        "data/flywire_annotations/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
        sep="\t",
        low_memory=False,
    )

    descending = annotations[annotations["super_class"].eq("descending")].copy()
    descending["index"] = descending["root_id"].map(id_to_index)
    descending = descending.dropna(subset=["index"])
    descending["index"] = descending["index"].astype(np.int64)

    motor = annotations[
        (annotations["super_class"].eq("motor"))
        & (annotations["flow"].eq("efferent"))
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
        "motor_maxillary_labial": indices(
            motor.loc[motor["nerve"].eq("MxLbN")]
        ),
        "motor_cervical": indices(motor.loc[motor["nerve"].eq("CV")]),
        "synapse_weighted_connections": motor_neuron_weights,
        "leg_targets_verified": False,
    }

    return mapping


if __name__ == "__main__":
    main()
