from pathlib import Path

import numpy as np
import torch

from src.brain.leg_drive_mapper import LegDriveMapper


class ConnectomeBrain:
    """Small stateful spiking interface for the FlyWire connectome."""

    def __init__(
        self,
        matrix_path: str | Path = "data/fly_adjacency_matrix.pt",
        groups_path: str | Path = "data/cell_groups.pt",
        motor_mapping_path: str | Path = "data/motor_mapping.pt",
        synapse_weight: float = 0.08,
        decay: float = 0.95,
        threshold: float = 1.0,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adjacency = torch.load(matrix_path, map_location=self.device)
        groups = torch.load(groups_path, map_location=self.device)
        self.groups = {name: indices.long() for name, indices in groups.items()}
        mapping = torch.load(motor_mapping_path, map_location=self.device)
        self.motor_groups = {
            name: indices.long()
            for name, indices in mapping.items()
            if name.startswith(("descending_", "motor_"))
        }
        self.decay = decay
        self.threshold = threshold
        self.synapse_weight = synapse_weight
        self.membrane = torch.zeros(
            self.adjacency.shape[0], device=self.device
        )
        self.spikes = torch.zeros_like(self.membrane)
        self.leg_mapper = LegDriveMapper()

    def reset(self) -> None:
        self.membrane.zero_()
        self.spikes.zero_()

    @torch.no_grad()
    def step(
        self,
        olfactory_drive: float = 0.0,
        mechanosensory_drive: float = 0.0,
        proprioceptive_drive: float = 0.0,
        contact_force_drive: float = 0.0,
    ) -> dict[str, float]:
        """Advance one neural timestep and return population motor activity."""
        sensory_input = torch.zeros_like(self.membrane)
        if olfactory_drive:
            sensory_input[self.groups["olfactory"]] = olfactory_drive
        if mechanosensory_drive:
            sensory_input[self.groups["mechanosensory"]] = mechanosensory_drive
        if contact_force_drive:
            sensory_input[self.groups["mechanosensory"]] += float(
                np.clip(contact_force_drive, 0.0, 1.0)
            )
        if proprioceptive_drive:
            sensory_input[self.groups["mechanosensory"]] += float(
                np.clip(proprioceptive_drive, 0.0, 1.0)
            )

        synaptic_current = (
            torch.sparse.mm(
                self.adjacency.transpose(0, 1),
                self.spikes.unsqueeze(1),
            ).squeeze(1)
            * self.synapse_weight
        )
        self.membrane.mul_(self.decay).add_(synaptic_current).add_(sensory_input)
        self.spikes = (self.membrane >= self.threshold).to(self.membrane.dtype)
        self.membrane.masked_fill_(self.spikes.bool(), 0.0)

        descending_spikes = self.spikes[self.motor_groups["descending_all"]].sum()
        motor_count = max(len(self.motor_groups["descending_all"]), 1)
        descending_drive = float((descending_spikes / motor_count).item())
        left_count = max(len(self.motor_groups["descending_left"]), 1)
        right_count = max(len(self.motor_groups["descending_right"]), 1)
        left_drive = float(
            (self.spikes[self.motor_groups["descending_left"]].sum() / left_count).item()
        )
        right_drive = float(
            (self.spikes[self.motor_groups["descending_right"]].sum() / right_count).item()
        )
        motor_neuron_count = max(len(self.motor_groups["motor_all"]), 1)
        motor_neuron_spikes = self.spikes[self.motor_groups["motor_all"]].sum()
        motor_neuron_drive = float((motor_neuron_spikes / motor_neuron_count).item())
        motor_left_count = max(len(self.motor_groups["motor_left"]), 1)
        motor_right_count = max(len(self.motor_groups["motor_right"]), 1)
        motor_left_drive = float(
            (self.spikes[self.motor_groups["motor_left"]].sum() / motor_left_count).item()
        )
        motor_right_drive = float(
            (self.spikes[self.motor_groups["motor_right"]].sum() / motor_right_count).item()
        )
        motor_drive = max(descending_drive, motor_neuron_drive)
        leg_drives = self.leg_mapper.step(left_drive, right_drive)
        return {
            "motor_spikes": float(descending_spikes.item()),
            "motor_drive": motor_drive,
            "left_drive": left_drive,
            "right_drive": right_drive,
            "motor_neuron_spikes": float(motor_neuron_spikes.item()),
            "motor_neuron_drive": motor_neuron_drive,
            "motor_neuron_left_drive": motor_left_drive,
            "motor_neuron_right_drive": motor_right_drive,
            "leg_drives": leg_drives,
        }
