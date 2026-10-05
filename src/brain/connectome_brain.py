from pathlib import Path

import torch


class ConnectomeBrain:
    """Small stateful spiking interface for the FlyWire connectome."""

    def __init__(
        self,
        matrix_path: str | Path = "data/fly_adjacency_matrix.pt",
        groups_path: str | Path = "data/cell_groups.pt",
        synapse_weight: float = 0.08,
        decay: float = 0.95,
        threshold: float = 1.0,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adjacency = torch.load(matrix_path, map_location=self.device)
        groups = torch.load(groups_path, map_location=self.device)
        self.groups = {name: indices.long() for name, indices in groups.items()}
        self.decay = decay
        self.threshold = threshold
        self.synapse_weight = synapse_weight
        self.membrane = torch.zeros(
            self.adjacency.shape[0], device=self.device
        )
        self.spikes = torch.zeros_like(self.membrane)

    def reset(self) -> None:
        self.membrane.zero_()
        self.spikes.zero_()

    @torch.no_grad()
    def step(self, olfactory_drive: float = 0.0) -> dict[str, float]:
        """Advance one neural timestep and return population motor activity."""
        sensory_input = torch.zeros_like(self.membrane)
        if olfactory_drive:
            sensory_input[self.groups["olfactory"]] = olfactory_drive

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

        motor_spikes = self.spikes[self.groups["descending_motor"]].sum()
        motor_count = max(len(self.groups["descending_motor"]), 1)
        motor_drive = float((motor_spikes / motor_count).item())
        return {
            "motor_spikes": float(motor_spikes.item()),
            "motor_drive": motor_drive,
            "leg_drives": {
                leg: motor_drive
                for leg in ("LF", "LM", "LH", "RF", "RM", "RH")
            },
        }
