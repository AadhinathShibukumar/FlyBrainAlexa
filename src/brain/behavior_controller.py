from dataclasses import dataclass

import numpy as np


@dataclass
class BehaviorCommand:
    name: str
    locomotion_drive: float
    allow_propulsion: bool


class BehaviorController:
    """Select a simple autonomous behavior from neural and body state."""

    def __init__(self, activation_threshold: float = 0.2) -> None:
        self.activation_threshold = activation_threshold
        self.name = "REST"

    def step(
        self,
        motor_drive: float,
        leg_contacts: np.ndarray,
    ) -> BehaviorCommand:
        contacts = np.asarray(leg_contacts, dtype=float)
        if contacts.shape != (6,):
            raise ValueError("leg_contacts must contain one value per leg")

        support_count = int(np.count_nonzero(contacts > 0.05))
        exploring = motor_drive >= self.activation_threshold
        if exploring:
            self.name = "EXPLORE"
        else:
            self.name = "REST"

        return BehaviorCommand(
            name=self.name,
            locomotion_drive=float(motor_drive if exploring else 0.0),
            allow_propulsion=exploring and support_count >= 3,
        )
