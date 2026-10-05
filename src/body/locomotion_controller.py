from dataclasses import dataclass

import numpy as np


@dataclass
class LocomotionState:
    phases: tuple[str, ...]
    support_count: int
    propulsion_enabled: bool


class StanceSwingController:
    """Coordinate tripod stance and swing phases from contact feedback."""

    def __init__(self) -> None:
        self.tripods = ((0, 4, 5), (1, 2, 3))
        self.phase = 0.0
        self.states = ["stance"] * 6

    def step(
        self,
        contacts: np.ndarray,
        neural_amplitudes: np.ndarray,
        contact_confidence: np.ndarray | None = None,
        dt: float = 0.05,
    ) -> tuple[np.ndarray, LocomotionState]:
        contacts = np.asarray(contacts, dtype=float)
        amplitudes = np.asarray(neural_amplitudes, dtype=float)
        if contacts.shape != (6,) or amplitudes.shape != (6,):
            raise ValueError("contacts and neural_amplitudes must have shape (6,)")
        if contact_confidence is None:
            confidence = contacts
        else:
            confidence = np.asarray(contact_confidence, dtype=float)
            if confidence.shape != (6,):
                raise ValueError("contact_confidence must have shape (6,)")

        self.phase = (self.phase + 4.0 * dt) % (2.0 * np.pi)
        support = confidence > 0.2
        support_count = int(np.count_nonzero(support))
        active_tripod = self.tripods[
            0 if np.sin(self.phase) >= 0.0 else 1
        ]
        swing_tripod = self.tripods[
            1 if np.sin(self.phase) >= 0.0 else 0
        ]

        for leg in active_tripod:
            self.states[leg] = "stance" if support[leg] else "recover"
        for leg in swing_tripod:
            self.states[leg] = "swing"

        propulsion_enabled = support_count >= 3
        output = np.clip(amplitudes, 0.0, 1.0)
        output *= np.clip(0.5 + 0.5 * confidence, 0.25, 1.0)
        for leg in swing_tripod:
            output[leg] *= 0.7
        if not propulsion_enabled:
            output *= 0.5

        return output, LocomotionState(
            phases=tuple(self.states),
            support_count=support_count,
            propulsion_enabled=propulsion_enabled,
        )
