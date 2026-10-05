from enum import Enum

import numpy as np


class FlightPhase(Enum):
    GROUNDED = "grounded"
    TAKEOFF = "takeoff"
    AIRBORNE = "airborne"
    LANDING = "landing"


class TakeoffController:
    """Ramp wing drive and classify ground contact during flight tests."""

    def __init__(
        self,
        ramp_seconds: float = 0.75,
        motor_threshold: float = 0.2,
        airborne_height: float = 1.5,
        contact_release_frames: int = 3,
    ) -> None:
        self.ramp_seconds = ramp_seconds
        self.motor_threshold = motor_threshold
        self.airborne_height = airborne_height
        self.phase = FlightPhase.GROUNDED
        self.elapsed = 0.0
        self.contact_release_frames = contact_release_frames
        self._no_contact_frames = 0

    def step(
        self,
        motor_drive: float,
        contacts: np.ndarray,
        height: float,
        dt: float,
    ) -> tuple[float, FlightPhase]:
        motor_drive = float(np.clip(motor_drive, 0.0, 1.0))
        has_support = bool(np.any(np.asarray(contacts) > 0.0))
        self._no_contact_frames = (
            0 if has_support else self._no_contact_frames + 1
        )

        if has_support and motor_drive >= self.motor_threshold:
            if self.phase in (FlightPhase.GROUNDED, FlightPhase.LANDING):
                self.phase = FlightPhase.TAKEOFF
                self.elapsed = 0.0
            self.elapsed += max(float(dt), 0.0)
        elif (
            not has_support
            and self._no_contact_frames >= self.contact_release_frames
            and height > self.airborne_height
        ):
            self.phase = FlightPhase.AIRBORNE
        elif has_support:
            self.phase = (
                FlightPhase.LANDING
                if self.phase is FlightPhase.AIRBORNE
                else FlightPhase.GROUNDED
            )
            self.elapsed = 0.0

        if self.phase is FlightPhase.TAKEOFF:
            ramp = np.clip(self.elapsed / self.ramp_seconds, 0.0, 1.0)
            drive = motor_drive * float(ramp)
            if ramp >= 1.0 and self._no_contact_frames >= self.contact_release_frames:
                self.phase = FlightPhase.AIRBORNE
            return drive, self.phase

        if self.phase is FlightPhase.AIRBORNE:
            return motor_drive if not has_support else 0.0, self.phase

        if self.phase is FlightPhase.LANDING:
            return 0.0, self.phase

        return 0.0, self.phase
