import numpy as np


class TripodGait:

    def __init__(self):
        self.phase = 0.0
        self.frequency = 4.0
        self.amplitude = 1.0

    def step(self, dt=0.05, amplitudes=None, contacts=None):

        self.phase += self.frequency * dt

        gait = np.full((6, 24), 0.5)
        if amplitudes is None:
            amplitudes = np.full(6, self.amplitude)
        else:
            amplitudes = np.asarray(amplitudes, dtype=float)
            if amplitudes.shape != (6,):
                raise ValueError("amplitudes must contain one value per leg")
        if contacts is None:
            contacts = np.zeros(6, dtype=float)
        else:
            contacts = np.asarray(contacts, dtype=float)
            if contacts.shape != (6,):
                raise ValueError("contacts must contain one value per leg")
        support_count = int(np.count_nonzero(contacts > 0.05))
        support_scale = 1.0 if support_count >= 3 else 0.5

        tripod_a = [0, 4, 5]  # LF RM RH
        tripod_b = [1, 2, 3]  # LM LH RF

        for leg in tripod_a + tripod_b:
            leg_phase = self.phase + (0.0 if leg in tripod_a else np.pi)
            swing = np.sin(leg_phase)
            lift = max(0.0, np.sin(leg_phase))
            contact_scale = 1.0 if contacts[leg] > 0.05 else 0.8
            drive = np.clip(
                amplitudes[leg] * contact_scale * support_scale,
                0.0,
                1.0,
            )

            # Each leg has three articulated pitch groups: coxa, femur, tibia.
            # Keep roll/yaw and distal tarsus joints at their neutral midpoint.
            gait[leg, 0] = 0.5 + 0.38 * drive * swing
            gait[leg, 3] = 0.5 - 0.52 * drive * swing
            gait[leg, 6] = 0.5 + 0.68 * drive * lift

            # Carry the proximal pattern through the tarsus while preserving
            # a smaller distal motion for stable ground contact.
            gait[leg, 9::3] = 0.5 + 0.30 * drive * lift

        return gait
