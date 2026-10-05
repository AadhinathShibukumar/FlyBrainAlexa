import numpy as np


class TripodGait:

    def __init__(self):
        self.phase = 0.0
        self.frequency = 4.0
        self.amplitude = 1.0

    def step(self, dt=0.05):

        self.phase += self.frequency * dt

        gait = np.zeros((6, 24))

        tripod_a = [0, 4, 5]  # LF RM RH
        tripod_b = [1, 2, 3]  # LM LH RF

        a = ((np.sin(self.phase) + 1.0) * 0.5 * self.amplitude)
        b = ((np.sin(self.phase + np.pi) + 1.0) * 0.5 * self.amplitude)

        for leg in tripod_a:
            gait[leg, :] = a

        for leg in tripod_b:
            gait[leg, :] = b

        return gait
