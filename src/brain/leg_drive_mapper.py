import numpy as np


class LegDriveMapper:
    """Expand side-level descending activity into six independent test channels.

    The current FlyWire annotation snapshot does not identify front/middle/hind
    descending pathways, so these gains are an explicit, replaceable fallback.
    """

    LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")

    def __init__(self, smoothing: float = 0.2) -> None:
        self.smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self.previous = np.zeros(6, dtype=float)

    def step(self, left_drive: float, right_drive: float) -> dict[str, float]:
        side_drives = np.array(
            [left_drive, left_drive, left_drive, right_drive, right_drive, right_drive],
            dtype=float,
        )
        anatomical_gains = np.array([0.92, 1.0, 1.08, 0.92, 1.0, 1.08])
        target = np.clip(side_drives * anatomical_gains, 0.0, 1.0)
        output = (1.0 - self.smoothing) * self.previous + self.smoothing * target
        self.previous = output
        return dict(zip(self.LEGS, output.tolist()))
