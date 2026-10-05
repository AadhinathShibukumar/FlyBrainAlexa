import numpy as np

from src.body.recovery_controller import RecoveryController


def test_recovery_stops_when_support_collapses() -> None:
    controller = RecoveryController()
    assert controller.scale(1, 0.0, 0.0, np.zeros(3)) == 0.0


def test_recovery_scales_high_tilt() -> None:
    controller = RecoveryController(max_tilt_deg=45.0)
    scale = controller.scale(6, np.radians(22.5), 0.0, np.zeros(3))
    assert 0.25 < scale < 1.0
