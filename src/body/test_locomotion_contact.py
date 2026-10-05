import numpy as np

from src.body.locomotion_controller import StanceSwingController


def test_force_confidence_gates_leg_output() -> None:
    controller = StanceSwingController()
    amplitudes = np.ones(6)
    contacts = np.ones(6)
    low = np.zeros(6)
    high = np.ones(6)
    low_output, _ = controller.step(contacts, amplitudes, low)
    high_output, _ = controller.step(contacts, amplitudes, high)
    assert np.all(low_output < high_output)
