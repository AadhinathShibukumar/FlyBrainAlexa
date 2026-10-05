import numpy as np

from src.body.muscle_activation import MuscleActivation


def test_activation_is_bounded_and_smooth() -> None:
    activation = MuscleActivation(3, activation_rate=2.0)
    first = activation.step(np.ones(3), 0.1)
    second = activation.step(np.ones(3), 0.1)
    assert np.allclose(first, 0.7)
    assert np.allclose(second, 0.9)


def test_activation_rejects_wrong_shape() -> None:
    activation = MuscleActivation(2)
    try:
        activation.step(np.ones(3), 0.1)
    except ValueError:
        return
    raise AssertionError("wrong-shaped activation target was accepted")


def test_command_rate_limit() -> None:
    activation = MuscleActivation(1, max_command_rate=2.0)
    activation.activation[:] = 0.5
    command = activation.command(np.ones(1), 0.1)
    assert np.allclose(command, 0.7)
