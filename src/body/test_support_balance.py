import numpy as np

from src.body.support_balance import SupportBalance


def test_support_balance_returns_weighted_error() -> None:
    controller = SupportBalance()
    targets = np.full(12, 0.5)
    mapping = {leg: [i * 2, i * 2 + 1] for i, leg in enumerate(("LF", "LM", "LH", "RF", "RM", "RH"))}
    positions = np.array([[1.0, 0.0, 0.0]] * 6)
    output, error = controller.correction(
        targets,
        mapping,
        positions,
        np.ones(6),
        np.array([0.0, 0.0, 0.0]),
    )
    assert np.allclose(error, [-1.0, 0.0])
    assert not np.allclose(output, targets)


def test_support_balance_damps_velocity() -> None:
    controller = SupportBalance(gain=1.0, limit=1.0)
    mapping = {leg: [i * 2, i * 2 + 1] for i, leg in enumerate(("LF", "LM", "LH", "RF", "RM", "RH"))}
    positions = np.zeros((6, 3))
    _, error = controller.correction(
        np.full(12, 0.5),
        mapping,
        positions,
        np.ones(6),
        np.zeros(3),
        np.array([2.0, 0.0, 0.0]),
    )
    assert error[0] < 0.0


def test_support_balance_accepts_attitude() -> None:
    controller = SupportBalance()
    mapping = {leg: [i * 2, i * 2 + 1] for i, leg in enumerate(("LF", "LM", "LH", "RF", "RM", "RH"))}
    output, error = controller.correction(
        np.full(12, 0.5),
        mapping,
        np.zeros((6, 3)),
        np.ones(6),
        np.zeros(3),
        quaternion=np.array([0.996, 0.087, 0.0, 0.0]),
    )
    assert np.all(np.isfinite(output))
    assert np.all(np.isfinite(error))
