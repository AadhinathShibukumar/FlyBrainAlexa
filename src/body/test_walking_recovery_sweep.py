from src.body.run_walking_recovery_sweep import main


def test_recovery_sweep_module_imports() -> None:
    assert callable(main)
