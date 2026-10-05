from src.body.validate_walking_summary import validate_summary


def test_stable_summary_passes() -> None:
    assert validate_summary(
        {"falls": 0, "max_tilt_deg": 7.0, "final_support_legs": 6}
    ) == []


def test_unstable_summary_reports_all_failures() -> None:
    failures = validate_summary(
        {"falls": 1, "max_tilt_deg": 20.0, "final_support_legs": 0}
    )
    assert len(failures) == 3
