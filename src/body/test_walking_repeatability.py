from src.body.run_walking_repeatability import summarize_trials


def test_repeatability_summary_reports_pass_rate_and_ranges() -> None:
    result = summarize_trials(
        [
            {
                "passed": True,
                "summary": {"average_speed": 1.0, "max_tilt_deg": 5.0},
            },
            {
                "passed": False,
                "summary": {"average_speed": 2.0, "max_tilt_deg": 8.0},
            },
        ]
    )
    assert result["trial_count"] == 2
    assert result["passed_count"] == 1
    assert result["pass_rate"] == 0.5
    assert result["average_speed_min"] == 1.0
    assert result["max_tilt_max"] == 8.0
