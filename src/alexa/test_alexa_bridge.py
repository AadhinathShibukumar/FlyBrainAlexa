from pathlib import Path

from src.alexa.alexa_bridge import AlexaBridge


def request(intent: str) -> dict:
    return {
        "request": {
            "type": "IntentRequest",
            "intent": {"name": intent},
        }
    }


def test_help_response_uses_alexa_envelope(tmp_path: Path) -> None:
    bridge = AlexaBridge(summary_path=tmp_path / "summary.json")
    response = bridge.handle(request("AMAZON.HelpIntent"))
    assert response["version"] == "1.0"
    assert "start fly" in response["response"]["outputSpeech"]["text"]


def test_status_reports_missing_experiment(tmp_path: Path) -> None:
    bridge = AlexaBridge(summary_path=tmp_path / "summary.json")
    response = bridge.handle(request("GetFlyStatusIntent"))
    assert "No fly experiment" in response["response"]["outputSpeech"]["text"]


def test_status_reports_saved_summary(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        '{"falls": 0, "final_support_legs": 6}\n',
        encoding="utf-8",
    )
    bridge = AlexaBridge(summary_path=summary)
    response = bridge.handle(request("GetFlyStatusIntent"))
    text = response["response"]["outputSpeech"]["text"]
    assert "0 falls" in text
    assert "6 supporting legs" in text
