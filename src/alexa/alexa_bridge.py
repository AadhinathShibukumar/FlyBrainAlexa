"""Local Alexa-compatible bridge for the autonomous fly benchmark."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


class AlexaBridge:
    """Translate Alexa request envelopes into managed fly experiment actions."""

    def __init__(
        self,
        summary_path: str | Path = "data/alexa_fly_summary.json",
        telemetry_path: str | Path = "data/alexa_fly_telemetry.csv",
        steps: int = 200,
        physics_substeps: int = 10,
        neural_interval: int = 5,
    ) -> None:
        if steps <= 0 or physics_substeps <= 0 or neural_interval <= 0:
            raise ValueError("steps, physics_substeps, and neural_interval must be positive")
        self.summary_path = Path(summary_path)
        self.telemetry_path = Path(telemetry_path)
        self.steps = steps
        self.physics_substeps = physics_substeps
        self.neural_interval = neural_interval
        self.process: subprocess.Popen[str] | None = None

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        request_type = request.get("request", {}).get("type")
        if request_type == "LaunchRequest":
            return self._response("Say start fly to begin the autonomous experiment.")
        if request_type == "SessionEndedRequest":
            return self._response("Goodbye.", should_end=True)
        if request_type != "IntentRequest":
            return self._response("I did not understand that request.")

        intent = request.get("request", {}).get("intent", {}).get("name")
        if intent in {"StartFlyIntent", "StartExperimentIntent"}:
            return self._start()
        if intent in {"StopFlyIntent", "StopExperimentIntent"}:
            return self._stop()
        if intent in {"GetFlyStatusIntent", "FlyStatusIntent"}:
            return self._status()
        if intent == "AMAZON.HelpIntent":
            return self._response(
                "You can say start fly, stop fly, or get fly status."
            )
        if intent == "AMAZON.StopIntent":
            return self._stop()
        return self._response("Try saying start fly, stop fly, or get fly status.")

    def _start(self) -> dict[str, Any]:
        if self.process is not None and self.process.poll() is None:
            return self._response("The fly experiment is already running.")
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "src.body.run_walking_benchmark",
            "--steps",
            str(self.steps),
            "--physics-substeps",
            str(self.physics_substeps),
            "--neural-interval",
            str(self.neural_interval),
            "--telemetry",
            str(self.telemetry_path),
            "--summary",
            str(self.summary_path),
        ]
        self.process = subprocess.Popen(command, text=True)
        return self._response("The autonomous fly experiment has started.")

    def _stop(self) -> dict[str, Any]:
        if self.process is None or self.process.poll() is not None:
            return self._response("The fly experiment is not running.")
        self.process.terminate()
        return self._response("The autonomous fly experiment has been stopped.")

    def _status(self) -> dict[str, Any]:
        if self.process is not None and self.process.poll() is None:
            return self._response("The fly experiment is currently running.")
        if self.summary_path.exists():
            summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
            return self._response(
                "The latest fly experiment finished with "
                f"{summary.get('falls', 'unknown')} falls and "
                f"{summary.get('final_support_legs', 'unknown')} supporting legs."
            )
        return self._response("No fly experiment has been run yet.")

    @staticmethod
    def _response(text: str, should_end: bool = False) -> dict[str, Any]:
        return {
            "version": "1.0",
            "response": {
                "outputSpeech": {"type": "PlainText", "text": text},
                "shouldEndSession": should_end,
            },
        }
