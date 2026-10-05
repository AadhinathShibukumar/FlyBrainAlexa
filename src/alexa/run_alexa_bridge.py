"""Run the local Alexa bridge using one JSON request per stdin line."""

import json
import sys

from src.alexa.alexa_bridge import AlexaBridge


def main() -> None:
    bridge = AlexaBridge()
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        print(json.dumps(bridge.handle(request)), flush=True)


if __name__ == "__main__":
    main()
