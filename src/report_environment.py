"""Report the environment selected for local experiments."""

from __future__ import annotations

import json

from dunlin.device import select_device
from dunlin.environment import collect_environment


def main() -> None:
    device, dtype = select_device()
    print(json.dumps(collect_environment(device, dtype), indent=2))


if __name__ == "__main__":
    main()
