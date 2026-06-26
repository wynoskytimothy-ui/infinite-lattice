#!/usr/bin/env python3
"""Walk the unit lattice — bare lumber vs infinite procedural 3D space."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lattice_retriever_v1.unit_lattice_codec import demo_digits_land, encode_bare_lumber


def main() -> None:
    import random

    rng = random.Random(0)
    data = bytes(rng.randint(0, 9) for _ in range(100_000))
    _, wire, fp = encode_bare_lumber(data)
    report = {
        "demo_0_9": demo_digits_land(),
        "random_100k_digits": {
            "raw_bytes": len(data),
            "bare_lumber_wire_bytes": len(wire),
            "footprint": fp.explain(),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
