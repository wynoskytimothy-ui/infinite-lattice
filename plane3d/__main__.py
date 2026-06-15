"""python -m plane3d — run geometry gates + demo."""

from plane3d.gates import verify_all_gates
from plane3d.psi import demo as psi_demo


def main() -> int:
    print(verify_all_gates().summary())
    print()
    psi_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
