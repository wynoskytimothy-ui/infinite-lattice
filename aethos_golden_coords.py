"""
Golden coordinate fixtures — regression for VA1–VA4 and selected wings.

Regenerate after intentional formula changes:
  python -m aethos_golden_coords --write
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aethos_lattice import BranchKind, LatticeId
from aethos_recursive import LatticeBank32, LatticeBank32K, canon_recursive
from aethos_sequences import canon_on_chain

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "golden_coords.json"


def _load_fixture() -> dict[str, Any]:
    if not FIXTURE_PATH.is_file():
        raise FileNotFoundError(f"missing {FIXTURE_PATH}; run: python -m aethos_golden_coords --write")
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def verify_golden_coords() -> list[str]:
    """Return list of mismatch messages (empty = pass)."""
    doc = _load_fixture()
    errors: list[str] = []

    for case in doc.get("cases", []):
        chain = tuple(case["chain"])
        n = int(case["n"])
        label = f"k={case['k']} chain={chain} n={n}"

        for bname, xyz in case.get("canonical", {}).items():
            branch = BranchKind[bname]
            got = canon_recursive(branch, chain, n)
            exp = tuple(xyz)
            if got != exp:
                errors.append(f"{label} {bname} canonical: got {got} != {exp}")

        chain_va1 = case.get("chain_va1")
        if chain_va1 is not None:
            got = canon_on_chain(BranchKind.VA1, chain, n)
            exp = tuple(chain_va1)
            if got != exp:
                errors.append(f"{label} chain_va1: got {got} != {exp}")

        bank = LatticeBank32K(chain) if len(chain) > 1 else None
        for lid_name, xyz in case.get("wings", {}).items():
            lid = LatticeId[lid_name]
            if bank:
                got = bank[lid].at(n)
            else:
                got = LatticeBank32.single_prime(chain[0])[lid].at(n)
            exp = tuple(xyz)
            if got != exp:
                errors.append(f"{label} wing {lid_name}: got {got} != {exp}")

    return errors


def write_fixture(path: Path | None = None) -> Path:
    """Regenerate golden_coords.json from current formulas."""
    path = path or FIXTURE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []

    def add_case(k: int, chain: tuple[int, ...], n: int) -> None:
        entry: dict[str, Any] = {
            "k": k,
            "chain": list(chain),
            "n": n,
            "canonical": {},
            "wings": {},
            "chain_va1": list(canon_on_chain(BranchKind.VA1, chain, n)),
        }
        for b in BranchKind:
            entry["canonical"][b.name] = list(canon_recursive(b, chain, n))
        if len(chain) == 1:
            bank = LatticeBank32.single_prime(chain[0])
        else:
            bank = LatticeBank32K(chain)
        for lid in (LatticeId.L01, LatticeId.L16, LatticeId.L32):
            entry["wings"][lid.name] = list(bank[lid].at(n))
        cases.append(entry)

    add_case(1, (5,), 7)
    add_case(2, (3, 11), 5)
    add_case(2, (3, 11), 11)
    add_case(2, (3, 11), 12)
    add_case(3, (3, 5, 7), 5)
    add_case(3, (3, 5, 7), 7)
    add_case(4, (3, 5, 7, 11), 7)
    add_case(4, (3, 5, 7, 11), 11)

    doc = {"version": 1, "cases": cases}
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def swap_meet_solo_all_wings(p: int, q: int) -> bool:
    """Solo anchor p at n=q meets solo anchor q at n=p on all 32 wings."""
    left = LatticeBank32.single_prime(p)
    right = LatticeBank32.single_prime(q)
    for lid in LatticeId:
        if left[lid].at(q) != right[lid].at(p):
            return False
    return True


def shallow_deep_meet_at_swap(p: int, q: int) -> bool:
    """Shallow (p,) bank at n=q meets deep (p,q) bank at n=p on all 32 wings."""
    if p > q:
        p, q = q, p
    shallow = LatticeBank32K((p,))
    deep = LatticeBank32K((p, q))
    for lid in LatticeId:
        if shallow[lid].at(q) != deep[lid].at(p):
            return False
    return True


def main() -> None:
    import sys

    if "--write" in sys.argv:
        out = write_fixture()
        print(f"Wrote {out}")
        return
    errs = verify_golden_coords()
    if errs:
        for e in errs:
            print(e)
        sys.exit(1)
    print(f"OK — {len(_load_fixture()['cases'])} golden cases verified")


if __name__ == "__main__":
    main()
