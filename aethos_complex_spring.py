"""
Spring = complex plane at every trigger node x 4 branches.

Physics mapping (Section 2 electron):
  - Spring elongation u ~ Re(psi); tension T = k_s u
  - Superposition = photon mid-bounce on the spring (complex phase)
  - |psi|^2 ~ |z|^2 ~ T^2 (Born from spring, section 02 derivations)

Lattice mapping:
  - Trigger node: transgressor n crosses anchor p_i (velocity boundary)
  - Complex coordinate: z = X + i Y from canon_va*(P, n)
  - Four branches VA1..VA4 = four spring states in C at that trigger
  - Z axis = extension depth (transgressor / composition lock), orthogonal to spring plane
"""

from __future__ import annotations

from dataclasses import dataclass

from aethos_lattice import BranchKind
from aethos_recursive import canon_recursive, segment_index


@dataclass(frozen=True)
class TriggerNode:
    """One anchor crossing — spring 'bounce' in the electron picture."""

    chain: tuple[int, ...]
    n: int
    segment: int
    anchor_index: int | None  # which p_i was crossed, if any

    @classmethod
    def at(cls, chain: tuple[int, ...], n: int) -> TriggerNode:
        seg = segment_index(chain, n)
        idx = None
        for i, p in enumerate(chain):
            if n == p:
                idx = i
                break
        return cls(chain=chain, n=n, segment=seg, anchor_index=idx)

    @property
    def is_anchor_crossing(self) -> bool:
        return self.anchor_index is not None


@dataclass(frozen=True)
class SpringState:
    """One branch's spring point in the complex plane at a trigger."""

    branch: BranchKind
    z: complex
    depth: float  # Z from formula (spring extension / lattice depth)

    @property
    def tension_squared(self) -> float:
        """|z|^2 proxy — ties to Born P ∝ T^2 in section 2."""
        return abs(self.z) ** 2

    @property
    def real_spring(self) -> float:
        return self.z.real

    @property
    def imag_phase(self) -> float:
        return self.z.imag


def spring_states_at(chain: tuple[int, ...], n: int) -> tuple[TriggerNode, dict[BranchKind, SpringState]]:
    """All four branch complex planes at one transgressor n."""
    trig = TriggerNode.at(chain, n)
    states: dict[BranchKind, SpringState] = {}
    for b in BranchKind:
        x, y, z = canon_recursive(b, chain, n)
        states[b] = SpringState(branch=b, z=complex(x, y), depth=float(z))
    return trig, states


def mirror_pairs(states: dict[BranchKind, SpringState]) -> list[tuple[BranchKind, BranchKind]]:
    """Branch pairs related by Im(z) -> -Im(z) with same Re(z) (spring reflection)."""
    pairs: list[tuple[BranchKind, BranchKind]] = []
    items = list(states.items())
    for i, (a, sa) in enumerate(items):
        for b, sb in items[i + 1 :]:
            if sa.z.real == sb.z.real and sa.z.imag == -sb.z.imag:
                pairs.append((a, b))
    return pairs


def demo() -> None:
    print("=" * 70)
    print("SPRING = COMPLEX PLANE at trigger nodes x 4 branches")
    print("=" * 70)

    print("\n--- k=1 solo anchor (electron-like): trigger at n=p=5 ---")
    trig, st = spring_states_at((5,), 5)
    print(f"  trigger: n={trig.n} segment={trig.segment} anchor_cross={trig.is_anchor_crossing}")
    for b in BranchKind:
        s = st[b]
        print(f"  {b.name}: z = {s.z.real:.0f} + {s.z.imag:.0f}i   |z|^2={s.tension_squared:.0f}   Z={s.depth:.0f}")
    mp = mirror_pairs(st)
    print(f"  Y-mirror pairs (same Re, opposite Im): {[(a.name, b.name) for a,b in mp]}")

    print("\n--- k=3 chain (3,5,7): each anchor is a trigger node ---")
    chain = (3, 5, 7)
    for n in chain:
        trig, st = spring_states_at(chain, n)
        print(f"\n  TRIGGER n={n} (crosses p_{trig.anchor_index}) seg={trig.segment}")
        for b in BranchKind:
            s = st[b]
            print(f"    {b.name}: z={s.z.real:.0f}{s.z.imag:+.0f}i  |z|^2={s.tension_squared:.0f}  Z={s.depth:.0f}")
        mp = mirror_pairs(st)
        if mp:
            print(f"    mirror pairs: {[(a.name,b.name) for a,b in mp]}")

    print("\n--- Interpretation ---")
    print("  (X,Y) = complex spring plane: Re ~ displacement, Im ~ phase / side")
    print("  Z     = extension along transgressor (compressed at observation)")
    print("  VA1/VA2 often Y-mirror at triggers -> conjugate-like spring pair")
    print("  4 branches = 4 spring corners at each bounce (4-way setBranch fan)")
    print("  |z|^2  = tension^2 proxy -> Born rule (section 2.8)")


if __name__ == "__main__":
    demo()
