"""
BIT 1 — Word → SpacetimeCell

Math (pipeline contract):
  chain(w) = sorted unique parent_primes(w)  [intersection-only]
          or parent_primes(w) ∪ {prime(w)}   [pool-promoted L3]
  Ψ₀(w) = wing_transform(VA1, chain(w), n₀, wing=1)     [hub / L01 canon]
  Ψ_k(w) = rotate_around_critical(Ψ₀, k)  for k = 0..3   [4-way branch fan]
  cell(w) = SpacetimeCell.from_psi(Ψ_k, n₀, chain, branch, wing)

VA1 (k=0) stays on the hub formula_coord path.  VA2..VA4 use plane3d rotation
around the imaginary line j = n+ni instead of legacy FSM re-canonicalize.

Chain must match PromotionRegistry.lattice_address (hub coord path), not the
notch spring chain (chain_for_word in aethos_notch_encoder may differ).

Gate: cell(w).z == complex(hub_x, hub_y) and cell(w).zeta == hub_z at n₀, L01.
"""

from __future__ import annotations

from typing import Sequence

from aethos_core import formula_coord
from aethos_lattice import BranchKind, LatticeId
from aethos_complex_plane import ComplexPlane3D
from aethos_physics import SpacetimeCell
from aethos_promotion import LatticeTier
from plane3d.velocity import (
    four_branch_rotation_planes,
    rotate_around_critical_line,
    segment_velocity,
)

DEFAULT_ANCHOR_N = 7


def branch_kind_to_k(branch: BranchKind) -> int:
    """VA1..VA4 → rotation index 0..3."""
    return int(branch) - 1


def branch_k_to_kind(branch_k: int) -> BranchKind:
    return BranchKind((branch_k % 4) + 1)


def chain_for_lattice_cell(registry, word: str) -> tuple[int, ...]:
    """
    L3 anchor chain for SpacetimeCell — same rule as lattice_address / hub coords.

    intersection-only: sorted unique parent_primes
    pool-promoted:     parent_primes ∪ {word prime}
    """
    tok = registry.resolve_token(word.lower())
    if tok.intersection_only:
        return tuple(sorted(set(int(p) for p in tok.parent_primes)))
    return tuple(sorted(set(int(p) for p in tok.parent_primes + (tok.prime,))))


def spacetime_cell_at_branch(
    chain: Sequence[int | float],
    n: float,
    branch: BranchKind,
    wing: int = 1,
    *,
    lock_interior: bool = True,
    apply_velocity: bool = True,
) -> SpacetimeCell:
    """
    SpacetimeCell at branch b via rotation around j.

    VA1 uses hub canon (wing_transform).  VA2..VA4 rotate (z, ζ) in the u×ζ plane.
    """
    k = branch_kind_to_k(branch)
    if k == 0:
        return SpacetimeCell.at(
            chain,
            n,
            branch,
            wing,
            lock_interior=lock_interior,
        )
    base = SpacetimeCell.at(
        chain,
        n,
        BranchKind.VA1,
        wing,
        lock_interior=lock_interior,
    )
    _, vel = segment_velocity(chain, n)
    zeta_in = base.zeta * vel if apply_velocity else base.zeta
    z_rot, zeta_rot = rotate_around_critical_line(base.z, zeta_in, k)
    return SpacetimeCell.from_psi(
        ComplexPlane3D(z=z_rot, zeta=zeta_rot),
        n,
        chain=tuple(chain),
        branch=branch,
        wing=wing,
    )


def four_branch_cells(
    chain: Sequence[int | float],
    n: float,
    *,
    wing: int = 1,
    lock_interior: bool = True,
) -> tuple[SpacetimeCell, ...]:
    """Four velocity-tier planes at quarter-turns k=0..3 around j."""
    planes = four_branch_rotation_planes(
        chain,
        n,
        wing=wing,
        lock_interior=lock_interior,
    )
    return tuple(
        SpacetimeCell.from_psi(
            p.psi,
            n,
            chain=tuple(chain),
            branch=branch_k_to_kind(p.branch_k),
            wing=wing,
        )
        for p in planes
    )


def word_to_spacetime_cell(
    registry,
    word: str,
    *,
    n: int = DEFAULT_ANCHOR_N,
    branch: BranchKind = BranchKind.VA1,
    wing: int = 1,
    lock_interior: bool = True,
) -> SpacetimeCell:
    """Map vocabulary token to SpacetimeCell at anchor rail n₀ (default 7)."""
    chain = chain_for_lattice_cell(registry, word)
    return spacetime_cell_at_branch(
        chain,
        n,
        branch,
        wing,
        lock_interior=lock_interior,
    )


def hub_formula_coord(
    registry,
    word: str,
    *,
    n: int = DEFAULT_ANCHOR_N,
    lattice_id: LatticeId = LatticeId.L01,
) -> tuple[float, float, float]:
    """Reference coord from hub path (lattice_address at L01)."""
    return registry.lattice_address(
        word,
        LatticeTier.L3_WORD,
        n,
        lattice_id,
    )


def verify_bit01_gate(
    registry,
    words: Sequence[str] | None = None,
    *,
    n: int = DEFAULT_ANCHOR_N,
    tol: float = 1e-9,
    max_words: int = 100,
) -> tuple[int, int, list[tuple[str, str]]]:
    """
    BIT 1 gate: cell(w) matches hub formula_coord for each word.

    Returns (passed, total, failures) where each failure is (word, reason).
    """
    if words is None:
        words = sorted(registry.word_counts.keys())[:max_words]
    passed = 0
    failures: list[tuple[str, str]] = []
    for w in words:
        try:
            cell = word_to_spacetime_cell(registry, w, n=n)
            x, y, z = hub_formula_coord(registry, w, n=n)
        except Exception as exc:
            failures.append((w, f"build error: {exc}"))
            continue
        if abs(cell.z.real - x) > tol or abs(cell.z.imag - y) > tol:
            failures.append(
                (
                    w,
                    f"z mismatch: cell={cell.z!r} hub=({x},{y}) chain={cell.chain!r}",
                )
            )
            continue
        if abs(cell.zeta - z) > tol:
            failures.append(
                (
                    w,
                    f"zeta mismatch: cell={cell.zeta} hub={z} chain={cell.chain!r}",
                )
            )
            continue
        # Cross-check: wing path equals formula_coord for same chain
        chain = chain_for_lattice_cell(registry, w)
        fx, fy, fz = formula_coord(chain, n, LatticeId.L01)
        if abs(fx - x) > tol or abs(fy - y) > tol or abs(fz - z) > tol:
            failures.append((w, "chain_for_lattice_cell != lattice_address chain"))
            continue
        passed += 1
    return passed, len(words), failures


def verify_bit01_rotation_gate(
    *,
    tol: float = 1e-9,
) -> tuple[bool, list[str]]:
    """
    BIT 1 rotation gate:
      - VA1 rotation path matches SpacetimeCell.at (hub canon)
      - four_branch_cells gives 4 distinct (z, zeta) at (3,5,7) n=5
      - j-parallel component invariant across the four branches
    """
    from plane3d.velocity import decompose_around_critical

    failures: list[str] = []
    chain = (3, 5, 7)
    n = 5.0

    for branch in BranchKind:
        legacy = SpacetimeCell.at(chain, n, branch, 1)
        if branch == BranchKind.VA1:
            rotated = spacetime_cell_at_branch(chain, n, branch, 1)
            if abs(rotated.z - legacy.z) > tol or abs(rotated.zeta - legacy.zeta) > tol:
                failures.append(f"VA1 rotation path diverged from hub: {rotated} vs {legacy}")
        else:
            rotated = spacetime_cell_at_branch(chain, n, branch, 1)
            if rotated.z == legacy.z and rotated.zeta == legacy.zeta:
                failures.append(f"{branch.name}: rotation matches legacy FSM (unexpected tie)")

    cells = four_branch_cells(chain, n)
    coords = {(c.z, c.zeta) for c in cells}
    if len(coords) != 4:
        failures.append(f"four_branch_cells expected 4 distinct coords, got {len(coords)}")

    par0, _ = decompose_around_critical(cells[0].z)
    for c in cells[1:]:
        par, _ = decompose_around_critical(c.z)
        if abs(par - par0) > tol:
            failures.append("j-parallel component not invariant across branch fan")

    return (len(failures) == 0, failures)
