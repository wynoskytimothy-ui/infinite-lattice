"""
Release gates for the standalone 3D complex plane (geometry only).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from plane3d.lattice import BranchKind
from plane3d.psi import (
    imaginary_start,
    swap_meet,
    triple_equalization,
    wing_transform,
)
from plane3d.sequences import canon_on_chain
from plane3d.roots import verify_sqrt_gates
from plane3d.velocity import verify_branch_rotation_gates
from plane3d.spring import verify_i_act_axioms, verify_va_vb_swap_diagonal


@dataclass
class GateReport:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"plane3d gates  {status}"]
        for name, ok in self.checks.items():
            lines.append(f"  {name}: {'ok' if ok else 'FAIL'}")
        for msg in self.failures:
            lines.append(f"  ! {msg}")
        return "\n".join(lines)


def verify_all_gates() -> GateReport:
    checks: dict[str, bool] = {}
    failures: list[str] = []

    # BIT 0 — Layer 0 certificate |z|² = 2n²
    bit0 = True
    for n in (1, 3, 7, 100):
        psi = imaginary_start(n)
        expect = 2 * n * n
        if abs(psi.modulus_squared - expect) > 1e-9:
            bit0 = False
            failures.append(f"BIT0 |z|²: n={n} got {psi.modulus_squared} want {expect}")
    checks["bit0_modulus"] = bit0

    # Triple equalization — one node (3,5,7)
    eq = triple_equalization(3, 5, 7)
    coords = {psi.coord for _, psi in eq.values()}
    checks["triple_equalization"] = len(coords) == 1
    if len(coords) != 1:
        failures.append(f"triple_equalization: {len(coords)} coords")

    # Swap meet — 2-way bank equality
    left, right = swap_meet(3, 5)
    checks["swap_meet"] = left.coord == right.coord
    if left.coord != right.coord:
        failures.append(f"swap_meet: {left.coord} != {right.coord}")

    # κ witness — (12, 5, 15) at equalization node
    from plane3d.key import kappa

    _, psi_eq = eq["ap"]
    k = kappa(psi_eq.z, psi_eq.zeta)
    checks["kappa_witness"] = k == (12, 5, 15)
    if k != (12, 5, 15):
        failures.append(f"kappa_witness: {k} != (12,5,15)")

    # Spring axioms + VA/VB diagonal fixed
    ax = verify_i_act_axioms()
    checks["i_act_axioms"] = all(ax.values())
    checks["va_vb_diagonal"] = verify_va_vb_swap_diagonal()

    # All 32 wings swap_meet (sample 3,11)
    sm32 = True
    for wing in range(1, 9):
        l, r = swap_meet(3, 11, wing=wing)
        if l.coord != r.coord:
            sm32 = False
            failures.append(f"swap_meet wing {wing}: mismatch")
    checks["swap_meet_8_wings"] = sm32

    sqrt_checks = verify_sqrt_gates()
    for name, ok in sqrt_checks.items():
        checks[f"sqrt_{name}"] = ok
        if not ok:
            failures.append(f"sqrt gate failed: {name}")

    branch_checks = verify_branch_rotation_gates()
    for name, ok in branch_checks.items():
        checks[f"branch_{name}"] = ok
        if not ok:
            failures.append(f"branch rotation gate failed: {name}")

    passed = all(checks.values())
    return GateReport(passed=passed, checks=checks, failures=failures)
