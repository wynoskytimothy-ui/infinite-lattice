"""plane3d must be standalone — geometry only, no RAG imports."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import plane3d as P

FORBIDDEN_ROOTS = (
    "aethos_hub",
    "aethos_discriminative",
    "aethos_promotion",
    "eval_beir",
    "aethos_pipeline",
    "aethos_physics",
    "aethos_token",
    "pipeline.bit_",
)


def _plane3d_py_files() -> list[Path]:
    root = Path(__file__).resolve().parent.parent / "plane3d"
    return sorted(p for p in root.glob("*.py") if p.name != "__pycache__")


def _import_roots_in_file(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


class TestPlane3dStandalone(unittest.TestCase):
    def test_no_forbidden_imports_in_package(self):
        for path in _plane3d_py_files():
            roots = _import_roots_in_file(path)
            for bad in FORBIDDEN_ROOTS:
                for r in roots:
                    self.assertFalse(
                        r.startswith(bad) or bad in r,
                        f"{path.name} imports forbidden root via {r!r}",
                    )

    def test_only_plane3d_and_stdlib_roots(self):
        allowed = {
            "plane3d", "__future__", "dataclasses", "enum", "math", "cmath",
            "typing", "collections", "itertools",
        }
        for path in _plane3d_py_files():
            roots = _import_roots_in_file(path)
            extra = roots - allowed
            self.assertEqual(extra, set(), f"{path.name} unexpected imports: {extra}")

    def test_gates_pass(self):
        report = P.verify_all_gates()
        self.assertTrue(report.passed, report.summary())

    def test_imaginary_start(self):
        psi = P.imaginary_start(7)
        self.assertEqual(psi.z, complex(7, 7))
        self.assertAlmostEqual(psi.modulus_squared, 2 * 7 * 7)

    def test_swap_meet(self):
        left, right = P.swap_meet(3, 5)
        self.assertEqual(left.coord, right.coord)

    def test_kappa_witness(self):
        eq = P.triple_equalization(3, 5, 7)
        _, psi = eq["ap"]
        self.assertEqual(P.kappa(psi.z, psi.zeta), (12, 5, 15))

    def test_sqrt_spring(self):
        self.assertAlmostEqual(P.sqrt_spring(-1.0, 0).imag, 1.0)
        z = 3 + 4j
        r = P.sqrt_spring(z, 0)
        self.assertAlmostEqual(abs(P.spring_square(r) - z), 0.0, places=9)

    def test_sqrt_depth_layer0(self):
        psi = P.imaginary_start(7)
        back = P.depth_square(psi)
        self.assertAlmostEqual(back.zeta, 7.0)
        self.assertAlmostEqual(P.layer0_depth_from_spring(psi.z), 7.0)

    def test_sqrt_gates(self):
        checks = P.verify_sqrt_gates()
        self.assertTrue(all(checks.values()), checks)

    def test_wing_sheet_mask(self):
        self.assertEqual(P.wing_sheet_mask(1).flat(), 0)
        self.assertEqual(P.wing_sheet_mask(2).depth_branch, 1)

    def test_branch_rotation_gates(self):
        checks = P.verify_branch_rotation_gates()
        self.assertTrue(all(checks.values()), checks)

    def test_four_branch_rotation_planes(self):
        planes = P.four_branch_rotation_planes((3.0, 5.0, 7.0), 5.0)
        self.assertEqual(len(planes), 4)
        coords = {(p.psi.z, p.psi.zeta) for p in planes}
        self.assertEqual(len(coords), 4)
        par0, _ = P.decompose_around_critical(planes[0].psi.z)
        for p in planes:
            par, _ = P.decompose_around_critical(p.psi.z)
            self.assertAlmostEqual(par.real, par0.real, places=6)
            self.assertAlmostEqual(par.imag, par0.imag, places=6)

    def test_segment_velocity_interior(self):
        seg, vel = P.segment_velocity((3.0, 5.0, 7.0), 6.0)
        self.assertEqual(vel, 0.5)
        self.assertGreater(seg, 0)
        self.assertLess(seg, 3)


if __name__ == "__main__":
    unittest.main()
