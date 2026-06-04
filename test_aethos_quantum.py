"""2-qubit quantum MVP tests. Run: python test_aethos_quantum.py"""

from __future__ import annotations

import math
import unittest

from aethos_blob import ElectronBlob
from aethos_ocean_graph import OceanGraph
from aethos_physics import bell_correlation_qm, chsh_s_quantum
from aethos_quantum import (
    TwoQubitRegister,
    aethos_correlation_from_fill,
    aethos_dephase,
    bell_correlation_register,
    bell_correlation_with_fill,
    chsh_s_aethos,
    chsh_s_register,
    demo_bell_chsh,
    entangled_weight,
    register_from_ocean_pair,
    session_from_ocean,
)


class TestTwoQubitRegister(unittest.TestCase):
    def test_singlet_correlation(self):
        reg = TwoQubitRegister.singlet()
        e = bell_correlation_register(reg, 0.0, math.pi / 4)
        self.assertAlmostEqual(e, bell_correlation_qm(0.0, math.pi / 4), places=5)

    def test_chsh_quantum_violation(self):
        reg = TwoQubitRegister.singlet()
        s = chsh_s_register(reg)
        self.assertAlmostEqual(s, chsh_s_quantum(), places=5)
        self.assertGreater(abs(s), 2.0)

    def test_circuit_prep_matches_singlet(self):
        circ = TwoQubitRegister.zeros().prepare_bell_singlet_circuit()
        e_circ = bell_correlation_register(circ, 0.0, math.pi / 4)
        e_ref = bell_correlation_register(TwoQubitRegister.singlet(), 0.0, math.pi / 4)
        self.assertAlmostEqual(e_circ, e_ref, places=4)

    def test_aethos_fill_contract(self):
        self.assertAlmostEqual(
            aethos_correlation_from_fill(0.0, math.pi / 4, 1.0),
            bell_correlation_qm(0.0, math.pi / 4),
            places=6,
        )
        self.assertAlmostEqual(chsh_s_aethos(1.0), chsh_s_quantum(), places=5)

    def test_dephase_reduces_entangled_weight(self):
        reg = TwoQubitRegister.singlet()
        w_full = entangled_weight(reg)
        aethos_dephase(reg, 0.0, 0.0)
        self.assertAlmostEqual(entangled_weight(reg), 0.0, places=5)
        reg2 = TwoQubitRegister.singlet()
        aethos_dephase(reg2, 1.0, 1.0)
        self.assertAlmostEqual(entangled_weight(reg2), w_full, places=5)

    def test_bell_with_fill_matches_contract(self):
        reg = TwoQubitRegister.singlet()
        e = bell_correlation_with_fill(reg, 0.0, math.pi / 4, 0.5, 1.0)
        self.assertAlmostEqual(e, aethos_correlation_from_fill(0.0, math.pi / 4, 0.5), places=5)

    def test_sample_bitstring_singlet(self):
        reg = TwoQubitRegister.singlet()
        hits = {(0, 1), (1, 0)}
        for _ in range(50):
            self.assertIn(reg.sample_bitstring(), hits)

    def test_demo_runs(self):
        self.assertIn("CHSH", demo_bell_chsh())


class TestOceanQuantumBridge(unittest.TestCase):
    def test_session_from_ocean_after_run(self):
        g = OceanGraph.from_blob(
            ElectronBlob(density=0.5, coupling=0.5),
            count=12,
            origin_max_depth=2,
            max_neighbors=3,
        )
        g.run(300, 1e-7)
        pairs = g.observable_pairs(min_phi=0.0, min_coherence=0.0)
        self.assertGreater(len(pairs), 0)
        p = pairs[0]
        sess, reg = session_from_ocean(g, p.a, p.b)
        self.assertEqual(sess.phi_ab, p.phi)
        self.assertEqual(sess.coherence, p.coherence)
        reg2 = register_from_ocean_pair(g, p)
        e1 = bell_correlation_register(reg, 0.0, math.pi / 4)
        e2 = bell_correlation_register(reg2, 0.0, math.pi / 4)
        self.assertAlmostEqual(e1, e2, places=6)


if __name__ == "__main__":
    unittest.main()
