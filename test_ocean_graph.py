"""Ocean-graph simulator tests. Run: python test_ocean_graph.py"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from aethos_active import ActiveNetwork100
from aethos_blob import ElectronBlob, assign_species
from aethos_ocean_graph import (
    EdgeKind,
    OceanGraph,
    demo_ocean_graph,
    origin_link_kind,
    related_origin_paths,
)
from aethos_physics import phi_ab_steady_state


class TestOriginLinks(unittest.TestCase):
    def test_parent_child_paths(self):
        self.assertEqual(origin_link_kind("O0", "O0.D1"), "parent_child")
        self.assertEqual(origin_link_kind("O0.D2", "O0"), "parent_child")

    def test_sibling_paths(self):
        self.assertEqual(origin_link_kind("O0.D1", "O0.D2"), "sibling")

    def test_related_origins(self):
        rel = related_origin_paths("O0", ["O0", "O0.D1", "O0.D2", "O0.D1.D1"])
        self.assertIn("O0.D1", rel)
        self.assertIn("O0.D2", rel)


class TestOceanGraph(unittest.TestCase):
    def test_builds_sites_and_edges(self):
        net = ActiveNetwork100.bootstrap(count=10, origin_max_depth=2)
        g = OceanGraph.from_network(net, max_neighbors=3, n=0)
        self.assertEqual(len(g.sites), 10)
        self.assertGreater(len(g.edges), 0)

    def test_origin_edges_added(self):
        net = ActiveNetwork100.bootstrap(count=15, origin_max_depth=3)
        g = OceanGraph.from_network(net, max_neighbors=3, use_origin_edges=True, n=0)
        kinds = g.count_edges_by_kind()
        self.assertGreater(kinds.get(EdgeKind.ORIGIN.value, 0), 0)

    def test_phi_increases_under_fill(self):
        net = ActiveNetwork100.bootstrap(count=8, origin_max_depth=1)
        g = OceanGraph.from_network(net, max_neighbors=3, gamma_fill=2.0, gamma_snap=0.1, n=0)
        p0 = g.mean_phi()
        g.run(500, 1e-8)
        self.assertGreater(g.mean_phi(), p0)

    def test_coherence_can_grow_with_graph_ell_c(self):
        g = demo_ocean_graph(count=12, steps=300, dt=1e-7)
        self.assertGreater(g.mean_coherence(), 0.0)

    def test_lazy_coord_refresh_on_n(self):
        net = ActiveNetwork100.bootstrap(count=5, origin_max_depth=2)
        g = OceanGraph.from_network(net, n=0)
        c0 = dict((i, s.coord) for i, s in g.sites.items())
        g.set_transgression(11)
        g.refresh_coordinates()
        c1 = dict((i, s.coord) for i, s in g.sites.items())
        self.assertNotEqual(c0, c1)

    def test_bell_e_scales_with_phi(self):
        net = ActiveNetwork100.bootstrap(count=6, origin_max_depth=1)
        g = OceanGraph.from_network(net, max_neighbors=2, n=0)
        e = g.edges[0]
        e.phi = 1.0
        val = g.bell_e(e.a, e.b, 0.0, math.pi / 4)
        self.assertAlmostEqual(val, -math.sqrt(2) / 2, places=4)
        e.phi = 0.0
        self.assertAlmostEqual(g.bell_e(e.a, e.b, 0.0, math.pi / 4), 0.0)

    def test_steady_phi_target_order(self):
        ss = phi_ab_steady_state(2.0, 0.5)
        self.assertAlmostEqual(ss, 2.0 / 2.5)

    def test_run_with_trace(self):
        net = ActiveNetwork100.bootstrap(count=8, origin_max_depth=2)
        g = OceanGraph.from_network(net, n=0)
        trace = g.run_with_trace(20, 1e-8, sample_every=5, n_values=[0, 5])
        self.assertGreaterEqual(len(trace), 2)
        self.assertGreater(trace[-1].edge_count, 0)

    def test_export_trace_csv(self):
        net = ActiveNetwork100.bootstrap(count=6, origin_max_depth=2)
        g = OceanGraph.from_network(net, n=0)
        trace = g.run_with_trace(10, 1e-8, sample_every=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = g.export_trace_csv(trace, Path(tmp) / "trace.csv")
            text = path.read_text(encoding="utf-8")
            self.assertIn("mean_phi", text)
            self.assertIn("origin_edges", text)

    def test_vectorized_matches_loop(self):
        net = ActiveNetwork100.bootstrap(count=10, origin_max_depth=2)
        g_loop = OceanGraph.from_network(net, n=0, use_vectorized=False)
        g_vec = OceanGraph.from_network(net, n=0, use_vectorized=True)
        g_loop.run(30, 1e-8)
        g_vec.run(30, 1e-8)
        self.assertAlmostEqual(g_loop.mean_phi(), g_vec.mean_phi(), places=4)
        self.assertAlmostEqual(g_loop.mean_coherence(), g_vec.mean_coherence(), places=4)


class TestBlobAnchorSets(unittest.TestCase):
    def test_assign_species_deterministic(self):
        blob = ElectronBlob(density=0.6, coupling=0.4)
        a = assign_species(blob, 3, origin_depth=1)
        b = assign_species(blob, 3, origin_depth=1)
        self.assertEqual(a, b)

    def test_bootstrap_from_blob_mixed_species(self):
        blob = ElectronBlob(density=0.55, coupling=0.9)
        net = ActiveNetwork100.bootstrap_from_blob(
            blob, count=30, origin_max_depth=3
        )
        species = {n.chain_species for n in net.nodes}
        self.assertGreater(len(species), 1)

    def test_apply_blob_changes_chains(self):
        blob_a = ElectronBlob(density=0.1, coupling=0.1)
        blob_b = ElectronBlob(density=0.95, coupling=0.9)
        net = ActiveNetwork100.bootstrap_from_blob(blob_a, count=12, origin_max_depth=2)
        chains_before = tuple(n.chain for n in net.nodes)
        net.apply_blob_chains(blob_b)
        chains_after = tuple(n.chain for n in net.nodes)
        self.assertNotEqual(chains_before, chains_after)

    def test_from_blob_sets_site_species(self):
        blob = ElectronBlob(density=0.55, coupling=0.9)
        g = OceanGraph.from_blob(blob, count=25, origin_max_depth=3, max_neighbors=3)
        species_on_sites = {s.chain_species for s in g.sites.values()}
        self.assertGreater(len(species_on_sites), 1)

    def test_different_blobs_different_observable_pairs(self):
        low = ElectronBlob(density=0.05, coupling=0.05)
        high = ElectronBlob(density=0.92, coupling=0.85)
        g_low = OceanGraph.from_blob(low, count=16, origin_max_depth=2, max_neighbors=4)
        g_high = OceanGraph.from_blob(high, count=16, origin_max_depth=2, max_neighbors=4)
        g_low.run(400, 1e-7)
        g_high.run(400, 1e-7)
        species_low = {s.chain_species for s in g_low.sites.values()}
        species_high = {s.chain_species for s in g_high.sites.values()}
        self.assertNotEqual(species_low, species_high)
        p_low = g_low.observable_pairs(min_phi=0.0, min_coherence=0.0)
        p_high = g_high.observable_pairs(min_phi=0.0, min_coherence=0.0)
        sig_low = g_low.species_pair_summary(p_low)
        sig_high = g_high.species_pair_summary(p_high)
        self.assertNotEqual(sig_low, sig_high)

    def test_observable_pairs_respects_thresholds(self):
        g = demo_ocean_graph(count=10, steps=200, dt=1e-7)
        all_pairs = g.observable_pairs(min_phi=0.0, min_coherence=0.0)
        strict = g.observable_pairs(min_phi=0.5, min_coherence=0.5)
        self.assertGreaterEqual(len(all_pairs), len(strict))
        for p in strict:
            self.assertGreaterEqual(p.phi, 0.5)
            self.assertGreaterEqual(p.coherence, 0.5)

    def test_apply_blob_mid_sim(self):
        g = OceanGraph.from_blob(
            ElectronBlob(density=0.1, coupling=0.1),
            count=12,
            origin_max_depth=2,
            max_neighbors=3,
        )
        g.run(100, 1e-7)
        before = len(g.observable_pairs(min_phi=0.0, min_coherence=0.0))
        g.apply_blob(ElectronBlob(density=0.9, coupling=0.9))
        g.run(100, 1e-7)
        after = len(g.observable_pairs(min_phi=0.0, min_coherence=0.0))
        self.assertIsInstance(before, int)
        self.assertIsInstance(after, int)


if __name__ == "__main__":
    unittest.main()
