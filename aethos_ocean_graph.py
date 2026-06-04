"""
Ocean-graph simulator — many-site coupling without full statevector.

Active nodes (aethos_active) sit on origins; ocean edges carry P11-3 fill phi_AB
and Sec-6 coherence C_ab. Coordinates are lazy-evaluated per transgression n.

Extensions:
  - Origin-tree links (parent / child / sibling dimensions)
  - Meet-boosted Gamma_fill when sites collide at same coord
  - Vectorized edge updates (NumPy batch; optional CuPy if installed)
  - Trace export for visualization
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator, Sequence

from aethos_active import ActiveNetwork100
from aethos_blob import ElectronBlob
from aethos_physics import (
    bell_correlation_phi_fill,
    coherence_steady_state,
    gamma_break_rate,
    gamma_form_rate,
    phi_ab_derivative,
    phi_ab_steady_state,
)

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NUMPY = False

try:
    import cupy as cp  # type: ignore

    _HAS_CUPY = True
except ImportError:  # pragma: no cover
    cp = None  # type: ignore
    _HAS_CUPY = False


class EdgeKind(str, Enum):
    KNN = "knn"
    ORIGIN = "origin"  # parent / child / sibling room link
    MEET = "meet"  # same coordinate collision


def _dist3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def origin_link_kind(path_a: str, path_b: str) -> str | None:
    """Classify origin-tree relation between two origin paths."""
    if path_a == path_b:
        return "same"
    if path_b.startswith(path_a + ".") or path_a.startswith(path_b + "."):
        return "parent_child"
    pa = path_a.rsplit(".", 1)
    pb = path_b.rsplit(".", 1)
    if len(pa) == 2 and len(pb) == 2 and pa[0] == pb[0]:
        return "sibling"
    return None


def related_origin_paths(origin_path: str, all_paths: Sequence[str]) -> set[str]:
    """Origins linked by tree (parent, children, siblings) — not same room."""
    related: set[str] = set()
    for p in all_paths:
        if p == origin_path:
            continue
        kind = origin_link_kind(origin_path, p)
        if kind in ("parent_child", "sibling"):
            related.add(p)
    return related


@dataclass
class OceanSite:
    """One pumped site (coin) on the graph."""

    node_id: int
    origin_path: str
    coord: tuple[float, float, float] = (0.0, 0.0, 0.0)
    coin_phase: float = 0.0
    sigma_z: float = 1.0
    chain_species: str = "primes"


@dataclass
class OceanEdge:
    """Ocean link between two sites — fill phi and pair coherence."""

    a: int
    b: int
    distance_u: float = 0.0
    phi: float = 0.0
    coherence: float = 0.0
    gamma_fill: float = 1.0
    gamma_snap: float = 0.5
    phi_env: float = 0.0
    kind: EdgeKind = EdgeKind.KNN
    meet_boost: float = 1.0  # applied to gamma_fill when endpoints meet


@dataclass(frozen=True)
class OceanTraceRow:
    time_s: float
    transgression_n: int
    mean_phi: float
    mean_coherence: float
    meet_classes: int
    edge_count: int
    origin_edges: int


@dataclass(frozen=True)
class ObservablePair:
    """Entanglement-ready pair observable after ocean dynamics."""

    a: int
    b: int
    phi: float
    coherence: float
    bell_e: float
    edge_kind: str
    species_a: str
    species_b: str
    distance_u: float


@dataclass
class OceanGraph:
    """
    Graph of active nodes + ocean edges.

    Lazy rule: refresh coordinates only when transgression n changes.
    """

    network: ActiveNetwork100
    sites: dict[int, OceanSite] = field(default_factory=dict)
    edges: list[OceanEdge] = field(default_factory=list)
    transgression_n: int = 0
    time_s: float = 0.0
    ell_c: float = 1.0
    meet_fill_boost: float = 5.0
    meet_coord_tol: float = 1e-6
    use_origin_edges: bool = True
    use_vectorized: bool = True
    use_gpu: bool = False
    _coords_n: int = -1
    _xp: object = field(default=None, repr=False)  # numpy or cupy module

    def __post_init__(self) -> None:
        if self.use_gpu and _HAS_CUPY:
            self._xp = cp
        elif _HAS_NUMPY:
            self._xp = np
        else:
            self._xp = None
            self.use_vectorized = False

    @classmethod
    def from_network(
        cls,
        network: ActiveNetwork100,
        *,
        max_neighbors: int = 6,
        same_origin_bonus: float = 0.5,
        n: int = 0,
        gamma_fill: float = 1.0,
        gamma_snap: float = 0.5,
        use_origin_edges: bool = True,
        meet_fill_boost: float = 5.0,
        use_vectorized: bool = True,
        use_gpu: bool = False,
    ) -> OceanGraph:
        """Build sites from active nodes; wire k-NN + origin-tree ocean edges."""
        g = cls(
            network=network,
            transgression_n=n,
            use_origin_edges=use_origin_edges,
            meet_fill_boost=meet_fill_boost,
            use_vectorized=use_vectorized,
            use_gpu=use_gpu,
        )
        for node in network.nodes:
            g.sites[node.node_id] = OceanSite(
                node_id=node.node_id,
                origin_path=node.origin_path,
                chain_species=node.chain_species.value,
            )
        g.rebuild_edges(max_neighbors, same_origin_bonus)
        for e in g.edges:
            e.gamma_fill = gamma_fill
            e.gamma_snap = gamma_snap
            e.phi = phi_ab_steady_state(gamma_fill, gamma_snap) * 0.1
        g._apply_meet_boosts()
        return g

    @classmethod
    def from_blob(
        cls,
        blob: ElectronBlob,
        *,
        count: int = 100,
        origin_max_depth: int = 3,
        max_neighbors: int = 6,
        same_origin_bonus: float = 0.5,
        n: int = 0,
        gamma_fill: float = 1.0,
        gamma_snap: float = 0.5,
        use_origin_edges: bool = True,
        meet_fill_boost: float = 5.0,
        use_vectorized: bool = True,
        use_gpu: bool = False,
    ) -> OceanGraph:
        """Material blob selects per-node anchor set, then builds ocean graph."""
        net = ActiveNetwork100.bootstrap_from_blob(
            blob, count=count, origin_max_depth=origin_max_depth
        )
        return cls.from_network(
            net,
            max_neighbors=max_neighbors,
            same_origin_bonus=same_origin_bonus,
            n=n,
            gamma_fill=gamma_fill,
            gamma_snap=gamma_snap,
            use_origin_edges=use_origin_edges,
            meet_fill_boost=meet_fill_boost,
            use_vectorized=use_vectorized,
            use_gpu=use_gpu,
        )

    def apply_blob(self, blob: ElectronBlob, *, max_neighbors: int = 6) -> None:
        """Reassign anchor chains from blob; rebuild ocean wiring."""
        self.network.apply_blob_chains(blob)
        for node in self.network.nodes:
            site = self.sites.get(node.node_id)
            if site is None:
                self.sites[node.node_id] = OceanSite(
                    node_id=node.node_id,
                    origin_path=node.origin_path,
                    chain_species=node.chain_species.value,
                )
            else:
                site.chain_species = node.chain_species.value
        self._coords_n = -1
        self.rebuild_edges(max_neighbors=max_neighbors)
        self._apply_meet_boosts()

    def refresh_coordinates(self) -> None:
        """Lazy eval: compute all site coords at current transgression n."""
        if self._coords_n == self.transgression_n:
            return
        for node in self.network.nodes:
            origin = self.network.origin_for(node)
            c = node.address(self.transgression_n, origin.coord)
            self.sites[node.node_id].coord = c
        self._coords_n = self.transgression_n

    def _edge_key(self, a: int, b: int) -> tuple[int, int]:
        return (min(a, b), max(a, b))

    def _add_edge(
        self,
        a: int,
        b: int,
        *,
        seen: set[tuple[int, int]],
        kind: EdgeKind,
    ) -> None:
        key = self._edge_key(a, b)
        if key in seen:
            return
        seen.add(key)
        sa, sb = self.sites[a], self.sites[b]
        d = _dist3(sa.coord, sb.coord)
        self.edges.append(
            OceanEdge(a=key[0], b=key[1], distance_u=max(d, 1e-30), kind=kind)
        )

    def _build_knn_edges(self, max_neighbors: int, same_origin_bonus: float) -> None:
        ids = sorted(self.sites)
        seen: set[tuple[int, int]] = set()
        knn: list[OceanEdge] = []
        self.edges.clear()
        for i in ids:
            si = self.sites[i]
            scored: list[tuple[float, int]] = []
            for j in ids:
                if i == j:
                    continue
                sj = self.sites[j]
                d = _dist3(si.coord, sj.coord)
                if si.origin_path == sj.origin_path:
                    d *= same_origin_bonus
                scored.append((d, j))
            scored.sort(key=lambda x: x[0])
            for _, j in scored[:max_neighbors]:
                key = self._edge_key(i, j)
                if key in seen:
                    continue
                seen.add(key)
                d_phys = _dist3(si.coord, self.sites[j].coord)
                knn.append(
                    OceanEdge(
                        a=key[0],
                        b=key[1],
                        distance_u=max(d_phys, 1e-30),
                        kind=EdgeKind.KNN,
                    )
                )
        self.edges.extend(knn)

    def _build_origin_tree_edges(self, seen: set[tuple[int, int]]) -> None:
        """Link each site to nearest site in related origin rooms (D1/D2/D3 tree)."""
        if not self.use_origin_edges or not self.network._origin_index:
            return
        all_paths = list(self.network._origin_index.keys())
        by_origin: dict[str, list[int]] = {}
        for sid, s in self.sites.items():
            by_origin.setdefault(s.origin_path, []).append(sid)

        for sid, site in self.sites.items():
            for rel_path in related_origin_paths(site.origin_path, all_paths):
                candidates = by_origin.get(rel_path, [])
                if not candidates:
                    continue
                best = min(candidates, key=lambda j: _dist3(site.coord, self.sites[j].coord))
                self._add_edge(sid, best, seen=seen, kind=EdgeKind.ORIGIN)

    def _build_meet_edges(self, seen: set[tuple[int, int]], tol: float) -> None:
        """Explicit edges between sites at the same coordinate (wing meet)."""
        meets = self.sites_at_same_coord(tol=tol)
        for group in meets:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    self._add_edge(group[i], group[j], seen=seen, kind=EdgeKind.MEET)

    def _calibrate_ell_c(self) -> None:
        if not self.edges:
            self.ell_c = 1.0
            return
        dists = sorted(e.distance_u for e in self.edges)
        self.ell_c = max(dists[len(dists) // 2] * 1.5, 1e-9)

    def _apply_meet_boosts(self) -> None:
        """Boost Gamma_fill on edges whose endpoints share a meet class."""
        meets = self.sites_at_same_coord(tol=self.meet_coord_tol)
        meet_set: set[int] = set()
        for group in meets:
            meet_set.update(group)
        pair_meet: set[tuple[int, int]] = set()
        for group in meets:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    pair_meet.add(self._edge_key(group[i], group[j]))

        tol = self.meet_coord_tol
        for e in self.edges:
            sa, sb = self.sites[e.a], self.sites[e.b]
            at_meet = (
                self._edge_key(e.a, e.b) in pair_meet
                or _dist3(sa.coord, sb.coord) <= tol
                or (e.a in meet_set and e.b in meet_set and sa.coord == sb.coord)
            )
            if e.kind == EdgeKind.MEET or at_meet:
                e.meet_boost = self.meet_fill_boost
            else:
                e.meet_boost = 1.0

    def set_transgression(self, n: int) -> None:
        self.transgression_n = n
        self._coords_n = -1

    def rebuild_edges(self, max_neighbors: int = 6, same_origin_bonus: float = 0.5) -> None:
        """Re-wire k-NN, origin-tree, and meet edges; recalibrate ell_c."""
        self.refresh_coordinates()
        seen: set[tuple[int, int]] = set()
        self._build_knn_edges(max_neighbors, same_origin_bonus)
        seen = {self._edge_key(e.a, e.b) for e in self.edges}
        self._build_origin_tree_edges(seen)
        self._build_meet_edges(seen, self.meet_coord_tol)
        self._calibrate_ell_c()
        self._apply_meet_boosts()

    def _effective_gamma_fill(self, edge: OceanEdge) -> float:
        return edge.gamma_fill * edge.meet_boost

    def _edge_rates(self, edge: OceanEdge) -> tuple[float, float]:
        gf = gamma_form_rate(
            edge.distance_u,
            phi_ab=edge.phi,
            ell_c=self.ell_c,
        )
        gb = gamma_break_rate(edge.phi_env, kappa=0.0)
        return gf, gb

    def _step_loop(self, dt: float) -> None:
        for e in self.edges:
            gf, gb = self._edge_rates(e)
            dp = phi_ab_derivative(
                e.phi,
                gamma_fill=self._effective_gamma_fill(e),
                gamma_snap=e.gamma_snap,
                gamma_break=gb,
            )
            dc = gf * (1.0 - e.coherence) - gb * e.coherence
            e.phi = max(0.0, min(1.0, e.phi + dp * dt))
            e.coherence = max(0.0, min(1.0, e.coherence + dc * dt))

    def _step_vectorized(self, dt: float) -> None:
        xp = self._xp
        if xp is None or not self.edges:
            self._step_loop(dt)
            return

        from aethos_physics import k_lock_from_geometry

        dist = xp.array([e.distance_u for e in self.edges], dtype=xp.float64)
        phi = xp.array([e.phi for e in self.edges], dtype=xp.float64)
        coh = xp.array([e.coherence for e in self.edges], dtype=xp.float64)
        gfill = xp.array([self._effective_gamma_fill(e) for e in self.edges], dtype=xp.float64)
        gsnap = xp.array([e.gamma_snap for e in self.edges], dtype=xp.float64)

        ell = float(self.ell_c)
        kl = k_lock_from_geometry()
        j_ab = xp.exp(-dist / ell)
        gf = kl * j_ab * phi
        gb = xp.zeros_like(phi)

        dphi = gfill * (1.0 - phi) - gsnap * phi - gb
        dcoh = gf * (1.0 - coh) - gb * coh

        phi = xp.clip(phi + dphi * dt, 0.0, 1.0)
        coh = xp.clip(coh + dcoh * dt, 0.0, 1.0)

        if self.use_gpu and _HAS_CUPY and xp is cp:
            phi_host = cp.asnumpy(phi)
            coh_host = cp.asnumpy(coh)
        else:
            phi_host = np.asarray(phi)
            coh_host = np.asarray(coh)

        for i, e in enumerate(self.edges):
            e.phi = float(phi_host[i])
            e.coherence = float(coh_host[i])

    def step(self, dt: float) -> None:
        """One ocean timestep."""
        self.refresh_coordinates()
        if dt <= 0:
            return
        if self.use_vectorized and self._xp is not None:
            self._step_vectorized(dt)
        else:
            self._step_loop(dt)
        self.time_s += dt

    def run(self, steps: int, dt: float) -> None:
        for _ in range(steps):
            self.step(dt)

    def snapshot_trace(self) -> OceanTraceRow:
        origin_n = sum(1 for e in self.edges if e.kind == EdgeKind.ORIGIN)
        return OceanTraceRow(
            time_s=self.time_s,
            transgression_n=self.transgression_n,
            mean_phi=self.mean_phi(),
            mean_coherence=self.mean_coherence(),
            meet_classes=len(self.sites_at_same_coord()),
            edge_count=len(self.edges),
            origin_edges=origin_n,
        )

    def run_with_trace(
        self,
        steps: int,
        dt: float,
        *,
        sample_every: int = 1,
        n_values: Sequence[int] | None = None,
        rebuild_on_n: bool = True,
        max_neighbors: int = 6,
    ) -> list[OceanTraceRow]:
        """
        Run simulation and record mean_phi, mean_C, meets, edge counts.

        If n_values is provided, sets transgression n each segment and optionally rebuilds edges.
        """
        trace: list[OceanTraceRow] = []
        trace.append(self.snapshot_trace())

        if n_values:
            per_n = max(1, steps // max(len(n_values), 1))
            for n_val in n_values:
                self.set_transgression(int(n_val))
                if rebuild_on_n:
                    self.rebuild_edges(max_neighbors=max_neighbors)
                trace.append(self.snapshot_trace())
                for i in range(per_n):
                    self.step(dt)
                    if (i + 1) % sample_every == 0:
                        trace.append(self.snapshot_trace())
        else:
            for i in range(steps):
                self.step(dt)
                if (i + 1) % sample_every == 0:
                    trace.append(self.snapshot_trace())
        return trace

    def export_trace_csv(self, trace: Sequence[OceanTraceRow], path: str | Path) -> Path:
        out = Path(path)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "time_s",
                    "transgression_n",
                    "mean_phi",
                    "mean_coherence",
                    "meet_classes",
                    "edge_count",
                    "origin_edges",
                ]
            )
            for row in trace:
                w.writerow(
                    [
                        row.time_s,
                        row.transgression_n,
                        row.mean_phi,
                        row.mean_coherence,
                        row.meet_classes,
                        row.edge_count,
                        row.origin_edges,
                    ]
                )
        return out

    def mean_phi(self) -> float:
        if not self.edges:
            return 0.0
        return sum(e.phi for e in self.edges) / len(self.edges)

    def mean_coherence(self) -> float:
        if not self.edges:
            return 0.0
        return sum(e.coherence for e in self.edges) / len(self.edges)

    def edge_between(self, a: int, b: int) -> OceanEdge | None:
        key = self._edge_key(a, b)
        for e in self.edges:
            if (e.a, e.b) == key:
                return e
        return None

    def bell_e(self, a: int, b: int, angle_a: float, angle_b: float) -> float:
        e = self.edge_between(a, b)
        if e is None:
            return 0.0
        return bell_correlation_phi_fill(angle_a, angle_b, e.phi)

    def sites_at_same_coord(self, tol: float = 1e-6) -> list[list[int]]:
        buckets: dict[tuple[float, float, float], list[int]] = {}
        for sid, s in self.sites.items():
            key = (
                round(s.coord[0] / tol) * tol,
                round(s.coord[1] / tol) * tol,
                round(s.coord[2] / tol) * tol,
            )
            buckets.setdefault(key, []).append(sid)
        return [v for v in buckets.values() if len(v) > 1]

    def count_edges_by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.edges:
            counts[e.kind.value] = counts.get(e.kind.value, 0) + 1
        return counts

    def observable_pairs(
        self,
        *,
        min_phi: float = 0.0,
        min_coherence: float = 0.0,
        angle_a: float = 0.0,
        angle_b: float | None = None,
    ) -> list[ObservablePair]:
        """
        Pairs whose ocean link exceeds thresholds — lab-observable entanglement candidates.

        bell_e uses E = -phi_AB cos(a-b) at given angles (default pi/4, pi/4).
        """
        if angle_b is None:
            angle_b = math.pi / 4.0
        out: list[ObservablePair] = []
        for e in self.edges:
            if e.phi < min_phi or e.coherence < min_coherence:
                continue
            sa = self.sites[e.a]
            sb = self.sites[e.b]
            be = bell_correlation_phi_fill(angle_a, angle_b, e.phi)
            out.append(
                ObservablePair(
                    a=e.a,
                    b=e.b,
                    phi=e.phi,
                    coherence=e.coherence,
                    bell_e=be,
                    edge_kind=e.kind.value,
                    species_a=sa.chain_species,
                    species_b=sb.chain_species,
                    distance_u=e.distance_u,
                )
            )
        out.sort(key=lambda p: (-p.coherence, -p.phi))
        return out

    def species_pair_summary(self, pairs: Sequence[ObservablePair]) -> dict[str, int]:
        """Tally observable pairs by (species_a, species_b) unordered."""
        tallies: dict[str, int] = {}
        for p in pairs:
            key = tuple(sorted((p.species_a, p.species_b)))
            label = f"{key[0]}|{key[1]}"
            tallies[label] = tallies.get(label, 0) + 1
        return tallies

    def quantum_session(self, node_a: int, node_b: int) -> tuple[object, object]:
        """2-qubit register + session from ocean edge (aethos_quantum)."""
        from aethos_quantum import session_from_ocean

        return session_from_ocean(self, node_a, node_b)

    def report(self) -> str:
        meets = self.sites_at_same_coord()
        kinds = self.count_edges_by_kind()
        backend = "cupy" if self.use_gpu and _HAS_CUPY else ("numpy" if self._xp is not None else "python")
        lines = [
            "AETHOS ocean graph simulator",
            "=" * 56,
            f"sites                  = {len(self.sites)}",
            f"edges                  = {len(self.edges)}  {kinds}",
            f"transgression n        = {self.transgression_n}",
            f"sim time               = {self.time_s:.4e} s",
            f"mean phi_AB            = {self.mean_phi():.4f}",
            f"mean coherence C       = {self.mean_coherence():.4f}",
            f"meet classes (n now)   = {len(meets)}",
            f"ell_c (graph units)    = {self.ell_c:.3e}",
            f"step backend           = {backend}",
            "",
        ]
        if self.edges:
            boosted = sum(1 for e in self.edges if e.meet_boost > 1.0)
            pairs = self.observable_pairs(min_phi=0.01, min_coherence=0.01)
            lines.append(f"meet-boosted edges     = {boosted}  (boost x{self.meet_fill_boost:g})")
            lines.append(f"observable pairs       = {len(pairs)}  (phi,C >= 0.01)")
            if pairs:
                top = pairs[0]
                lines.append(
                    f"  top pair ({top.a},{top.b}) [{top.edge_kind}] "
                    f"phi={top.phi:.3f} C={top.coherence:.3f} "
                    f"species={top.species_a}/{top.species_b}"
                )
            e0 = self.edges[0]
            gf, gb = self._edge_rates(e0)
            cs = coherence_steady_state(gf, gb)
            lines.append(
                f"sample edge ({e0.a},{e0.b}) [{e0.kind.value}]: "
                f"d={e0.distance_u:.3e}  phi={e0.phi:.3f}  C={e0.coherence:.3f}  C_*={cs:.3f}"
            )
        return "\n".join(lines)


def demo_blob_contrast(count: int = 24, steps: int = 400, dt: float = 1e-7) -> None:
    """Compare uniform primes vs dense blob — different observable pairs."""
    low = ElectronBlob(density=0.1, coupling=0.1)
    high = ElectronBlob(density=0.9, coupling=0.8)
    g_low = OceanGraph.from_blob(low, count=count, origin_max_depth=2, max_neighbors=4)
    g_high = OceanGraph.from_blob(high, count=count, origin_max_depth=2, max_neighbors=4)
    g_low.run(steps, dt)
    g_high.run(steps, dt)
    p_low = g_low.observable_pairs(min_phi=0.05, min_coherence=0.05)
    p_high = g_high.observable_pairs(min_phi=0.05, min_coherence=0.05)
    print("=== BLOB vs anchor set (C6) ===")
    print(f"low density blob:  {len(p_low)} observable pairs")
    print(f"  species mix: {g_low.species_pair_summary(p_low)}")
    print(f"high density blob: {len(p_high)} observable pairs")
    print(f"  species mix: {g_high.species_pair_summary(p_high)}")


def demo_ocean_graph(count: int = 20, steps: int = 200, dt: float = 1e-9) -> OceanGraph:
    net = ActiveNetwork100.bootstrap(count=count, origin_max_depth=3)
    g = OceanGraph.from_network(net, max_neighbors=4, n=0, use_origin_edges=True)
    trace = g.run_with_trace(steps, dt, sample_every=50, n_values=[0, 3, 7, 11], rebuild_on_n=True)
    return g


if __name__ == "__main__":
    demo_blob_contrast()
    print()
    g = demo_ocean_graph()
    print(g.report())
    full_trace = g.run_with_trace(50, 1e-8, sample_every=10, n_values=[13], rebuild_on_n=True)
    csv_path = g.export_trace_csv(full_trace, "ocean_trace_demo.csv")
    print(f"\nTrace exported: {csv_path}  ({len(full_trace)} rows)")
    pairs = g.observable_pairs(min_phi=0.01, min_coherence=0.01)
    print(f"Observable pairs: {len(pairs)}")
    if pairs:
        p = pairs[0]
        print(f"  top: ({p.a},{p.b}) bell_E={p.bell_e:.4f} species={p.species_a}/{p.species_b}")
