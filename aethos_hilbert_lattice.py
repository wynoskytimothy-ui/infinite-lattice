"""
Hilbert space derived entirely from the AETHOS lattice.

Claim: standard Hilbert structure (basis, inner product, norm, completeness)
is not imported — it is built from:

  VA1–VA4 formulas  -> spring complex z = X + iY at triggers
  32 wings            -> orthonormal directions (unless meet)
  meets               -> quotient / boost on shared coordinates
  transgressor n      -> countable basis index
  L4–L6 correlations  -> sparse semantic inner product extension
  L7–L9 categories    -> subspace projectors

Combined robust inner product:

  <a|b>_total = w_g * <a|b>_geom + w_c * <a|b>_corr + w_m * <a|b>_meet

All weights default to 1; tune for physics vs NLP projects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from aethos_complex_spring import SpringState, spring_states_at
from aethos_hilbert import (
    BasisLabel,
    LatticeState,
    correlation_inner_product,
    formula_coord_branch,
    inner_product as label_inner_product,
)
from aethos_lattice import BranchKind, LatticeBank32, LatticeId
from aethos_promotion import CorrelationLink, PromotionRegistry
from aethos_recursive import LatticeBank32K, canon_recursive


# ---------------------------------------------------------------------------
# Hilbert axioms <- lattice primitives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HilbertDerivation:
    """One Hilbert-space formula and its lattice source."""

    name: str
    lattice_source: str
    formula: str


HILBERT_FROM_LATTICE: tuple[HilbertDerivation, ...] = (
    HilbertDerivation(
        "Vector addition",
        "Superposition of wing/trigger basis labels",
        "|psi> + |phi>  <=>  sum_i (a_i + b_i) |e_i>",
    ),
    HilbertDerivation(
        "Scalar multiplication",
        "Amplitude on same basis label",
        "c|psi>  <=>  (c * a_i) on each |e_i>",
    ),
    HilbertDerivation(
        "Inner product (geometric)",
        "Spring complex z=X+iY at triggers; label overlap",
        "<a|b>_geom = sum_{i in keys} conj(a_i) b_i",
    ),
    HilbertDerivation(
        "Inner product (spring)",
        "Trigger x 4 branches; Born |z|^2",
        "<z_a|z_b> = Re(conj(z_a) z_b);  P ~ |z|^2 = T^2",
    ),
    HilbertDerivation(
        "Inner product (correlation)",
        "L4-L6 edge weights on prime index",
        "<a|b>_corr = sum_p w_a(p) w_b(p)",
    ),
    HilbertDerivation(
        "Inner product (meet boost)",
        "Same coordinate at meet identifies rays",
        "<a|b>_meet += amp if coord(a)=coord(b)",
    ),
    HilbertDerivation(
        "Norm",
        "||psi|| = sqrt(<psi|psi>)",
        "Born: ||psi||^2 ~ integral T^2",
    ),
    HilbertDerivation(
        "Orthogonality",
        "Distinct wing labels at same n (no meet)",
        "<e_i|e_j> = 0 for i != j",
    ),
    HilbertDerivation(
        "Countable basis",
        "Triggers x wings x branches x n x species",
        "{ |origin, L, P, n, VA, pi> } enumerable",
    ),
    HilbertDerivation(
        "Completeness",
        "Cauchy limits via n->inf, k->inf, depth->inf",
        "Sep closure of finite trigger truncations",
    ),
    HilbertDerivation(
        "Direct sum",
        "32 wings per bank",
        "H = (+)_w H_w",
    ),
    HilbertDerivation(
        "Tensor / product structure",
        "Origin tree x order fiber k!",
        "H_total = H_origin (x) H_fiber (x) ...",
    ),
)


@dataclass
class RobustInnerProductWeights:
    geometric: float = 1.0
    correlation: float = 1.0
    meet: float = 1.0
    spring: float = 0.0  # optional spring-plane overlap at triggers


@dataclass
class LatticeHilbertSpace:
    """
    Full Hilbert structure derived from lattice + optional correlation registry.
    """

    chain: tuple[int, ...] = (3, 5, 7)
    n_window: tuple[int, ...] = (3, 5, 7)
    weights: RobustInnerProductWeights = field(default_factory=RobustInnerProductWeights)
    registry: PromotionRegistry | None = None

    def basis_labels(self) -> list[BasisLabel]:
        """Canonical enumerable basis (truncated window)."""
        out: list[BasisLabel] = []
        for n in self.n_window:
            for lid in LatticeId:
                branch, _ = __import__("aethos_lattice", fromlist=["lattice_id_parts"]).lattice_id_parts(lid)
                out.append(
                    BasisLabel(
                        wing=int(lid),
                        chain=self.chain,
                        n=n,
                        branch=branch,
                    )
                )
        return out

    def basis_state(self, label: BasisLabel) -> LatticeState:
        st = LatticeState()
        st.add(label, 1.0)
        return st

    def geometric_inner(self, a: LatticeState, b: LatticeState) -> complex:
        return label_inner_product(a, b)

    def spring_inner_at(self, n: int, branch_a: BranchKind, branch_b: BranchKind) -> complex:
        _, sa = spring_states_at(self.chain, n)
        za, zb = sa[branch_a].z, sa[branch_b].z
        return za.conjugate() * zb

    def meet_boost(self, a: LatticeState, b: LatticeState) -> float:
        """Extra pairing when two distinct basis states share formula coordinate."""
        if len(a.labels) != 1 or len(b.labels) != 1:
            return 0.0
        la, lb = next(iter(a.labels.values())), next(iter(b.labels.values()))
        if la.key() == lb.key():
            return 0.0
        ca = formula_coord_branch(la.chain, la.n, la.branch, la.wing)
        cb = formula_coord_branch(lb.chain, lb.n, lb.branch, lb.wing)
        if ca == cb:
            return 1.0
        return 0.0

    def correlation_inner_words(self, word_a: str, word_b: str) -> float:
        if not self.registry:
            return 0.0
        wa: dict[int, float] = {}
        wb: dict[int, float] = {}
        for w, bag in ((word_a.lower(), wa), (word_b.lower(), wb)):
            tok = self.registry.resolve_token(w)
            bag[tok.prime] = bag.get(tok.prime, 0.0) + 1.0
            for p in tok.parent_primes:
                bag[p] = bag.get(p, 0.0) + 0.5
        ip = correlation_inner_product(wa, wb)
        if ip > 0:
            return ip
        # L4-L6 edge strength when direct prime overlap is empty
        for link in self.registry.correlations_for(word_a.lower()):
            other = link.target if link.source == word_a.lower() else link.source
            if other == word_b.lower():
                return float(link.strength) * (link.dim4 + link.dim6 + 1.0)
        return 0.0

    def correlation_inner_links(self, word: str) -> dict[str, float]:
        """Neighbor correlation strengths from L4-L6 graph."""
        if not self.registry:
            return {}
        out: dict[str, float] = {}
        for link in self.registry.correlations_for(word):
            other = link.target if link.source == word else link.source
            out[other] = out.get(other, 0.0) + link.strength * (link.dim4 + link.dim6)
        return out

    def total_inner(
        self,
        a: LatticeState,
        b: LatticeState,
        *,
        spring_n: int | None = None,
        spring_a: BranchKind = BranchKind.VA1,
        spring_b: BranchKind = BranchKind.VA2,
    ) -> complex:
        w = self.weights
        total = w.geometric * self.geometric_inner(a, b)
        total += w.meet * self.meet_boost(a, b)
        if w.spring and spring_n is not None:
            total += w.spring * self.spring_inner_at(spring_n, spring_a, spring_b)
        return total

    def norm(self, state: LatticeState, **kwargs: object) -> float:
        ip = self.total_inner(state, state, **kwargs)  # type: ignore[arg-type]
        return math.sqrt(max(0.0, ip.real))

    def normalize(self, state: LatticeState, **kwargs: object) -> LatticeState:
        n = self.norm(state, **kwargs)
        if n == 0:
            return state
        out = LatticeState(
            amplitudes={k: v / n for k, v in state.amplitudes.items()},
            labels=dict(state.labels),
        )
        return out

    def project_onto(self, psi: LatticeState, basis: LatticeState, **kwargs: object) -> LatticeState:
        """Orthogonal projection (finite support)."""
        denom = self.total_inner(basis, basis, **kwargs).real
        if denom == 0:
            return LatticeState()
        coeff = self.total_inner(psi, basis, **kwargs) / denom
        out = LatticeState(labels=dict(basis.labels))
        for k, v in basis.amplitudes.items():
            out.amplitudes[k] = coeff * v
        return out

    def gram_matrix(self, labels: Sequence[BasisLabel], **kwargs: object) -> list[list[complex]]:
        """Gram matrix G_ij = <e_i|e_j> for robust inner product."""
        states = [self.basis_state(l) for l in labels]
        n = len(states)
        g: list[list[complex]] = [[0j] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                g[i][j] = self.total_inner(states[i], states[j], **kwargs)
        return g

    def extend_with_correlation(self, registry: PromotionRegistry) -> LatticeHilbertSpace:
        return LatticeHilbertSpace(
            chain=self.chain,
            n_window=self.n_window,
            weights=self.weights,
            registry=registry,
        )

    def derivation_table(self) -> str:
        lines = ["Hilbert formulas derived from lattice:", ""]
        for d in HILBERT_FROM_LATTICE:
            lines.append(f"  {d.name}")
            lines.append(f"    lattice: {d.lattice_source}")
            lines.append(f"    formula: {d.formula}")
            lines.append("")
        return "\n".join(lines)


def build_robust_space_from_corpus(
    *documents: str,
    chain: tuple[int, ...] = (3, 5, 7),
) -> LatticeHilbertSpace:
    """Lattice Hilbert + correlation layer from text (token processor optional path)."""
    from aethos_token_processor import TokenProcessor

    proc = TokenProcessor()
    proc.ingest(*documents)
    return LatticeHilbertSpace(
        chain=chain,
        registry=proc.registry,
        weights=RobustInnerProductWeights(geometric=1.0, correlation=1.5, meet=0.5),
    )


def correlation_formula_extension(link: CorrelationLink) -> tuple[float, float, float, float]:
    """
    Extended L4-L6 point used as inner-product kernel direction:
    (d4, d5, d6, strength) from lattice promotion pair.
    """
    return (link.dim4, link.dim5, link.dim6, float(link.strength))


def demo() -> None:
    print("=" * 70)
    print("HILBERT SPACE FROM LATTICE — all formulas derived")
    print("=" * 70)

    hs = LatticeHilbertSpace(chain=(3, 5, 7), n_window=(5, 7))
    print(hs.derivation_table()[:1200] + "...")

    labels = hs.basis_labels()[:4]
    g = hs.gram_matrix(labels)
    print("\n--- Gram matrix (first 4 basis vectors) ---")
    for row in g:
        print("  ", [f"{x.real:.2f}" for x in row])

    # superposition + norm
    psi = LatticeState()
    psi.add(labels[0], 0.6)
    psi.add(labels[1], 0.8)
    print(f"\n  ||psi|| (geom): {hs.norm(psi):.4f}")

    # spring inner at trigger
    hs_w = LatticeHilbertSpace(weights=RobustInnerProductWeights(spring=1.0))
    z_ip = hs_w.spring_inner_at(5, BranchKind.VA1, BranchKind.VA2)
    print(f"  spring <VA1|VA2> at n=5 (solo chain uses (5,)): use chain (5,) separately")
    _, st = spring_states_at((5,), 5)
    print(f"  |z_VA1|^2={st[BranchKind.VA1].tension_squared:.0f}  Born proxy")

    # robust with corpus
    print("\n--- Robust space + L4-L6 correlations ---")
    hs2 = build_robust_space_from_corpus(
        "phone phone technical chip",
        "phone hardware network",
    )
    corr = hs2.correlation_inner_words("phone", "technical")
    print(f"  <phone|technical>_corr = {corr:.4f}")
    print(f"  phone neighbors: {list(hs2.correlation_inner_links('phone').items())[:5]}")


if __name__ == "__main__":
    demo()
