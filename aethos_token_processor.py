"""
AETHOS token processor — L1–L9 semantics on top of the lattice core.

This module is intentionally separate from aethos_core:
  - Core: 32 wings, formulas, origins, countable species (project-agnostic)
  - Tokens: promotion, intersection policy, natural clusters, overlays

Import aethos_core alone for physics / codec / custom lattice projects.
Import TokenProcessor when you need vocabulary and reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from aethos_core import AethosLatticeCore
from aethos_crossmeaning import MarkovCrossLattice, SemanticStack
from aethos_frequency import FrequencyProfile
from aethos_natural import NaturalReader
from aethos_overlay import SemanticOverlay, overlay_for_word
from aethos_persist import load_reader, save_reader
from aethos_promotion import LatticeTier, PromotionRegistry, letter_chain
from aethos_token_savings import TokenSavingsReport, report_from_reader


def semantic_chain_for_word(registry: PromotionRegistry, word: str) -> tuple[int, ...]:
    """Anchor chain used for L3 lattice address (token layer)."""
    tok = registry.resolve_token(word)
    if tok.intersection_only:
        return tuple(sorted(set(tok.parent_primes)))
    return tuple(sorted(set(tok.parent_primes + (tok.prime,))))


@dataclass
class TokenProcessor:
    """
    Corpus ingestion, promotion, clusters, and semantic overlay.
    Uses AethosLatticeCore for all formula coordinates.
    """

    core: AethosLatticeCore = field(default_factory=AethosLatticeCore)
    reader: NaturalReader | None = None
    rebuild_every: int = 3

    def __post_init__(self) -> None:
        if self.reader is None:
            self.reader = NaturalReader(rebuild_every=self.rebuild_every)

    @property
    def registry(self) -> PromotionRegistry:
        return self.reader.registry

    def ingest(self, *documents: str) -> None:
        self.reader.read(*documents)

    def resolve(self, word: str, context: Iterable[str] | None = None) -> dict[str, object]:
        ctx = list(context) if context is not None else None
        cid, score = self.reader.infer_cluster(word, ctx)
        w = word.lower()
        if (LatticeTier.L3_WORD, w) in self.registry.promoted:
            tier = "dedicated_l3"
        elif w in self.registry.intersections:
            tier = "intersection_only"
        else:
            tier = "letters_only"
        tok = self.registry.resolve_token(w)
        return {
            "word": w,
            "tier": tier,
            "cluster_id": cid,
            "cluster_score": score,
            "prime": tok.prime,
            "intersection_only": self.registry.is_intersection_only(w),
            "parent_primes": tok.parent_primes,
            "lattice_local": self.lattice_address(w),
            "letter_chain": letter_chain(w),
        }

    def lattice_address(self, word: str, n: int = 7, lattice_id: int = 1) -> tuple[float, float, float]:
        from aethos_lattice import LatticeId

        return self.registry.lattice_address(word, LatticeTier.L3_WORD, n=n, lattice_id=LatticeId(lattice_id))

    def semantic_overlay(self, word: str, **kwargs: object) -> SemanticOverlay:
        return overlay_for_word(self.registry, word, **kwargs)  # type: ignore[arg-type]

    def frequency_profile(self) -> FrequencyProfile:
        return FrequencyProfile.from_reader(self.reader)

    def token_savings(self) -> TokenSavingsReport:
        return report_from_reader(self.reader)

    def save(self, path: str) -> None:
        save_reader(self.reader, path)

    @classmethod
    def load(cls, path: str, *, rebuild_every: int = 3, core: AethosLatticeCore | None = None) -> TokenProcessor:
        reader = load_reader(path, rebuild_every=rebuild_every)
        return cls(core=core or AethosLatticeCore(), reader=reader, rebuild_every=rebuild_every)

    def explain(self, word: str, context: list[str] | None = None) -> str:
        r = self.resolve(word, context)
        lines = [
            self.reader.explain(word),
            self.reader.explain_natural(word, context),
            f"  token tier: {r['tier']}, prime={r['prime']}",
            f"  lattice local (core): {r['lattice_local']}",
            f"  cluster: {r['cluster_id']!r} ({r['cluster_score']:.4f})",
        ]
        return "\n".join(lines)


def demo() -> None:
    from aethos_pipeline import smoke_corpus

    print("=" * 60)
    print("TOKEN PROCESSOR (uses lattice core for coordinates only)")
    print("=" * 60)
    core = AethosLatticeCore()
    print(core.summary())
    print()

    proc = TokenProcessor(core=core, rebuild_every=2)
    proc.ingest(*smoke_corpus())
    print(f"  words: {len(proc.registry.word_counts)}")
    print(f"  savings: {proc.token_savings().pool_primes_saved} prime slots saved")
    print(proc.explain("apple", ["phone", "chip"]))


if __name__ == "__main__":
    demo()
