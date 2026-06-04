"""
AETHOS unified pipeline — one entry for read, resolve, dots, and reports.

Explicit subsystems:
  - NaturalReader: L1-L9 from text (promotion + clusters)
  - Codec: bytes -> formula dot (payload witness)
  - Words: ordered letters -> dot at shared site (tab/bat)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from aethos_codec import Dot, encode_bytes, encode_text, verify_dot
from aethos_core import AethosLatticeCore
from aethos_lattice import LatticeId
from aethos_overlay import SemanticOverlay
from aethos_promotion import LatticeTier, PromotionRegistry
from aethos_token_processor import TokenProcessor
from aethos_token_savings import TokenSavingsReport
from aethos_words import SharedSite, encode_word_at_site


@dataclass
class PipelineReport:
    """Snapshot after ingest."""

    token_savings: TokenSavingsReport
    documents_read: int
    cluster_count: int
    correlation_edges: int
    intersection_only_words: int
    dedicated_l3_words: int
    pool_primes_allocated: int
    invariant_errors: tuple[str, ...] = ()

    def summary(self) -> str:
        ts = self.token_savings
        inv = "OK" if not self.invariant_errors else f"FAIL ({len(self.invariant_errors)})"
        pool = ""
        if self.pool_primes_allocated:
            pool = f"\n  pool tiers:            see pipe.pool_report()"
        return (
            f"AETHOS pipeline report\n"
            f"  documents read:        {self.documents_read}\n"
            f"  emergent clusters:     {self.cluster_count}\n"
            f"  unique words:          {ts.unique_words}\n"
            f"  intersection-only:     {self.intersection_only_words}\n"
            f"  dedicated L3:          {self.dedicated_l3_words}\n"
            f"  pool primes used:      {self.pool_primes_allocated}\n"
            f"  prime slots saved:     {ts.pool_primes_saved} ({ts.prime_slots_saved_ratio * 100:.0f}%)\n"
            f"  L4-L6 edges (floats):  {self.correlation_edges}\n"
            f"  promotion invariants:  {inv}{pool}"
        )


def check_promotion_invariants(registry: PromotionRegistry) -> list[str]:
    """Rules that must hold after any ingest."""
    errors: list[str] = []

    for w in registry.intersections:
        if (LatticeTier.L3_WORD, w) in registry.promoted:
            errors.append(f"{w!r}: both intersection and dedicated L3")

    for (tier, text), tok in registry.promoted.items():
        if tier != LatticeTier.L3_WORD:
            continue
        if tok.intersection_only:
            errors.append(f"{text!r}: L3 promoted token marked intersection_only")
        if registry.word_counts.get(text, 0) >= 2 and not registry.contexts_differ(text):
            errors.append(f"{text!r}: dedicated L3 but contexts do not differ")

    for w, count in registry.word_counts.items():
        if count == 1 and (LatticeTier.L3_WORD, w) in registry.promoted:
            errors.append(f"{w!r}: singleton has dedicated L3 pool prime")

    return errors


@dataclass
class AethosPipeline:
    """
    Full stack: lattice core + token processor + codec/word dots.

    For physics or custom projects use AethosLatticeCore alone.
    For NLP / promotion use TokenProcessor or this combined pipeline.
    """

    rebuild_every: int = 3
    core: AethosLatticeCore = field(default_factory=AethosLatticeCore)
    tokens: TokenProcessor | None = None

    def __post_init__(self) -> None:
        if self.tokens is None:
            self.tokens = TokenProcessor(core=self.core, rebuild_every=self.rebuild_every)
        elif self.tokens.core is not self.core:
            self.tokens.core = self.core

    @property
    def reader(self):
        return self.tokens.reader

    @property
    def registry(self) -> PromotionRegistry:
        return self.tokens.registry

    def ingest(self, *documents: str) -> None:
        self.tokens.ingest(*documents)

    def ingest_one(self, text: str, *, finalize: bool = False) -> None:
        """Single-document ingest for streaming / scale benchmarks."""
        self.reader.read_one(text, finalize=finalize)

    def flush(self) -> None:
        """Finalize lazy cluster discovery after batched ingest_one calls."""
        self.reader.ensure_clusters()

    def apply_scale_config(self, cfg: object) -> None:
        from aethos_scale import ScaleConfig

        if isinstance(cfg, ScaleConfig):
            self.rebuild_every = cfg.rebuild_every
            self.tokens.rebuild_every = cfg.rebuild_every
            self.reader.apply_scale(cfg)

    def resolve(self, word: str, context: Iterable[str] | None = None) -> dict[str, object]:
        return self.tokens.resolve(word, context)

    def dot_for_bytes(self, raw: bytes, prime_order: tuple[int, ...] | None = None) -> Dot:
        dot = encode_bytes(raw, prime_order=prime_order)
        if not verify_dot(dot):
            raise ValueError("encoded dot failed verify_dot")
        return dot

    def dot_for_text(self, text: str) -> Dot:
        dot = encode_text(text)
        if not verify_dot(dot):
            raise ValueError("encoded dot failed verify_dot")
        return dot

    def dot_for_word(self, word: str, site: SharedSite | None = None) -> Dot:
        dot = encode_word_at_site(word, site)
        if not verify_dot(dot):
            raise ValueError("word dot failed verify_dot")
        return dot

    def semantic_overlay(
        self,
        word: str,
        *,
        n: int = 7,
        lattice_id: LatticeId = LatticeId.L01,
        origin_path: str = "O0",
        include_word_dot: bool = True,
    ) -> SemanticOverlay:
        return self.tokens.semantic_overlay(
            word,
            n=n,
            lattice_id=lattice_id,
            origin_path=origin_path,
            include_word_dot=include_word_dot,
        )

    def report(self) -> PipelineReport:
        reg = self.registry
        ts = self.tokens.token_savings()
        return PipelineReport(
            token_savings=ts,
            documents_read=self.reader.documents_read,
            cluster_count=len(self.reader.cross.categories),
            correlation_edges=len(reg.correlations),
            intersection_only_words=len(reg.intersections),
            dedicated_l3_words=sum(1 for k in reg.promoted if k[0] == LatticeTier.L3_WORD),
            pool_primes_allocated=reg._next_promotion_idx,
            invariant_errors=tuple(check_promotion_invariants(reg)),
        )

    def pool_report(self) -> str:
        return self.registry.pool_usage_report()

    def save(self, path: str) -> None:
        self.tokens.save(path)

    @classmethod
    def load(cls, path: str, *, rebuild_every: int = 3) -> AethosPipeline:
        core = AethosLatticeCore()
        tokens = TokenProcessor.load(path, rebuild_every=rebuild_every, core=core)
        return cls(rebuild_every=rebuild_every, core=core, tokens=tokens)

    def explain(self, word: str, context: list[str] | None = None) -> str:
        return self.tokens.explain(word, context)

    def encode_document(self, text: str, doc_index: int = 0):
        """Return LatticeToken list for one document (Tier 2 emitter)."""
        from aethos_lattice_token import encode_document

        return encode_document(
            text,
            self.registry,
            doc_index=doc_index,
            infer_cluster=self.reader.infer_cluster,
        )

    def open_lattice_project(self, name: str, **kwargs: object) -> object:
        """Open a pure lattice project on the core (no tokens)."""
        return self.core.open_project(name, **kwargs)


def smoke_corpus() -> tuple[str, ...]:
    """Small fixed corpus for unified smoke tests and run_aethos."""
    return (
        "phone phone phone technical chip software",
        "phone technical hardware network",
        "apple phone chip technical",
        "apple fruit pie orchard",
        "xylophone obscure rareword once",
        "banana fruit once",
        "zebra unique animal",
    )


def demo() -> None:
    pipe = AethosPipeline(rebuild_every=2)
    pipe.ingest(*smoke_corpus())
    print(pipe.report().summary())
    print()
    print(pipe.explain("apple", ["phone", "chip"]))
    print()
    print(pipe.explain("apple", ["fruit", "pie"]))
    dot = pipe.dot_for_text("hello pipeline")
    print(f"\n  text dot: {dot.coord}  verify={verify_dot(dot)}")


if __name__ == "__main__":
    demo()
