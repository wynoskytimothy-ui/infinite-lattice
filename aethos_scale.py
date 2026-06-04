"""
Scale targets and metrics for corpus-wide ingest.

Goals (user):
  - Any corpus size
  - <= 50 ms per document ingest (p99 on typical web prose)
  - ~100 bytes per document index metadata
  - Accuracy >= strong baselines (PMI clusters + context routing)

Section 5 mapping: fast ingest = light compression passes; full cluster
discover = heavier measurement pin (deferred until query or batch flush).
"""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass, field


@dataclass
class ScaleConfig:
    """Tunable knobs for large-corpus ingest."""

    target_ms_per_doc: float = 50.0
    target_bytes_per_doc: int = 100
    rebuild_every: int = 64
    lazy_clusters: bool = True
    max_window_tokens: int = 48
    max_corr_pairs_per_doc: int = 256
    skip_stopword_pairs: bool = True
    defer_l2_promotion: bool = True
    fast_cluster: bool = True
    fast_ingest: bool = True
    max_contexts_per_word: int = 8


@dataclass(frozen=True)
class DocFingerprint:
    """Compact per-document index (~24–100 bytes)."""

    doc_id: int
    token_count: int
    unique_tokens: int
    stream_crc: int
    top_hub: str = ""

    def encoded_size(self) -> int:
        hub = len(self.top_hub.encode("utf-8"))
        return 4 + 4 + 4 + 4 + 1 + hub  # struct-like upper bound

    def as_dict(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "token_count": self.token_count,
            "unique_tokens": self.unique_tokens,
            "stream_crc": self.stream_crc,
            "top_hub": self.top_hub,
            "bytes": self.encoded_size(),
        }


@dataclass
class DocTiming:
    doc_id: int
    tokens: int
    elapsed_ms: float
    fingerprint_bytes: int

    @property
    def within_target(self) -> bool:
        return self.elapsed_ms <= 50.0


@dataclass
class ScaleMetrics:
    documents: int = 0
    total_tokens: int = 0
    timings: list[DocTiming] = field(default_factory=list)
    fingerprints: list[DocFingerprint] = field(default_factory=list)

    def record(self, timing: DocTiming, fp: DocFingerprint) -> None:
        self.documents += 1
        self.total_tokens += timing.tokens
        self.timings.append(timing)
        self.fingerprints.append(fp)

    @property
    def p50_ms(self) -> float:
        if not self.timings:
            return 0.0
        xs = sorted(t.elapsed_ms for t in self.timings)
        return xs[len(xs) // 2]

    @property
    def p99_ms(self) -> float:
        if not self.timings:
            return 0.0
        xs = sorted(t.elapsed_ms for t in self.timings)
        return xs[min(len(xs) - 1, int(len(xs) * 0.99))]

    @property
    def mean_bytes_per_doc(self) -> float:
        if not self.fingerprints:
            return 0.0
        return sum(f.encoded_size() for f in self.fingerprints) / len(self.fingerprints)

    def pass_latency(self, target_ms: float = 50.0) -> bool:
        return self.p99_ms <= target_ms

    def pass_fingerprint(self, target_bytes: int = 100) -> bool:
        return self.mean_bytes_per_doc <= target_bytes

    def summary(self, *, target_ms: float = 50.0, target_bytes: int = 100) -> str:
        lat = "PASS" if self.pass_latency(target_ms) else "FAIL"
        fp = "PASS" if self.pass_fingerprint(target_bytes) else "FAIL"
        return (
            f"Scale metrics ({self.documents} docs, {self.total_tokens} tokens)\n"
            f"  latency p50: {self.p50_ms:.2f} ms  p99: {self.p99_ms:.2f} ms  [{lat} vs {target_ms} ms]\n"
            f"  fingerprint: {self.mean_bytes_per_doc:.0f} bytes/doc  [{fp} vs {target_bytes} B]"
        )


def fingerprint_document(doc_id: int, tokens: list[str], top_hub: str = "") -> DocFingerprint:
    uniq = len(set(tokens))
    blob = " ".join(tokens).encode("utf-8")
    return DocFingerprint(
        doc_id=doc_id,
        token_count=len(tokens),
        unique_tokens=uniq,
        stream_crc=zlib.crc32(blob) & 0xFFFFFFFF,
        top_hub=top_hub[:32],
    )


def timed_ingest_one(pipe, doc_id: int, text: str) -> tuple[DocTiming, DocFingerprint]:
    from aethos_tokenize import tokenize_words

    tokens = tokenize_words(text)
    t0 = time.perf_counter()
    pipe.ingest_one(text, finalize=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    hub = ""
    if pipe.reader.cluster_hubs:
        hub = next(iter(pipe.reader.cluster_hubs.values()), "")
    fp = fingerprint_document(doc_id, tokens, top_hub=hub)
    timing = DocTiming(doc_id=doc_id, tokens=len(tokens), elapsed_ms=elapsed_ms, fingerprint_bytes=fp.encoded_size())
    return timing, fp
