"""Shared types for the clean pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StorageReport:
    """Hot per-document storage budget (cert + hub, no pickle sidecars)."""

    fingerprint_bytes_per_doc: float
    hub_bytes_per_doc: float
    hot_bytes_per_doc: float
    n_docs: int
    pool_cap: int

    def summary(self) -> str:
        return (
            f"storage  docs={self.n_docs}  pool_cap={self.pool_cap}\n"
            f"  fingerprint:  {self.fingerprint_bytes_per_doc:.0f} B/doc\n"
            f"  hub cert:     {self.hub_bytes_per_doc:.0f} B/doc\n"
            f"  hot total:    {self.hot_bytes_per_doc:.0f} B/doc"
        )


@dataclass
class CleanQueryResult:
    query: str
    ranked_ids: list[str]
    scores: list[float] | None
    latency_ms: float
    n_candidates: int
    route_tier: str
    n_kappa_keys: int


@dataclass
class CleanEvalResult:
    dataset: str
    preset: str
    mode: str
    n_docs: int
    n_queries: int
    ndcg10: float
    recall10: float
    recall100: float
    p50_query_ms: float
    p99_query_ms: float
    p50_ingest_ms: float
    storage: StorageReport
    bm25_ref: float | None = None
    gate_passed: bool | None = None
    gate_report: str | None = None

    def summary(self) -> str:
        ref = ""
        if self.bm25_ref is not None:
            delta = self.ndcg10 - self.bm25_ref
            ref = f"  BM25 ref:     {self.bm25_ref:.3f}  (delta {delta:+.3f})\n"
        gate = ""
        if self.gate_passed is not None:
            gate = f"  gate:         {'PASS' if self.gate_passed else 'FAIL'}\n"
        return (
            f"{self.dataset} [{self.preset}/{self.mode}]  "
            f"docs={self.n_docs}  queries={self.n_queries}\n"
            f"  NDCG@10:      {self.ndcg10:.4f}\n"
            f"  R@10:         {self.recall10:.4f}\n"
            f"  R@100:        {self.recall100:.4f}\n"
            f"{ref}{gate}"
            f"  query p50:    {self.p50_query_ms:.2f} ms\n"
            f"  query p99:    {self.p99_query_ms:.2f} ms\n"
            f"  ingest p50:   {self.p50_ingest_ms:.2f} ms/doc\n"
            f"  hot B/doc:    {self.storage.hot_bytes_per_doc:.0f}"
        )

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "preset": self.preset,
            "mode": self.mode,
            "n_docs": self.n_docs,
            "n_queries": self.n_queries,
            "ndcg10": self.ndcg10,
            "recall10": self.recall10,
            "recall100": self.recall100,
            "p50_query_ms": self.p50_query_ms,
            "p99_query_ms": self.p99_query_ms,
            "p50_ingest_ms": self.p50_ingest_ms,
            "fingerprint_bpd": self.storage.fingerprint_bytes_per_doc,
            "hub_bpd": self.storage.hub_bytes_per_doc,
            "hot_bpd": self.storage.hot_bytes_per_doc,
            "bm25_ref": self.bm25_ref,
            "gate_passed": self.gate_passed,
        }
