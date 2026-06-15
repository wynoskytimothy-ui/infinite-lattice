#!/usr/bin/env python3
"""
Seven data-type compression deep dive — lattice layers vs baselines.

Measures per document (avg):
  raw UTF-8, zlib, codec witness, hub signatures, κ keys, notches,
  critical-line pin estimate, lattice total, ratios.

Does not require BEIR download — uses synthetic corpora per type.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aethos_codec import witness_from_bytes
from aethos_hub_signature import build_all_hub_signatures
from aethos_tokenize import tokenize_words
from eval_beir import ingest_corpus, make_pipeline, _tune_registry_for_beir
from pipeline.bit_03_doc_attractor_set import (
    build_attractor_index_from_hub_signatures,
    doc_attractor_set_from_signature,
)
from pipeline.bit_06_notch_bind import build_all_notch_fingerprints

KAPPA_KEY_BYTES = 12  # 3 × int32
CRITICAL_PIN_BYTES = 8  # prime(4) + leg_sum(2) + band_side(1) + n(1) — proposed wire


@dataclass
class TypeCorpus:
    name: str
    description: str
    docs: dict[str, str]


@dataclass
class DocMetrics:
    raw: int
    zlib: int
    codec_compact: int
    hub: int
    kappa_keys: int
    notch: int
    critical_pins: int
    naive_float_hubs: int
    unique_tokens: int
    hub_count: int
    kappa_count: int


@dataclass
class TypeReport:
    name: str
    description: str
    n_docs: int
    avg: DocMetrics
    totals: DocMetrics = field(default_factory=DocMetrics)
    notes: list[str] = field(default_factory=list)


def _leg_sum_im_led(coord: tuple[float, float, float]) -> tuple[int, bool]:
    x, y = int(round(coord[0])), int(round(coord[1]))
    return x + y, y > x


def critical_pin_bytes_for_sig(sig) -> int:
    """Proposed: one pin per hub (prime + leg_sum + band/side + n)."""
    return len(sig.hubs) * CRITICAL_PIN_BYTES


def naive_float_hub_bytes(sig, *, hubs: int = 12) -> int:
    """Naive: 12 hubs × (3 float32 + word string)."""
    n = min(hubs, len(sig.hubs))
    total = 0
    for entry in list(sig.hubs.values())[:n]:
        total += 12 + len(entry.word.encode("utf-8"))
    return total


def measure_doc(
    doc_id: str,
    text: str,
    *,
    registry,
    hub_sigs: dict,
    notch_fps: dict,
) -> DocMetrics:
    raw = text.encode("utf-8")
    z = zlib.compress(raw, level=9)
    w = witness_from_bytes(raw)
    compact = len(w.to_compact().encode("ascii"))

    sig = hub_sigs[doc_id]
    das = doc_attractor_set_from_signature(registry, sig)
    kappa_n = len(das.keys) if das else 0

    notch_b = 0
    if doc_id in notch_fps:
        notch_b = notch_fps[doc_id].encoded_size()

    words = tokenize_words(text)
    unique = len(frozenset(words))

    return DocMetrics(
        raw=len(raw),
        zlib=len(z),
        codec_compact=compact,
        hub=sig.encoded_size(),
        kappa_keys=kappa_n * KAPPA_KEY_BYTES,
        notch=notch_b,
        critical_pins=critical_pin_bytes_for_sig(sig),
        naive_float_hubs=naive_float_hub_bytes(sig),
        unique_tokens=unique,
        hub_count=len(sig.hubs),
        kappa_count=kappa_n,
    )


def build_corpora() -> list[TypeCorpus]:
    rng = os.urandom(4096)

    scientific = [
        "Vitamin D supplementation reduces inflammatory cytokine expression in macrophages.",
        "Randomized trials show mRNA vaccines elicit robust neutralizing antibody titers.",
        "CRISPR Cas9 editing introduces double strand breaks repaired by NHEJ or HDR.",
        "Mitochondrial oxidative phosphorylation couples electron transport to ATP synthesis.",
        "Microglia activation contributes to neuroinflammation in Alzheimer disease models.",
    ] * 4

    technical = [
        "autophagy lysosome phagocytosis endocytosis exocytosis ribosome transcriptome",
        "kinase phosphatase methyltransferase acetyltransferase ubiquitin proteasome",
        "hypoxia angiogenesis vasculogenesis homeostasis hemostasis thrombosis",
        "electrophoresis chromatography spectroscopy crystallography diffractometry",
    ] * 5

    repetitive = [
        "INFO worker-3 task complete status=OK latency_ms=12\n" * 80,
        "METRIC cpu=0.12 mem=4096 disk=ok shard=7\n" * 60,
    ] * 5

    random_text = [
        base64.b64encode(rng[i : i + 512]).decode("ascii") for i in range(0, 2048, 256)
    ] * 4

    json_docs = []
    for i in range(20):
        rows = [
            {"id": j, "sensor": "temp", "value": 20.0 + j * 0.1, "ok": True}
            for j in range(50)
        ]
        json_docs.append(json.dumps({"batch": i, "readings": rows}, separators=(",", ":")))

    morph = [
        "autophagy autophagic autophagous autophagosome autophagolysosome lysosomal",
        "phosphorylate phosphorylation phosphorylated phosphatase dephosphorylation",
        "inflammation inflammatory inflammatories antiinflammatory immunomodulatory",
        "replicate replication replicative replicator replicon transcript transcriptional",
    ] * 5

    phrases = [
        "mitochondrial oxidative stress response pathway activation under hypoxic conditions",
        "blood brain barrier permeability increases during acute neuroinflammatory episodes",
        "single cell RNA sequencing reveals heterogeneous tumor microenvironment states",
        "long noncoding RNA regulates chromatin remodeling at developmental enhancer loci",
        "pattern recognition receptor signaling drives innate immune interferon production",
    ] * 4

    def pack(name: str, desc: str, texts: list[str]) -> TypeCorpus:
        return TypeCorpus(
            name=name,
            description=desc,
            docs={f"{name}_{i}": t for i, t in enumerate(texts)},
        )

    return [
        pack(
            "scientific_prose",
            "SciFact-style declarative sentences; moderate vocabulary overlap",
            scientific,
        ),
        pack(
            "technical_vocabulary",
            "Dense domain term lists; high promotion / pool-prime potential",
            technical,
        ),
        pack(
            "repetitive_logs",
            "Near-duplicate lines; zlib-friendly, high token repetition",
            repetitive,
        ),
        pack(
            "random_incompressible",
            "Base64 random payloads; poor zlib and weak lattice reuse",
            random_text,
        ),
        pack(
            "structured_json",
            "Repeated keys/records; schema redundancy, string-heavy",
            json_docs,
        ),
        pack(
            "morphological_variants",
            "Shared stems/suffixes; correlation via letter primes and meets",
            morph,
        ),
        pack(
            "long_phrase_correlations",
            "Multi-word scientific phrases; L4/L5 composite depth",
            phrases,
        ),
    ]


def analyze_type(corpus: TypeCorpus, *, mode: str = "scale") -> TypeReport:
    pipe = make_pipeline(mode)
    _tune_registry_for_beir(pipe.registry)

    beir_corpus = {did: {"text": txt} for did, txt in corpus.docs.items()}
    metrics, cidx = ingest_corpus(pipe, beir_corpus, mode=mode)

    hub_sigs = build_all_hub_signatures(
        cidx.doc_ids, cidx.doc_tokens, pipe.registry, top_k=12,
    )
    attractor_index = build_attractor_index_from_hub_signatures(
        pipe.registry, hub_sigs,
    )
    notch_fps = build_all_notch_fingerprints(hub_sigs, pipe.registry)

    doc_metrics: list[DocMetrics] = []
    for did in cidx.doc_ids:
        doc_metrics.append(
            measure_doc(
                did,
                corpus.docs[did],
                registry=pipe.registry,
                hub_sigs=hub_sigs,
                notch_fps=notch_fps,
            )
        )

    def avg_field(fn):
        return sum(fn(d) for d in doc_metrics) / max(len(doc_metrics), 1)

    def sum_field(fn):
        return sum(fn(d) for d in doc_metrics)

    avg = DocMetrics(
        raw=int(avg_field(lambda d: d.raw)),
        zlib=int(avg_field(lambda d: d.zlib)),
        codec_compact=int(avg_field(lambda d: d.codec_compact)),
        hub=int(avg_field(lambda d: d.hub)),
        kappa_keys=int(avg_field(lambda d: d.kappa_keys)),
        notch=int(avg_field(lambda d: d.notch)),
        critical_pins=int(avg_field(lambda d: d.critical_pins)),
        naive_float_hubs=int(avg_field(lambda d: d.naive_float_hubs)),
        unique_tokens=int(avg_field(lambda d: d.unique_tokens)),
        hub_count=int(avg_field(lambda d: d.hub_count)),
        kappa_count=int(avg_field(lambda d: d.kappa_count)),
    )

    totals = DocMetrics(
        raw=sum_field(lambda d: d.raw),
        zlib=sum_field(lambda d: d.zlib),
        codec_compact=sum_field(lambda d: d.codec_compact),
        hub=sum_field(lambda d: d.hub),
        kappa_keys=sum_field(lambda d: d.kappa_keys),
        notch=sum_field(lambda d: d.notch),
        critical_pins=sum_field(lambda d: d.critical_pins),
        naive_float_hubs=sum_field(lambda d: d.naive_float_hubs),
        unique_tokens=sum_field(lambda d: d.unique_tokens),
        hub_count=sum_field(lambda d: d.hub_count),
        kappa_count=sum_field(lambda d: d.kappa_count),
    )

    notes: list[str] = []
    notes.append(f"ingest fingerprint avg {metrics.mean_bytes_per_doc:.0f} B/doc (scale metrics)")
    notes.append(
        f"attractor index: {attractor_index.summary()['buckets']} buckets, "
        f"avg {attractor_index.summary()['avg_keys_per_doc']:.1f} κ/doc"
    )

    # S-partner dedup potential on hubs
    dedup_savings = 0
    for sig in hub_sigs.values():
        seen_sums: dict[int, int] = {}
        for e in sig.hubs.values():
            s, _ = _leg_sum_im_led(e.coord)
            if s in seen_sums:
                dedup_savings += CRITICAL_PIN_BYTES
            seen_sums[s] = seen_sums.get(s, 0) + 1
    if dedup_savings:
        notes.append(f"critical-line S-partner dedup could save ~{dedup_savings // len(hub_sigs)} B/doc avg")

    return TypeReport(
        name=corpus.name,
        description=corpus.description,
        n_docs=len(cidx.doc_ids),
        avg=avg,
        totals=totals,
        notes=notes,
    )


def lattice_total(m: DocMetrics) -> int:
    return m.hub + m.kappa_keys + m.notch


def ratio(raw: int, encoded: int) -> float:
    if encoded <= 0:
        return 0.0
    return raw / encoded


def print_report(reports: list[TypeReport]) -> None:
    print("=" * 88)
    print("  AETHOS compression — seven data types (synthetic corpora, scale ingest)")
    print("=" * 88)

    header = (
        f"{'type':<22} {'raw':>8} {'zlib':>7} {'codec':>7} "
        f"{'hubs':>6} {'kappa':>6} {'notch':>6} {'lattice':>8} "
        f"{'crit':>6} {'lat/raw':>7} {'z/raw':>6}"
    )
    print(header)
    print("-" * len(header))

    for r in reports:
        a = r.avg
        lat = lattice_total(a)
        print(
            f"{r.name:<22} {a.raw:>8} {a.zlib:>7} {a.codec_compact:>7} "
            f"{a.hub:>6} {a.kappa_keys:>6} {a.notch:>6} {lat:>8} "
            f"{a.critical_pins:>6} {ratio(a.raw, lat):>7.1f}x {ratio(a.raw, a.zlib):>6.1f}x"
        )

    print()
    print("Columns (bytes/doc avg):")
    print("  raw     = UTF-8 document text")
    print("  zlib    = zlib.compress(level=9) — lossless generic baseline")
    print("  codec   = aethos_codec IntersectionWitness.to_compact() — lossless payload+witness")
    print("  hubs    = LatticeHubSignature.encoded_size() — top-12 hub entries")
    print("  kappa   = unique κ keys × 12 B — attractor buckets (BIT 3)")
    print("  notch   = bound notch fingerprint (BIT 6)")
    print("  lattice = hubs + kappa + notch — retrieval geometry layer (NOT lossless for raw text)")
    print("  crit    = proposed critical-line pins (8 B × hub count)")
    print("  lat/raw = how many× smaller lattice index is vs raw (correlation store, not full text)")
    print()

    for r in reports:
        a = r.avg
        print(f"## {r.name}")
        print(f"   {r.description}")
        print(f"   docs={r.n_docs}  unique_tokens/doc≈{a.unique_tokens}  hubs/doc≈{a.hub_count}  κ/doc≈{a.kappa_count}")
        print(f"   naive float hubs (12×3f+word): {a.naive_float_hubs} B/doc")
        print(f"   critical pins (proposed):       {a.critical_pins} B/doc")
        print(f"   lattice vs naive hubs:          {ratio(a.naive_float_hubs, lattice_total(a)):.1f}× smaller index")
        for note in r.notes:
            print(f"   - {note}")
        print()

    print("=" * 88)
    print("INTERPRETATION (honest)")
    print("  - Lattice layer is for CORRELATION / RETRIEVAL geometry — not lossless corpus replacement.")
    print("  - Best lat/raw: repetitive + morph (shared primes/meets). Worst: random + zlib-resistant.")
    print("  - codec column is lossless but includes zlib payload — compare zlib for fair entropy coding.")
    print("  - critical-line pins not yet in HubEntry wire format — column is projected savings.")
    print("=" * 88)


def main() -> int:
    reports = []
    for corpus in build_corpora():
        print(f"  analyzing {corpus.name} ({len(corpus.docs)} docs)...", flush=True)
        reports.append(analyze_type(corpus))
    print_report(reports)

    out = ROOT / "logs" / "compression_seven_types.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for r in reports:
        a = r.avg
        payload.append({
            "name": r.name,
            "description": r.description,
            "n_docs": r.n_docs,
            "avg": a.__dict__,
            "lattice_total": lattice_total(a),
            "lattice_over_raw": ratio(a.raw, lattice_total(a)),
            "notes": r.notes,
        })
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"JSON saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
