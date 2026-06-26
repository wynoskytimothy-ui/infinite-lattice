"""
Miss-r1 glossary teach — when train gold has rarest-2 but not literal rarest-1,
build a TeachStore bridge: r1 -> company snippet from that gold doc.

Deterministic, append-only, no external KB. Complements corridor_bridge learning.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from aethos_append_index import words
from aethos_bridges import rarest_query_pair
from aethos_promotion import _chunk_subwords

_SENT_SPLIT = re.compile(r"[.!?]\s+")


def _idf_fn(idx, N):
    cache: dict[str, float] = {}

    def idf(w: str) -> float:
        v = cache.get(w)
        if v is None:
            p = idx.token_prime.get(("w", w))
            v = idx._idf(p, N) if p else 0.0
            cache[w] = v
        return v

    return idf


def gold_company_snippet(
    text: str,
    r1: str,
    r2: str,
    idf,
    *,
    rare_gate: float = 2.5,
    max_chars: int = 360,
) -> str:
    """Short definitional context: r2 anchor + rare company terms from gold."""
    toks = words(text)
    doc_set = set(toks)
    rare = sorted(
        (w for w in doc_set if w != r1 and idf(w) >= rare_gate),
        key=lambda w: (-idf(w), w),
    )[:10]
    anchors = [r2] + [w for w in rare if w != r2][:8]
    for piece in _chunk_subwords(r1):
        if len(piece) >= 4 and piece in doc_set and piece not in anchors:
            anchors.append(piece)

    snippet = ""
    for sent in _SENT_SPLIT.split(text):
        if r2 in set(words(sent)):
            snippet = sent.strip()
            break
    if not snippet:
        snippet = " ".join(anchors)
    else:
        extra = " ".join(w for w in anchors[1:5] if w not in set(words(snippet)))
        if extra:
            snippet = f"{snippet} {extra}"
    return snippet[:max_chars].strip()


def learn_miss_r1_glossary(
    teach,
    idx,
    N: int,
    queries: dict[str, str],
    train_qrels: dict,
    corpus: dict[str, str],
    *,
    min_gold_hits: int = 1,
    rare_gate: float = 2.5,
    per_term: int = 14,
) -> dict[str, int]:
    """Teach r1 -> gold company snippet for train pairs missing literal r1."""
    idf = _idf_fn(idx, N)
    snippets: dict[str, list[str]] = defaultdict(list)
    hits: Counter[str] = Counter()
    qcache: dict[str, tuple[str, str] | None] = {}

    for qid, rels in train_qrels.items():
        if qid not in queries:
            continue
        pair = qcache.get(qid)
        if pair is None:
            pair = rarest_query_pair(queries[qid], idf)
            qcache[qid] = pair
        if pair is None:
            continue
        r1, r2 = pair
        for cid, sc in rels.items():
            if sc <= 0 or cid not in corpus:
                continue
            dtoks = set(words(corpus[cid]))
            if r1 in dtoks or r2 not in dtoks:
                continue
            hits[r1] += 1
            snippets[r1].append(
                gold_company_snippet(corpus[cid], r1, r2, idf, rare_gate=rare_gate)
            )

    taught = 0
    for r1, snips in snippets.items():
        if hits[r1] < min_gold_hits:
            continue
        merged = " ".join(dict.fromkeys(snips))[:480]
        n = teach.teach_edges(r1, merged, bridge_gate=rare_gate, per_term=per_term)
        if n:
            taught += 1

    teach.finalize(top_k=16)
    return {
        "terms_taught": taught,
        "train_miss_r1_pairs": sum(hits.values()),
        "unique_r1_terms": len(snippets),
    }
