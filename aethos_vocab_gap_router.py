"""Vocabulary-gap router — label-free per-query gate for semantic expansion.

Routes zero-shot corridor PRF and taught-correlation expansion only where they
help. Corpus-level mode (from train qrel gap fraction, like choose_bridge) sets
how aggressively PRF fires; teach expansion always requires a key-term gap.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from aethos_append_index import words


@dataclass(frozen=True)
class GapSignal:
    """Glass-box routing decision for one query."""

    route: bool
    gap_score: float
    overlap_ratio: float
    rare_qterms: tuple[str, ...]
    missing_in_pseudo: tuple[str, ...]
    key_missing: tuple[str, ...]  # idf >= key_gate, absent from pseudo pool
    route_prf: bool = False
    route_teach: bool = False


def choose_expansion_mode(queries, train_qrels, corpus):
    """Corpus-level strict vs mismatch mode from train qrel vocabulary gaps.

    Mirrors choose_bridge: characterize the corpus once from train qrels; per-query
    routing remains label-free at test time.
    """
    n_pairs = zero_ov = 0
    for qid, rels in train_qrels.items():
        if qid not in queries:
            continue
        qwords = set(words(queries[qid]))
        for d, sc in rels.items():
            if sc <= 0 or d not in corpus:
                continue
            n_pairs += 1
            if not (qwords & set(words(corpus[d]))):
                zero_ov += 1
    gap_frac = zero_ov / max(n_pairs, 1)
    mode = "mismatch" if gap_frac >= 0.12 else "strict"
    return mode, {"gap_frac": round(gap_frac, 3), "mode": mode}


def measure_vocab_gap(
    query: str,
    lex: dict,
    corpus: dict,
    idf,
    *,
    k_pseudo: int = 10,
    rare_gate: float = 3.0,
    key_gate: float = 5.5,
    overlap_max: float = 0.35,
    mode: str = "strict",
    prf_doc_count: int = 0,
) -> GapSignal:
    """Score vocabulary gap between query and lexical top-K pseudo docs.

    Label-free at query time: uses only query text, lexical scores, corpus text.
    """
    qwords = set(words(query))
    if not qwords or not lex:
        return GapSignal(False, 0.0, 1.0, (), (), ())

    pseudo = sorted(lex, key=lex.get, reverse=True)[:k_pseudo]
    pseudo_words: set[str] = set()
    for d in pseudo:
        pseudo_words |= set(words(corpus.get(d, "")))

    rare = tuple(w for w in qwords if idf(w) >= rare_gate)
    missing = tuple(w for w in rare if w not in pseudo_words)
    key_missing = tuple(w for w in qwords if idf(w) >= key_gate and w not in pseudo_words)
    overlap = len(qwords & pseudo_words) / len(qwords)

    gap_route = False
    if key_missing:
        gap_route = True
    elif missing and overlap <= overlap_max:
        gap_route = True
    elif missing and len(missing) >= max(1, (len(rare) + 1) // 2):
        gap_route = True

    # PRF: strict = gap only; mismatch corpus = trust corridor bunch broadly
    if mode == "mismatch":
        route_prf = bool(rare) or prf_doc_count >= 8
    else:
        route_prf = gap_route

    # Teach: always gap-gated (key rare term missing from pseudo pool)
    route_teach = bool(key_missing)

    gap_score = (len(missing) / max(len(rare), 1)) * (1.0 - overlap)
    if key_missing:
        gap_score = max(gap_score, 0.5 + 0.1 * len(key_missing))

    return GapSignal(
        gap_route or route_prf or route_teach,
        gap_score,
        overlap,
        rare,
        missing,
        key_missing,
        route_prf=route_prf,
        route_teach=route_teach,
    )


def prf_expansion(
    idx,
    corpus,
    idf,
    query: str,
    lex: dict,
    *,
    k_prf: int = 10,
    min_docs: int = 2,
    rare_gate: float = 3.0,
    top_terms: int = 40,
    rare_doc_cache: dict | None = None,
) -> dict[int, float]:
    """Zero-shot rare-term corridor bunch → doc expansion weights (no fusion)."""
    cand = sorted(lex, key=lex.get, reverse=True)[:100]
    if not cand:
        return {}
    pseudo = cand[:k_prf]
    bunch_docs: Counter[str] = Counter()
    bunch_idf: dict[str, float] = {}
    qwords = set(words(query))

    for d in pseudo:
        if rare_doc_cache is not None:
            rt = rare_doc_cache.get(d)
            if rt is None:
                rt = {w for w in set(words(corpus[d])) if idf(w) >= rare_gate}
                rare_doc_cache[d] = rt
        else:
            rt = {w for w in set(words(corpus[d])) if idf(w) >= rare_gate}
        for w in rt:
            bunch_docs[w] += 1
            bunch_idf[w] = idf(w)

    bunch = [(w, c) for w, c in bunch_docs.items() if c >= min_docs and w not in qwords]
    bunch.sort(key=lambda x: (-(x[1] * bunch_idf[x[0]]), x[0]))
    bunch = bunch[:top_terms]

    exp: dict[int, float] = defaultdict(float)
    for w, c in bunch:
        p = idx.token_prime.get(("w", w))
        if p is None:
            continue
        wt = bunch_idf[w] * c
        for d, tf in idx.postings.get(p, {}).items():
            if d in idx.alive:
                exp[d] += wt * tf / (tf + 1.0)
    return dict(exp)


def fuse_lex_expansion(
    lex: dict,
    exp: dict,
    *,
    lam: float = 0.3,
    n_expand: int = 30,
    k: int = 100,
) -> list:
    """Conservative rerank: lexical candidates + bounded pool expansion."""
    if not exp:
        return sorted(lex, key=lex.get, reverse=True)[:k]
    cand = sorted(lex, key=lex.get, reverse=True)[:100]
    cset = set(cand)
    extra = [d for d in sorted(exp, key=exp.get, reverse=True) if d not in cset][:n_expand]
    pool = cand + extra
    lmax = max((lex.get(d, 0.0) for d in pool), default=1.0) or 1.0
    emax = max(exp.values()) or 1.0
    final = {d: lex.get(d, 0.0) / lmax + lam * exp.get(d, 0.0) / emax for d in pool}
    return sorted(final, key=final.get, reverse=True)[:k]


def routed_search(
    idx,
    corpus,
    idf,
    query: str,
    lex: dict,
    teach=None,
    *,
    lam: float = 0.3,
    n_expand: int = 30,
    k: int = 100,
    rare_doc_cache: dict | None = None,
    mode: str = "strict",
    signal: GapSignal | None = None,
    pair_bridges=None,
) -> tuple[list, GapSignal]:
    """Lexical floor, or lexical + gated PRF + gated teach expansion."""
    prf = prf_expansion(
        idx, corpus, idf, query, lex, rare_doc_cache=rare_doc_cache,
    )
    sig = signal or measure_vocab_gap(
        query, lex, corpus, idf, mode=mode, prf_doc_count=len(prf),
    )
    route_teach = sig.route_teach
    if teach is not None:
        qwords = set(words(query))
        if any(w in teach.edges for w in qwords) or any(
            w in getattr(teach, "definitions", {}) for w in qwords
        ):
            route_teach = True

    exp: dict[int, float] = defaultdict(float)
    if pair_bridges is not None:
        from aethos_bridges import bridge_expansion
        for d, s in bridge_expansion(idx, pair_bridges, query).items():
            exp[d] += s

    if not sig.route_prf and not route_teach and not exp:
        return sorted(lex, key=lex.get, reverse=True)[:k], sig

    if route_teach and teach is not None:
        rq = teach.rewrite_query(query)
        if rq != query:
            lex = idx._score(rq)

    if sig.route_prf:
        for d, s in prf.items():
            exp[d] += s

    if route_teach and teach is not None:
        for d, s in teach.expand_scores(query).items():
            exp[d] += s

    if not exp:
        return sorted(lex, key=lex.get, reverse=True)[:k], sig

    return fuse_lex_expansion(lex, dict(exp), lam=lam, n_expand=n_expand, k=k), sig
