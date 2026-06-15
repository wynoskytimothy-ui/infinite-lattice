"""
Encyclopedia teacher — fill gap terms with real definitions as correlations only.

Sources (in order):
  1. Dataset glossary (scifact / nfcorpus / fiqa — curated medical/scientific)
  2. Wikipedia REST API (cached in wiki_cache.json)
  3. Corpus self-definition — the corpus is its own encyclopedia: when no external
     KB defines a term, its rarest corpus co-occurrents become the bridge.
  4. Synthetic bundle text for missing pairs/triples (terms + their definitions)

Nothing is indexed as a retrievable document — only TeachStore correlation edges.
"""

from __future__ import annotations

import importlib
from collections import Counter
from dataclasses import dataclass

from aethos_append_index import words
from aethos_gap_miner import GapReport, format_pair
from aethos_teach_store import TeachStore
from wiki_teacher import _load_cache, _save_cache, define as wiki_define


def corpus_self_definition(
    term: str,
    idx,
    texts: dict[str, str],
    n_docs: int,
    *,
    keep: set[str] | None = None,
    max_docs: int = 12,
    max_words: int = 12,
    min_idf: float = 2.5,
) -> str:
    """Mine a pseudo-definition for an in-corpus term from its own documents.

    The corpus is its own encyclopedia: take the docs where ``term`` is densest,
    collect their rarest co-occurring words, and return them as bridge vocabulary.

    Naive co-occurrence drifts toward popular docs, so by default this is
    GOLD-VALIDATED: ``keep`` restricts co-occurrents to vocabulary that actually
    appears in the training gold documents of queries containing ``term``. The
    corpus proposes the corridor; ground truth filters it. When ``keep`` is None
    no filtering is applied (unsupervised mode — use with care). Returns '' for
    absent terms (no postings) so external KB still owns truly missing vocabulary.
    """
    p = idx.token_prime.get(("w", term))
    if p is None:
        return ""
    postings = idx.postings.get(p)
    if not postings:
        return ""
    top_docs = sorted(
        (d for d in postings if d in idx.alive),
        key=lambda d: postings[d],
        reverse=True,
    )[:max_docs]
    cooc: Counter = Counter()
    for d in top_docs:
        text = texts.get(d)
        if not text:
            continue
        for w in set(words(text)):
            if w == term:
                continue
            if keep is not None and w not in keep:
                continue
            wp = idx.token_prime.get(("w", w))
            if wp is None:
                continue
            iv = idx._idf(wp, n_docs)
            if iv >= min_idf:
                cooc[w] += iv
    return " ".join(w for w, _ in cooc.most_common(max_words))


def build_gold_vocab_by_term(
    queries: dict[str, str],
    train_qrels: dict[str, dict[str, int]],
    texts: dict[str, str],
    *,
    global_id=None,
) -> dict[str, set[str]]:
    """Per query-term, the vocabulary of its train gold documents.

    Used to gold-validate corpus self-teaching: a self-mined corridor word is
    kept only if it appears in a gold doc for some train query using the term.
    ``texts``/``global_id`` map qrel doc ids to the indexed (global) text keys.
    """
    out: dict[str, set[str]] = {}
    for qid, rels in train_qrels.items():
        if qid not in queries:
            continue
        gold_words: set[str] = set()
        for d, s in rels.items():
            if s <= 0:
                continue
            key = global_id(d) if global_id else d
            text = texts.get(key)
            if text:
                gold_words |= set(words(text))
        if not gold_words:
            continue
        for t in set(words(queries[qid])):
            if t in out:
                out[t] |= gold_words
            else:
                out[t] = set(gold_words)
    return out


def inject_glossary_bridges(
    br,
    glossary: dict[str, str],
    idx,
    N: int,
    *,
    per_term: int = 12,
    gate: float = 2.0,
) -> int:
    """Merge glossary definition bridges into RelevanceBridges (knowledge_bridges shape)."""
    added = 0
    for term, definition in glossary.items():
        defw = []
        for w in dict.fromkeys(words(definition)):
            if w == term:
                continue
            p = idx.token_prime.get(("w", w))
            if p is None:
                continue
            i = idx._idf(p, N)
            if i >= gate:
                defw.append((w, i))
        defw.sort(key=lambda x: -x[1])
        kb = defw[:per_term]
        if kb:
            existing = dict(br.bridge.get(term, []))
            for w, wt in kb:
                existing[w] = max(existing.get(w, 0.0), wt)
            br.bridge[term] = sorted(existing.items(), key=lambda x: -x[1])[:12]
            added += 1
    return added


GLOSSARY_MODULES = {
    "scifact": "scifact_glossary",
    "nfcorpus": "nfcorpus_glossary",
    "fiqa": "fiqa_glossary",
}


def load_glossary(name: str) -> dict[str, str]:
    mod = GLOSSARY_MODULES.get(name.split("+")[0].lower())
    if not mod:
        return {}
    return getattr(importlib.import_module(mod), "GLOSSARY", {})


@dataclass
class TeachGapResult:
    terms_taught: int
    pairs_taught: int
    triples_taught: int
    subwords_taught: int
    wiki_fetched: int
    glossary_hits: int
    skipped_empty: int
    bytes_estimate: int


def _lookup_definition(
    term: str,
    glossary: dict[str, str],
    wiki_cache: dict[str, str],
    *,
    use_wiki: bool,
) -> tuple[str, str]:
    """Returns (definition, source) where source is glossary|wiki|''."""
    if term in glossary:
        return glossary[term], "glossary"
    if term in wiki_cache:
        return wiki_cache[term], "wiki" if wiki_cache[term] else ""
    if use_wiki:
        text = wiki_define(term, cache=wiki_cache)
        return text, "wiki" if text else ""
    return "", ""


def teach_gaps(
    teach: TeachStore,
    report: GapReport,
    *,
    glossary: dict[str, str] | None = None,
    queries: dict[str, str] | None = None,
    use_wiki: bool = True,
    max_terms: int = 40,
    max_pairs: int = 24,
    max_triples: int = 16,
    max_subwords: int = 12,
) -> TeachGapResult:
    """
    Feed encyclopedia material into TeachStore for mined gaps.

    Each teach() call builds rare-term co-occurrence corridors on existing
    corpus primes — the text is never added to the retrievable index.
    """
    glossary = glossary or {}
    queries = queries or {}
    wiki_cache = _load_cache()
    res = TeachGapResult(0, 0, 0, 0, 0, 0, 0, 0)
    taught_terms: set[str] = set()

    # Phase 1: glossary sweep on miss queries (curated medical/scientific KB)
    if queries and glossary:
        for term in report.glossary_targets(glossary, queries):
            defn = glossary[term]
            teach.teach_bridge(term, defn)
            taught_terms.add(term)
            res.terms_taught += 1
            res.glossary_hits += 1
            res.bytes_estimate += len(term) + len(defn)

    # Phase 2: priority gaps via glossary + Wikipedia
    for term in report.priority_terms:
        if term in taught_terms or res.terms_taught >= max_terms:
            continue
        defn, src = _lookup_definition(term, glossary, wiki_cache, use_wiki=use_wiki)
        if not defn:
            res.skipped_empty += 1
            continue
        if src == "glossary":
            res.glossary_hits += 1
        elif src == "wiki":
            res.wiki_fetched += 1
        teach.teach_bridge(term, defn)
        taught_terms.add(term)
        res.terms_taught += 1
        res.bytes_estimate += len(term) + len(defn)

    for piece, _ in report.missing_subwords.most_common(max_subwords):
        defn, src = _lookup_definition(piece, glossary, wiki_cache, use_wiki=use_wiki)
        if not defn:
            parent_ctx = " ".join(
                t for t in report.priority_terms[:6]
                if piece in t
            )
            defn = f"{piece} morphological subword component in {parent_ctx or 'technical vocabulary'}"
        teach.teach(f"{piece} {defn}")
        res.subwords_taught += 1

    for pair, _ in report.missing_pairs.most_common(max_pairs):
        a, b = pair
        parts = []
        for t in (a, b):
            d, _ = _lookup_definition(t, glossary, wiki_cache, use_wiki=use_wiki)
            parts.append(f"{t} {d}" if d else t)
        bundle = f"{format_pair(pair)} correlated concepts: " + " ".join(parts)
        teach.teach(bundle)
        res.pairs_taught += 1

    for triple, _ in report.missing_triples.most_common(max_triples):
        parts = []
        for t in triple:
            d, _ = _lookup_definition(t, glossary, wiki_cache, use_wiki=use_wiki)
            parts.append(f"{t} {d}" if d else t)
        bundle = f"{format_pair(triple)} three-way context: " + " ".join(parts)
        teach.teach(bundle)
        res.triples_taught += 1

    teach.finalize(top_k=16)
    _save_cache(wiki_cache)
    return res


def teach_full_knowledge(
    teach: TeachStore,
    report: GapReport,
    *,
    glossary: dict[str, str] | None = None,
    use_wiki: bool = True,
    max_absent_wiki: int = 200,
    max_pairs: int = 40,
    max_triples: int = 24,
    max_subwords: int = 20,
) -> TeachGapResult:
    """Full glossary + Wikipedia for absent terms + pair/triple bundles (big win)."""
    glossary = glossary or {}
    wiki_cache = _load_cache()
    res = TeachGapResult(0, 0, 0, 0, 0, 0, 0, 0)
    taught: set[str] = set()

    for term, defn in glossary.items():
        teach.teach_bridge(term, defn)
        taught.add(term)
        res.terms_taught += 1
        res.glossary_hits += 1
        res.bytes_estimate += len(term) + len(defn)

    for term, _ in report.absent_from_corpus.most_common(max_absent_wiki):
        if term in taught or len(term) < 4:
            continue
        defn, src = _lookup_definition(term, glossary, wiki_cache, use_wiki=use_wiki)
        if not defn:
            res.skipped_empty += 1
            continue
        if src == "wiki":
            res.wiki_fetched += 1
        elif src == "glossary":
            res.glossary_hits += 1
        teach.teach_bridge(term, defn)
        taught.add(term)
        res.terms_taught += 1
        res.bytes_estimate += len(term) + len(defn)

    for piece, _ in report.missing_subwords.most_common(max_subwords):
        if piece in taught:
            continue
        defn, _ = _lookup_definition(piece, glossary, wiki_cache, use_wiki=use_wiki)
        if not defn:
            defn = f"{piece} morphological subword in technical vocabulary"
        teach.teach_bridge(piece, defn)
        res.subwords_taught += 1

    for pair, _ in report.missing_pairs.most_common(max_pairs):
        parts = []
        for t in pair:
            d, _ = _lookup_definition(t, glossary, wiki_cache, use_wiki=use_wiki)
            parts.append(f"{t} {d}" if d else t)
            if d and t not in taught:
                teach.teach_bridge(t, d)
                taught.add(t)
        bundle = f"{format_pair(pair)} correlated concepts: " + " ".join(parts)
        teach.teach(bundle)
        res.pairs_taught += 1

    for triple, _ in report.missing_triples.most_common(max_triples):
        parts = []
        for t in triple:
            d, _ = _lookup_definition(t, glossary, wiki_cache, use_wiki=use_wiki)
            parts.append(f"{t} {d}" if d else t)
        bundle = f"{format_pair(triple)} three-way context: " + " ".join(parts)
        teach.teach(bundle)
        res.triples_taught += 1

    teach.finalize(top_k=16)
    _save_cache(wiki_cache)
    return res


def teach_gaps_for_corpus(
    brain,
    corpus_name: str,
    report: GapReport,
    queries: dict[str, str] | None = None,
    **kwargs,
) -> TeachGapResult:
    """Convenience wrapper for MultiCorpusBrain."""
    glossary = load_glossary(corpus_name)
    branch = brain._corpora[corpus_name]
    return teach_gaps(
        branch.teach, report, glossary=glossary, queries=queries, **kwargs,
    )


def _inject_new_defs(branch, new_defs: dict[str, str]) -> int:
    """Merge freshly taught definitions into supervised bridges."""
    if not new_defs or branch.pair_bridges is None:
        return 0
    return inject_glossary_bridges(
        branch.pair_bridges, new_defs, branch.idx, branch.n_docs,
    )


def teach_incremental(
    teach: TeachStore,
    report: GapReport,
    *,
    glossary: dict[str, str] | None = None,
    queries: dict[str, str] | None = None,
    bootstrap_glossary: bool = False,
    use_wiki: bool = True,
    max_absent_wiki: int = 50,
    max_pairs: int = 24,
    max_triples: int = 16,
    max_subwords: int = 12,
    self_texts: dict[str, str] | None = None,
    self_gold_vocab: dict[str, set[str]] | None = None,
    max_self_terms: int = 120,
) -> tuple[TeachGapResult, dict[str, str]]:
    """Teach only terms not already in teach.definitions; returns (result, new_defs).

    When ``self_texts`` is provided, in-corpus gap terms with no external
    definition are taught from their own corpus co-occurrents (self-teaching),
    gold-validated against ``self_gold_vocab`` so corridors only point at
    vocabulary that appears in training gold documents.
    """
    glossary = glossary or {}
    queries = queries or {}
    wiki_cache = _load_cache()
    res = TeachGapResult(0, 0, 0, 0, 0, 0, 0, 0)
    known = set(teach.definitions)
    new_defs: dict[str, str] = {}

    def _self_def(term: str) -> str:
        if not self_texts:
            return ""
        keep = self_gold_vocab.get(term) if self_gold_vocab is not None else None
        if self_gold_vocab is not None and not keep:
            return ""  # no train-gold corridor for this term — skip (safe)
        return corpus_self_definition(term, teach.idx, self_texts, teach.N, keep=keep)

    def _define(term: str) -> tuple[str, str]:
        defn, src = _lookup_definition(term, glossary, wiki_cache, use_wiki=use_wiki)
        if defn:
            return defn, src
        sd = _self_def(term)
        return (sd, "self") if sd else ("", "")

    def _bridge(term: str, defn: str, src: str) -> None:
        if term in known:
            return
        teach.teach_bridge(term, defn)
        known.add(term)
        new_defs[term] = defn
        res.terms_taught += 1
        res.bytes_estimate += len(term) + len(defn)
        if src == "glossary":
            res.glossary_hits += 1
        elif src == "wiki":
            res.wiki_fetched += 1

    if bootstrap_glossary:
        for term, defn in glossary.items():
            _bridge(term, defn, "glossary")
    elif queries and glossary:
        for term in report.glossary_targets(glossary, queries):
            if term in glossary:
                _bridge(term, glossary[term], "glossary")

    for term, _ in report.absent_from_corpus.most_common(max_absent_wiki):
        if term in known or len(term) < 4:
            continue
        defn, src = _lookup_definition(term, glossary, wiki_cache, use_wiki=use_wiki)
        if not defn:
            res.skipped_empty += 1
            continue
        _bridge(term, defn, src or "wiki")

    # Corpus self-teaching: in-corpus rare gap terms learn from their own docs
    if self_texts:
        for term, _ in report.missing_words.most_common(max_self_terms):
            if term in known or len(term) < 4:
                continue
            sd = _self_def(term)
            if sd:
                _bridge(term, sd, "self")

    for piece, _ in report.missing_subwords.most_common(max_subwords):
        if piece in known:
            continue
        defn, _ = _define(piece)
        if not defn:
            defn = f"{piece} morphological subword in technical vocabulary"
        _bridge(piece, defn, "synthetic")
        res.subwords_taught += 1

    for pair, _ in report.missing_pairs.most_common(max_pairs):
        for t in pair:
            d, src = _define(t)
            if d:
                _bridge(t, d, src or "glossary")
        parts = []
        for t in pair:
            d = teach.definitions.get(t) or glossary.get(t, "")
            parts.append(f"{t} {d}" if d else t)
        teach.teach(f"{format_pair(pair)} correlated concepts: " + " ".join(parts))
        res.pairs_taught += 1

    for triple, _ in report.missing_triples.most_common(max_triples):
        for t in triple:
            d, src = _define(t)
            if d:
                _bridge(t, d, src or "glossary")
        parts = []
        for t in triple:
            d = teach.definitions.get(t) or glossary.get(t, "")
            parts.append(f"{t} {d}" if d else t)
        teach.teach(f"{format_pair(triple)} three-way context: " + " ".join(parts))
        res.triples_taught += 1

    teach.finalize(top_k=16)
    _save_cache(wiki_cache)
    return res, new_defs


def _branch_global_texts(branch) -> dict[str, str]:
    return {branch.global_id(k): v for k, v in branch.texts.items()}


def _self_teach_context(branch, queries, train_qrels, self_teach):
    """Build (self_texts, self_gold_vocab) for gold-validated self-teaching."""
    if not self_teach:
        return None, None
    self_texts = _branch_global_texts(branch)
    gold_vocab = None
    if queries and train_qrels:
        gold_vocab = build_gold_vocab_by_term(
            queries, train_qrels, self_texts, global_id=branch.global_id,
        )
    return self_texts, gold_vocab


def teach_full_knowledge_for_corpus(
    brain,
    corpus_name: str,
    report: GapReport,
    *,
    self_teach: bool = False,
    queries: dict[str, str] | None = None,
    train_qrels: dict[str, dict[str, int]] | None = None,
    **kwargs,
) -> TeachGapResult:
    """Full glossary + wiki absent terms + gold-validated self-teaching on one branch."""
    glossary = load_glossary(corpus_name)
    branch = brain._corpora[corpus_name]
    self_texts, gold_vocab = _self_teach_context(
        branch, queries, train_qrels, self_teach,
    )
    result, new_defs = teach_incremental(
        branch.teach, report, glossary=glossary, bootstrap_glossary=True,
        self_texts=self_texts, self_gold_vocab=gold_vocab, **kwargs,
    )
    _inject_new_defs(branch, new_defs)
    return result


def teach_incremental_for_corpus(
    brain,
    corpus_name: str,
    report: GapReport,
    queries: dict[str, str] | None = None,
    *,
    self_teach: bool = False,
    train_qrels: dict[str, dict[str, int]] | None = None,
    **kwargs,
) -> TeachGapResult:
    """One saturation-loop teach step (new terms only), with gold-validated self-teaching."""
    glossary = load_glossary(corpus_name)
    branch = brain._corpora[corpus_name]
    self_texts, gold_vocab = _self_teach_context(
        branch, queries, train_qrels, self_teach,
    )
    result, new_defs = teach_incremental(
        branch.teach, report, glossary=glossary, queries=queries,
        self_texts=self_texts, self_gold_vocab=gold_vocab, **kwargs,
    )
    _inject_new_defs(branch, new_defs)
    return result
