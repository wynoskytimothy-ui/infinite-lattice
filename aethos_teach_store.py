"""Correlation-only teaching — learn concepts without indexing documents."""

from __future__ import annotations

from collections import Counter, defaultdict

from aethos_append_index import words


class TeachStore:
    """Correlation-only memory. teach(text) records rare-term co-occurrence edges
    on EXISTING primes; the text is never indexed as a retrievable document."""

    def __init__(self, idx, N, rare_gate=3.0, max_terms_per_doc=24):
        self.idx = idx
        self.N = N
        self.rare_gate = rare_gate
        self.max_terms = max_terms_per_doc
        self.edges = defaultdict(Counter)
        self.definitions: dict[str, str] = {}
        self._idf = {}
        self.new_symbols = 0
        self.n_taught = 0
        self.tokens_seen = 0

    def _ensure_symbol(self, w: str) -> None:
        """Allocate a prime for w in the shared alphabet (no corpus posting)."""
        if self.idx.token_prime.get(("w", w)) is None:
            self.idx._prime_for(("w", w))
            self.new_symbols += 1

    def idf(self, w):
        v = self._idf.get(w)
        if v is None:
            p = self.idx.token_prime.get(("w", w))
            if p is None:
                v = None
            elif self.idx.df.get(p, 0) == 0:
                v = 99.0
            else:
                v = self.idx._idf(p, self.N)
            self._idf[w] = v
        return v

    def teach(self, text):
        """Ingest a real document as correlations ONLY (not retrievable)."""
        self.n_taught += 1
        toks = list(dict.fromkeys(words(text)))
        self.tokens_seen += len(words(text))
        for w in toks:
            self._ensure_symbol(w)
        self._idf.clear()
        rare = []
        for w in toks:
            iv = self.idf(w)
            if iv is None:
                continue
            if iv >= self.rare_gate or w in self.edges:
                rare.append((w, iv if iv != 99.0 else self.rare_gate + 1))
        rare.sort(key=lambda x: -x[1])
        rare = rare[: self.max_terms]
        for a, ia in rare:
            for b, ib in rare:
                if a != b:
                    self.edges[a][b] += ib
        return len(rare)

    def teach_bridge(
        self,
        term: str,
        definition: str,
        *,
        bridge_gate: float = 2.0,
        per_term: int = 12,
    ) -> int:
        """Star bridge: head term -> rare words in definition (knowledge_bridges shape)."""
        self.n_taught += 1
        self._ensure_symbol(term)
        self.definitions[term] = definition
        self._idf.clear()
        bridges: list[tuple[str, float]] = []
        for w in dict.fromkeys(words(definition)):
            if w == term:
                continue
            p = self.idx.token_prime.get(("w", w))
            if p is None:
                continue
            iv = self.idf(w)
            if iv is not None and iv >= bridge_gate:
                bridges.append((w, iv))
        bridges.sort(key=lambda x: -x[1])
        for w, wt in bridges[:per_term]:
            self.edges[term][w] = max(self.edges[term].get(w, 0.0), wt)
        return len(bridges[:per_term])

    def rewrite_query(
        self,
        query: str,
        *,
        min_idf: float = 2.5,
        max_extra: int = 10,
    ) -> str:
        """Append definition bridge vocabulary (full lexical weight, like knowledge_bridges)."""
        extra: list[str] = []
        seen: set[str] = set()
        for t in set(words(query)):
            defn = self.definitions.get(t)
            if not defn:
                continue
            for w in dict.fromkeys(words(defn)):
                if w == t or w in seen:
                    continue
                p = self.idx.token_prime.get(("w", w))
                if p is None:
                    continue
                iv = self.idf(w)
                if iv is not None and iv >= min_idf:
                    extra.append(w)
                    seen.add(w)
                if len(extra) >= max_extra:
                    break
            if len(extra) >= max_extra:
                break
        if not extra:
            return query
        return query + " " + " ".join(extra[:max_extra])

    def finalize(self, top_k=16):
        """Prune to top-k correlates per term for linear memory."""
        for t, partners in list(self.edges.items()):
            if len(partners) > top_k:
                self.edges[t] = Counter(dict(partners.most_common(top_k)))
        return self

    def expand_scores(self, query, per_term=12, query_term_gate=5.5):
        """Doc weights via taught correlations of rare query terms only."""
        exp = defaultdict(float)
        for qt in set(words(query)):
            partners = self.edges.get(qt)
            if not partners:
                continue
            iv = self.idf(qt)
            if qt not in self.definitions and iv is not None and iv < query_term_gate:
                continue
            for dt, wt in partners.most_common(per_term):
                p = self.idx.token_prime.get(("w", dt))
                if p is None:
                    continue
                for d, tf in self.idx.postings.get(p, {}).items():
                    if d in self.idx.alive:
                        exp[d] += wt * tf / (tf + 1.0)
        return exp

    def memory_bytes(self):
        n_edges = sum(len(v) for v in self.edges.values())
        return n_edges, n_edges * 8
