"""
Natural reading — L7-L9 cross-meaning emerges from co-occurrence alone (no manual tags).

Read enough text -> word graph clusters -> auto category vectors with prime weights.
Query with context: 'apple' + ['phone','chip'] naturally routes to tech cluster.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from aethos_crossmeaning import CategoryVector, MarkovCrossLattice, SemanticStack
from aethos_promotion import LatticeTier, is_stopword
from aethos_tokenize import tokenize_words as tokenize


@dataclass
class CooccurrenceGraph:
    """Word co-occurrence from reading — foundation for natural clusters."""

    pair_count: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    word_count: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    neighbors: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(dict))
    window_count: int = 0

    def observe_window(self, words: list[str]) -> None:
        if len(words) < 2:
            for w in words:
                self.word_count[w] += 1
            return
        self.window_count += 1
        uniq = list(dict.fromkeys(words))
        for w in uniq:
            self.word_count[w] += 1
        for i, a in enumerate(uniq):
            for b in uniq[i + 1 :]:
                pair = tuple(sorted((a, b)))
                self.pair_count[pair] += 1
                self.neighbors[a][b] = self.neighbors[a].get(b, 0) + 1
                self.neighbors[b][a] = self.neighbors[b].get(a, 0) + 1

    def pmi(self, a: str, b: str) -> float:
        import math

        pa = self.word_count.get(a, 0) / max(self.window_count, 1)
        pb = self.word_count.get(b, 0) / max(self.window_count, 1)
        pab = self.pair_count.get(tuple(sorted((a, b))), 0) / max(self.window_count, 1)
        if pa * pb == 0 or pab == 0:
            return 0.0
        return math.log2(pab / (pa * pb))


@dataclass
class NaturalReader(SemanticStack):
    """
    Read raw text only. Clusters and L7-L9 vectors form automatically.
    """

    graph: CooccurrenceGraph = field(default_factory=CooccurrenceGraph)
    word_to_cluster: dict[str, str] = field(default_factory=dict)
    cluster_hubs: dict[str, str] = field(default_factory=dict)  # cluster_id -> hub word
    bridge_words: set[str] = field(default_factory=set)
    bridge_cluster_map: dict[str, set[str]] = field(default_factory=dict)
    min_pair_count: int = 2
    min_vocab_count: int = 1  # include hapax words in cluster union graph
    min_union_cooccur: int = 1  # union threshold (small corpora); strong edges use min_pair_count
    min_pmi: float = 0.5
    min_cluster_score: float = 0.08
    documents_read: int = 0
    rebuild_every: int = 3
    lazy_clusters: bool = False
    clusters_dirty: bool = False
    max_window_tokens: int = 48
    fast_cluster: bool = False

    def apply_scale(self, cfg: object) -> None:
        """Apply aethos_scale.ScaleConfig to reader + registry."""
        self.rebuild_every = getattr(cfg, "rebuild_every", self.rebuild_every)
        self.lazy_clusters = getattr(cfg, "lazy_clusters", self.lazy_clusters)
        self.max_window_tokens = getattr(cfg, "max_window_tokens", self.max_window_tokens)
        self.fast_cluster = getattr(cfg, "fast_cluster", self.fast_cluster)
        reg = self.registry
        reg.max_window_tokens = getattr(cfg, "max_window_tokens", reg.max_window_tokens)
        reg.max_corr_pairs = getattr(cfg, "max_corr_pairs_per_doc", reg.max_corr_pairs)
        reg.skip_stopword_pairs = getattr(cfg, "skip_stopword_pairs", reg.skip_stopword_pairs)
        reg.defer_l2_promotion = getattr(cfg, "defer_l2_promotion", reg.defer_l2_promotion)
        reg.fast_ingest = getattr(cfg, "fast_ingest", reg.fast_ingest)
        reg.max_contexts_per_word = getattr(cfg, "max_contexts_per_word", reg.max_contexts_per_word)

    def ensure_clusters(self) -> None:
        if self.clusters_dirty or not self.cluster_hubs:
            self.discover_clusters()
            self.clusters_dirty = False

    def _is_known(self, word: str) -> bool:
        w = word.lower()
        return w in self.registry.word_counts or w in self.registry.number_counts

    def _active_clusters(self) -> set[str]:
        return set(self.cluster_hubs.keys())

    def _strong_neighbors(self, word: str) -> list[str]:
        w = word.lower()
        out: list[str] = []
        for o, cnt in self.graph.neighbors.get(w, {}).items():
            if cnt >= self.min_pair_count and self.graph.pmi(w, o) >= self.min_pmi:
                if self.graph.word_count.get(o, 0) >= self.min_pair_count:
                    out.append(o)
        return out

    def _pair_linked(self, a: str, b: str) -> bool:
        """True when two words co-occurred or have an L4-L6 correlation edge."""
        a, b = a.lower(), b.lower()
        if self.graph.pair_count.get(tuple(sorted((a, b))), 0) > 0:
            return True
        return tuple(sorted((a, b))) in self.registry.correlations

    def _allowed_clusters(self, word: str) -> set[str] | None:
        """Restrict scoring for bridge / OOV words to neighbor clusters."""
        w = word.lower()
        if w in self.bridge_cluster_map and self.bridge_cluster_map[w]:
            return set(self.bridge_cluster_map[w])
        if w in self.bridge_words:
            cids = {self.word_to_cluster.get(n) for n in self._strong_neighbors(w)}
            cids.discard(None)
            if cids:
                return cids
        return None

    def read(self, *documents: str, finalize: bool = True) -> None:
        """Ingest plain text — no category labels."""
        for doc in documents:
            words = tokenize(doc)
            if not words:
                continue
            if len(words) > self.max_window_tokens:
                graph_words = words[: self.max_window_tokens]
            else:
                graph_words = words
            self.registry.observe_text(doc)
            self.graph.observe_window(graph_words)
            self._observe_sentence_natural(words)
            self.documents_read += 1
            self.clusters_dirty = True
            if not self.lazy_clusters and self.documents_read % self.rebuild_every == 0:
                self.discover_clusters()
                self.clusters_dirty = False
        if finalize or not self.lazy_clusters:
            self.ensure_clusters()

    def read_one(self, doc: str, *, finalize: bool = False) -> None:
        self.read(doc, finalize=finalize)

    def _observe_sentence_natural(self, words: list[str]) -> None:
        """Assign sentence to emergent cluster via current word->cluster map."""
        if not words:
            return
        clusters_in_sentence: dict[str, list[str]] = defaultdict(list)
        for w in words:
            cid = self.word_to_cluster.get(w)
            if cid:
                clusters_in_sentence[cid].append(w)
        if not clusters_in_sentence:
            return  # wait for PMI discover_clusters — no provisional theme_* noise
        for cid, members in clusters_in_sentence.items():
            self.cross.observe(cid, words)

    def discover_clusters(self) -> list[str]:
        """
        Cluster co-occurrence graph -> auto L7-L9 category vectors.
        Bridge words (e.g. apple in phone vs fruit contexts) do NOT merge clusters.
        """
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        vocab = [w for w, c in self.graph.word_count.items() if c >= self.min_vocab_count]
        content = [w for w in vocab if not is_stopword(w)]

        def cooccur_neighbors(w: str) -> list[str]:
            out = []
            for o, cnt in self.graph.neighbors.get(w, {}).items():
                if o in content and cnt >= self.min_union_cooccur:
                    out.append(o)
            return out

        def strong_neighbors(w: str) -> list[str]:
            out = []
            for o, cnt in self.graph.neighbors.get(w, {}).items():
                if o not in content or o == w:
                    continue
                if cnt >= self.min_pair_count and self.graph.pmi(w, o) >= self.min_pmi:
                    out.append(o)
            return out

        def is_bridge(w: str) -> bool:
            """
            True when w co-occurs with words in 2+ disconnected corpus components
            (cross-theme hub, e.g. apple in tech vs food sentences).
            """
            if not self.registry.contexts_differ(w):
                return False
            nbrs = cooccur_neighbors(w)
            if len(nbrs) < 2:
                return False

            # Connected components among content words (stopwords excluded).
            comp_id: dict[str, int] = {}
            comp = 0
            for seed in content:
                if seed == w or seed in comp_id:
                    continue
                stack = [seed]
                comp_id[seed] = comp
                while stack:
                    cur = stack.pop()
                    for o in content:
                        if o == w or o == cur or o in comp_id:
                            continue
                        if self.graph.pair_count.get(tuple(sorted((cur, o))), 0) >= self.min_union_cooccur:
                            comp_id[o] = comp
                            stack.append(o)
                comp += 1

            hit: set[int] = set()
            for n in nbrs:
                if n in comp_id:
                    hit.add(comp_id[n])
            return len(hit) >= 2

        bridges = (
            {w for w in vocab if self.registry.contexts_differ(w)}
            if self.fast_cluster
            else {w for w in vocab if is_bridge(w)}
        )

        for a in vocab:
            if a in bridges or is_stopword(a):
                continue
            for b, cnt in self.graph.neighbors.get(a, {}).items():
                if b not in vocab or b in bridges or is_stopword(b):
                    continue
                if a > b:
                    continue
                if cnt >= self.min_union_cooccur and self.graph.pmi(a, b) >= self.min_pmi:
                    union(a, b)

        groups: dict[str, list[str]] = defaultdict(list)
        for w in vocab:
            if w in bridges or is_stopword(w):
                continue
            groups[find(w)].append(w)

        self.word_to_cluster.clear()
        self.cluster_hubs.clear()
        self.bridge_words = bridges
        self.bridge_cluster_map.clear()
        cluster_ids: list[str] = []

        for _root, members in groups.items():
            if len(members) < 2:
                continue
            hub = max(
                members,
                key=lambda w: sum(
                    self.graph.pair_count.get(tuple(sorted((w, o))), 0)
                    for o in members
                    if o != w
                ),
            )
            cid = f"theme_{hub}"
            cluster_ids.append(cid)
            self.cluster_hubs[cid] = hub
            cat = self.cross.ensure_category(cid)
            for w in members:
                self.word_to_cluster[w] = cid
                tok = self.cross._word_token(self.registry, w)
                if tok:
                    cat.add_token(w, tok.prime, float(self.graph.word_count[w]))

        # Bridge words: attach to each co-occurring neighbor cluster
        for w in bridges:
            attached: set[str] = set()
            for n in cooccur_neighbors(w):
                cid = self.word_to_cluster.get(n)
                if not cid:
                    continue
                attached.add(cid)
                cat = self.cross.categories[cid]
                tok = self.cross._word_token(self.registry, w)
                if tok:
                    cat.add_token(w, tok.prime, float(self.graph.pair_count.get(tuple(sorted((w, n))), 0)))
            if attached:
                self.bridge_cluster_map[w] = attached
                best_n = max(
                    cooccur_neighbors(w),
                    key=lambda n: self.graph.pair_count.get(tuple(sorted((w, n))), 0),
                )
                best_cid = self.word_to_cluster.get(best_n)
                if best_cid:
                    self.word_to_cluster[w] = best_cid

        self._attach_orphans_and_bridges(vocab, bridges)

        self.cross.prune_active(set(self.cluster_hubs.keys()))

        return cluster_ids

    def _attach_orphans_and_bridges(self, vocab: list[str], bridges: set[str]) -> None:
        """Link under-connected or polysemous words to neighbor clusters."""
        for w in vocab:
            if w in bridges or is_stopword(w):
                continue
            neighbor_clusters: dict[str, int] = {}
            for n in self.graph.word_count:
                if n == w:
                    continue
                cnt = self.graph.pair_count.get(tuple(sorted((w, n))), 0)
                if cnt <= 0:
                    continue
                cid = self.word_to_cluster.get(n)
                if cid:
                    neighbor_clusters[cid] = neighbor_clusters.get(cid, 0) + cnt

            if not neighbor_clusters:
                continue

            if self.registry.contexts_differ(w) and len(neighbor_clusters) >= 2:
                self.bridge_words.add(w)
                self.bridge_cluster_map[w] = set(neighbor_clusters.keys())
                best_cid = max(neighbor_clusters.items(), key=lambda x: x[1])[0]
                self.word_to_cluster[w] = best_cid
            elif w not in self.word_to_cluster:
                best_cid = max(neighbor_clusters.items(), key=lambda x: x[1])[0]
                self.word_to_cluster[w] = best_cid

    def infer_cluster(self, word: str, context: Iterable[str] | None = None) -> tuple[str, float]:
        """
        Natural meaning from context (Markovian).
        'apple' with ['phone','chip'] -> theme_phone not theme_fruit.
        """
        self.ensure_clusters()
        w = word.lower()
        ctx = [c.lower() for c in (context or []) if c]

        active = self._active_clusters()
        if not active:
            return ("", 0.0)

        # Out-of-vocabulary: no corpus evidence for this word.
        if not self._is_known(w):
            if not ctx:
                return ("", 0.0)
            known_ctx = [c for c in ctx if self._is_known(c)]
            if not known_ctx:
                return ("", 0.0)
            votes: dict[str, float] = {}
            for c in known_ctx:
                cid = self.word_to_cluster.get(c)
                if cid and cid in active:
                    votes[cid] = votes.get(cid, 0.0) + 1.0
            if not votes:
                return ("", 0.0)
            best_cid, best_v = max(votes.items(), key=lambda x: x[1])
            return (best_cid, best_v * 0.5)

        if ctx and not any(self._pair_linked(w, c) for c in ctx):
            ctx = []  # unrelated context — do not reroute (e.g. phone + fruit,pie)

        # Known word, no context: use assigned cluster directly.
        if not ctx and w in self.word_to_cluster:
            cid = self.word_to_cluster[w]
            if cid in active:
                score = max(self.cross.belonging_score(w, cid), 0.15)
                return (cid, score)

        allowed = self._allowed_clusters(w)
        pool = allowed if allowed else active

        scores: dict[str, float] = {}
        for cid in pool:
            if cid not in self.cross.categories:
                continue
            if ctx:
                score = self.cross.phrase_belonging([w] + ctx, cid)
            else:
                score = self.cross.belonging_score(w, cid)
            scores[cid] = score

        if ctx:
            for c in ctx:
                if not self._pair_linked(w, c):
                    continue
                cid = self.word_to_cluster.get(c)
                if cid and cid in pool:
                    scores[cid] = scores.get(cid, 0.0) + 1.0
            for c in ctx:
                if not self._pair_linked(w, c):
                    continue
                for cid in pool:
                    if cid not in self.cross.categories:
                        continue
                    scores[cid] = scores.get(cid, 0.0) + self.cross.phrase_belonging([w, c], cid) * 0.5

        if not scores:
            if w in self.word_to_cluster and self.word_to_cluster[w] in active:
                return (self.word_to_cluster[w], 0.1)
            return ("", 0.0)

        best_cid, best_score = max(scores.items(), key=lambda x: x[1])
        if best_score < self.min_cluster_score:
            mapped = self.word_to_cluster.get(w)
            if mapped and mapped in active:
                return (mapped, max(best_score, 0.1))
            return ("", best_score)
        return (best_cid, best_score)

    def related_in_cluster(self, cluster_id: str, limit: int = 10) -> list[tuple[str, float]]:
        return self.cross.all_in_category(cluster_id)[:limit]

    def explain_natural(self, word: str, context: list[str] | None = None) -> str:
        cid, score = self.infer_cluster(word, context)
        hub = self.cluster_hubs.get(cid, cid)
        lines = [
            f"Natural inference for {word!r}",
            f"  context: {context or []}",
            f"  emergent cluster: {cid} (hub={hub!r})",
            f"  confidence: {score:.4f}",
        ]
        if cid:
            lines.append(f"  related in cluster:")
            for w, wt in self.related_in_cluster(cid, 8):
                lines.append(f"    {w:12s} {wt:.4f}")
            cat = self.cross.categories.get(cid)
            if cat:
                lines.append(f"  L7-L9 primes (top): {cat.top_primes(5)}")
        return "\n".join(lines)


def demo() -> None:
    reader = NaturalReader(rebuild_every=2)

    corpus = [
        "apple phone chip processor software hardware technical",
        "apple phone software update technical support",
        "samsung phone chip technical engineering tablet",
        "phone computer technical hardware software network",
        "apple phone technical camera screen battery",
        "phone technical modem wireless chip",
        "apple fruit pie recipe orchard baking",
        "apple fruit fresh salad orchard cinnamon",
        "banana fruit pie recipe dessert sweet",
        "apple orchard fruit harvest pie baking",
        "fruit salad fresh apple banana dessert",
    ]

    print("=" * 60)
    print("NATURAL READING — no tags, clusters emerge from co-occurrence")
    print("=" * 60)
    print(f"\n  Reading {len(corpus)} documents (untagged)...\n")
    reader.read(*corpus)

    print(f"  Documents read:     {reader.documents_read}")
    print(f"  Emergent clusters:  {list(reader.cluster_hubs.keys())}")
    print(f"  Hub words:          {reader.cluster_hubs}")

    print("\n--- Auto L7-L9 vectors (discovered, not labeled) ---")
    for cid, hub in reader.cluster_hubs.items():
        cat = reader.cross.categories[cid]
        print(f"\n  {cid} (hub={hub!r}):")
        print(f"    top words:  {cat.top_words(6)}")
        print(f"    top primes: {cat.top_primes(4)}")

    print("\n--- apple + phone context -> tech cluster naturally ---")
    print(reader.explain_natural("apple", ["phone", "chip", "software"]))

    print("\n--- apple + fruit context -> food cluster naturally ---")
    print(reader.explain_natural("apple", ["fruit", "pie", "orchard"]))

    c1, s1 = reader.infer_cluster("apple", ["phone", "chip"])
    c2, s2 = reader.infer_cluster("apple", ["fruit", "pie"])
    print(f"\n  apple|phone,chip  -> {c1} ({s1:.4f})")
    print(f"  apple|fruit,pie   -> {c2} ({s2:.4f})")

    if c1 and c2:
        print(f"  disambiguation:    {'OK' if c1 != c2 else 'same cluster'}")


if __name__ == "__main__":
    demo()
