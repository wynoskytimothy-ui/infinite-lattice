"""
aethos_sharded_index.py - distributed-exact sharding over the lattice index.

Docs are hash-routed to K independent AppendOnlyLatticeIndex shards that SHARE
one vocabulary (a token gets the same prime everywhere). Global stats (N, df,
avgdl) are maintained across shards and broadcast at query time, so each shard
scores its own docs EXACTLY as a single index would - a fan-out query + top-k
merge is identical to querying one big index, while each shard stays sub-ms and
lossless.

This is how the engine keeps SOTA accuracy AND sub-ms at unbounded N: add shards
(machines) as the corpus grows. The append-only design needs no coordination
beyond the shared term dictionary + global counts; each shard still appends in
O(1) and can be finalized/queried independently (in parallel, in production).
"""

from __future__ import annotations

import hashlib
import heapq
from collections import defaultdict

try:
    import numpy as np
except ImportError:                       # pragma: no cover
    np = None

from aethos_append_index import AppendOnlyLatticeIndex
from core.primes import chain_primes


class ShardedIndex:
    """K lattice shards with a shared vocabulary and global scoring stats."""

    def __init__(self, n_shards=4, **index_kwargs):
        self.k = max(1, n_shards)
        self._tp: dict = {}                            # shared (view,token) -> prime
        primes = chain_primes(200000)                  # shared prime pool
        self.shards = [
            AppendOnlyLatticeIndex(token_prime=self._tp, _primes=primes, **index_kwargs)
            for _ in range(self.k)
        ]
        self.gN = 0
        self.g_total = 0.0
        self.gdf = None
        self._ready = False

    def _route(self, doc_id):
        h = hashlib.blake2b(str(doc_id).encode(), digest_size=8).hexdigest()
        return int(h, 16) % self.k

    # ---- ingest: O(1) append into the routed shard ----
    def add(self, doc_id, text):
        self.shards[self._route(doc_id)].add(doc_id, text)
        self._ready = False

    # ---- finalize: gather global stats, build each shard with them ----
    def finalize(self, champion_m=None):
        self.gN = sum(len(s.alive) for s in self.shards)
        self.g_total = sum(s._total_len for s in self.shards)
        gavg = self.g_total / max(1, self.gN)
        gdf = defaultdict(int)
        for s in self.shards:
            for p, d in s.df.items():
                if d:
                    gdf[p] += d
        self.gdf = gdf
        for s in self.shards:
            if s.alive:
                s.finalize(champion_m=champion_m, global_avgdl=gavg)
        self._ready = True
        return self

    # ---- query: fan out with GLOBAL stats, merge top-k (exact) ----
    def search(self, query, k=10):
        if np is None:
            raise RuntimeError("ShardedIndex.search requires numpy")
        cands = []
        for s in self.shards:
            if not s._dense_ready:
                continue
            scores = s._dense_score_array(query, gN=self.gN, gdf=self.gdf)
            docs = s._d_docs
            kk = min(k, len(docs))
            if kk == 0:
                continue
            part = np.argpartition(scores, -kk)[-kk:]
            for i in part:
                if scores[i] > 0.0:
                    cands.append((float(scores[i]), docs[i]))
        return [d for _, d in heapq.nlargest(k, cands)]

    def stats(self):
        return {
            "shards": self.k,
            "live_docs": self.gN,
            "vocab": len(self._tp),
            "per_shard_docs": [len(s.alive) for s in self.shards],
        }
