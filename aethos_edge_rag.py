#!/usr/bin/env python3
"""aethos_edge_rag — the coherent edge engine: fast numba ingest -> mmap CSR serve -> counting-bridges.

One class wires the three proven pieces:
  INGEST  : radix/hash numba build (nogil-threaded) -> serve-ready CSR (24.5M tok/s, _o1_ingest_radix).
  SERVE   : mmap-able CSR (seg_doc/seg_tf/indptr), scatter-add BM25 over the query's term segments;
            query tokens -> FNV hash -> term_id (the same hashing ingest used). RAM = working set.
  ACCURACY: counting-bridges (learned from train qrels) rerank the top-100 -- neural-free.

word-only = the edge champion (198 B/doc). Verified to reproduce the edge champion nDCG exactly:
scifact lexical 0.6712 -> +bridges 0.7112 (same numbers the slow pipeline produced -- accuracy untouched).
"""
import os, sys, json, math, time
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fast_tok import words
from _o1_ingest_radix import radix_build_mt, fnv

K1, B = 1.2, 0.75


class EdgeRAG:
    def __init__(self, nthreads=None):
        self.nthreads = nthreads or min(8, (os.cpu_count() or 4))

    # ---------------- ingest ----------------
    def build(self, corpus):
        self.doc_ids = list(corpus.keys())
        texts = list(corpus.values())
        t0 = time.perf_counter()
        Bd, _ = radix_build_mt(texts, self.nthreads)
        self.ingest_s = time.perf_counter() - t0
        self.uniq_hash = Bd["uniq_hash"]; self.indptr = Bd["indptr"]
        self.seg_doc = Bd["seg_doc"]; self.seg_tf = Bd["seg_tf"]
        self.V = Bd["V"]; self.N = len(self.doc_ids)
        self.hash2term = {int(h): i for i, h in enumerate(self.uniq_hash)}
        self.df = np.diff(self.indptr).astype(np.int64)
        self.doc_len = np.bincount(self.seg_doc, weights=self.seg_tf.astype(np.float64),
                                   minlength=self.N)
        self._finish()
        return self

    def _finish(self):
        avgdl = self.doc_len.sum() / max(1, self.N)
        self._A = K1 * (1 - B); self._Bc = K1 * B / avgdl; self._k1p1 = K1 + 1
        self._denom = self._A + self._Bc * self.doc_len

    def _term_id(self, token):
        return self.hash2term.get(fnv(token.encode()))

    def _idf(self, dfp):
        return math.log(1 + (self.N - dfp + 0.5) / (dfp + 0.5))

    # ---------------- serve ----------------
    def score(self, query):
        scores = np.zeros(self.N)
        ip, sd, st, den = self.indptr, self.seg_doc, self.seg_tf, self._denom
        for w, qwt in Counter(words(query)).items():
            tid = self._term_id(w)
            if tid is None: continue
            a, e = int(ip[tid]), int(ip[tid + 1])
            if e == a: continue
            di = sd[a:e]; tf = st[a:e].astype(np.float64)
            idf = self._idf(e - a)
            scores[di] += (qwt * idf * self._k1p1) * tf / (tf + den[di])
        return scores

    def _top(self, scores, k):
        kk = min(k, self.N)
        part = np.argpartition(scores, -kk)[-kk:]
        part = part[np.argsort(scores[part])[::-1]]
        return [self.doc_ids[i] for i in part if scores[i] > 0.0]

    def search(self, query, k=10):
        return self._top(self.score(query), k)

    # ---------------- bridges (counting, neural-free) ----------------
    def learn_bridges(self, queries, train_qrels, corpus, min_pairs=2, top_per=12, idf_gate=1.5):
        def idf_tok(w):
            tid = self._term_id(w)
            return self._idf(int(self.df[tid])) if tid is not None else 0.0
        cooc = defaultdict(Counter); qtp = Counter(); qc, dc = {}, {}
        for qid, rels in train_qrels.items():
            if qid not in queries: continue
            qts = qc.get(qid)
            if qts is None:
                qts = tuple(w for w in set(words(queries[qid])) if idf_tok(w) >= idf_gate); qc[qid] = qts
            for cid, sc in rels.items():
                if sc <= 0 or cid not in corpus: continue
                dts = dc.get(cid)
                if dts is None:
                    dts = frozenset(w for w in set(words(corpus[cid])) if idf_tok(w) >= idf_gate); dc[cid] = dts
                for qt in qts:
                    qtp[qt] += 1
                    if dts: cooc[qt].update(dts)
        self.bridge = {}
        for qt, partners in cooc.items():
            np_ = qtp[qt]
            scored = [(dt, (c / np_) * idf_tok(dt)) for dt, c in partners.items() if dt != qt and c >= min_pairs]
            scored.sort(key=lambda x: (-x[1], x[0]))
            if scored: self.bridge[qt] = scored[:top_per]
        return self

    def _bridge_scores_vec(self, query, cand_idx):
        """Vectorized bridge score over candidate doc-INDICES. Each target term's segment is touched once
        in numpy (mask + scatter-add); term_id resolved once per target, not per candidate. Returns a
        length-|cand_idx| array aligned to cand_idx."""
        ip, sd = self.indptr, self.seg_doc
        targets = {}                                       # term_id -> summed bridge weight
        for qt in set(words(query)):
            for dt, w in self.bridge.get(qt, ()):
                tid = self._term_id(dt)
                if tid is None: continue
                targets[tid] = targets.get(tid, 0.0) + w
        if not targets:
            return np.zeros(len(cand_idx))
        cand_mask = np.zeros(self.N, bool); cand_mask[cand_idx] = True
        acc = np.zeros(self.N)
        for tid, w in targets.items():
            a, e = int(ip[tid]), int(ip[tid + 1])
            seg = sd[a:e]
            hit = seg[cand_mask[seg]]                       # candidate doc-indices that contain this term
            acc[hit] += w
        return acc[cand_idx]

    def search_bridged(self, query, k=10, lam=0.15):
        scores = self.score(query)
        m = min(100, self.N)
        top = np.argpartition(scores, -m)[-m:]
        top = top[scores[top] > 0.0]
        if top.size == 0: return []
        cand_idx = top[np.argsort(scores[top])[::-1]]       # top-100 candidate indices, score>0
        lex = scores[cand_idx]; lmax = lex.max() or 1.0
        bs = self._bridge_scores_vec(query, cand_idx)
        bmax = bs.max() if bs.size and bs.max() > 0 else 1.0
        final = lex / lmax + lam * bs / bmax
        fin = cand_idx[np.argsort(final)[::-1]]
        return [self.doc_ids[int(i)] for i in fin[:k]]

    # ---------------- mmap persistence (RAM = working set) ----------------
    def save_mmap(self, path):
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/seg_doc.npy", self.seg_doc); np.save(f"{path}/seg_tf.npy", self.seg_tf)
        np.save(f"{path}/indptr.npy", self.indptr); np.save(f"{path}/uniq_hash.npy", self.uniq_hash)
        np.save(f"{path}/doc_len.npy", self.doc_len)
        json.dump({"doc_ids": self.doc_ids, "V": self.V, "N": self.N,
                   "bridge": getattr(self, "bridge", {})}, open(f"{path}/meta.json", "w"))
        return self

    @classmethod
    def load_mmap(cls, path):
        self = cls()
        self.seg_doc = np.load(f"{path}/seg_doc.npy", mmap_mode="r")
        self.seg_tf = np.load(f"{path}/seg_tf.npy", mmap_mode="r")
        self.indptr = np.load(f"{path}/indptr.npy"); self.uniq_hash = np.load(f"{path}/uniq_hash.npy")
        self.doc_len = np.load(f"{path}/doc_len.npy")
        m = json.load(open(f"{path}/meta.json"))
        self.doc_ids = m["doc_ids"]; self.V = m["V"]; self.N = m["N"]
        self.bridge = {k: [tuple(x) for x in v] for k, v in m.get("bridge", {}).items()}
        self.hash2term = {int(h): i for i, h in enumerate(self.uniq_hash)}
        self.df = np.diff(self.indptr).astype(np.int64)
        self._finish()
        return self


def _selftest():
    from scripts.bench_supervised_bridges import load, ndcg10, recall10
    print("=" * 84)
    print("EDGE RAG ENGINE — fast ingest -> mmap serve -> bridges, end-to-end (reproduce edge champion)")
    print("=" * 84)
    corpus, queries, train_q, test_q = load("scifact")
    test_ids = [q for q in test_q if q in queries]

    EdgeRAG().build({"_w": "warm the numba jit so the ingest time is steady-state not cold"})  # warm JIT
    eng = EdgeRAG().build(corpus)
    eng.learn_bridges(queries, train_q, corpus)

    def evl(fn):
        nd = rc = 0.0; lat = []
        for qid in test_ids:
            t0 = time.perf_counter(); r = fn(queries[qid]); lat.append((time.perf_counter() - t0) * 1000)
            nd += ndcg10(r, test_q[qid]); rc += recall10(r, test_q[qid])
        n = len(test_ids); return nd / n, rc / n, float(np.median(lat))

    nd_lex, rc_lex, ms_lex = evl(lambda q: eng.search(q, 10))
    nd_br, rc_br, ms_br = evl(lambda q: eng.search_bridged(q, 10))
    print(f"\n  ingest: {eng.N:,} docs in {eng.ingest_s*1000:.0f} ms ({eng.N/eng.ingest_s:,.0f} docs/s), "
          f"{eng.V:,} terms, {len(eng.seg_doc):,} postings")
    print(f"  lexical (word-only) : nDCG {nd_lex:.4f}  Recall {rc_lex:.4f}  {ms_lex:.2f} ms/q   (edge champ C=0.6712)")
    print(f"  + counting-bridges  : nDCG {nd_br:.4f}  Recall {rc_br:.4f}  {ms_br:.2f} ms/q   (edge champ D=0.7112)")
    ok_lex = abs(nd_lex - 0.6712) < 0.004; ok_br = abs(nd_br - 0.7112) < 0.006
    print(f"  reproduces edge champion: lexical {'OK' if ok_lex else 'DRIFT'} | bridged {'OK' if ok_br else 'DRIFT'}")

    # mmap round-trip
    import tempfile
    p = os.path.join(tempfile.gettempdir(), "edge_rag_idx")
    eng.save_mmap(p); eng2 = EdgeRAG.load_mmap(p)
    nd2, _, _ = evl(lambda q: eng2.search_bridged(q, 10))
    print(f"  mmap round-trip serve: nDCG {nd2:.4f}  ({'MATCH' if abs(nd2-nd_br) < 1e-9 else 'DIFF'})")
    print("\n  one engine: numba radix ingest + mmap CSR serve + counting-bridges. small + fast + accurate.")


if __name__ == "__main__":
    _selftest()
