"""STAGE 1 audit: gear ablation (word vs word+tri+prefix) on scifact.

Faithful ablation: subclass AppendOnlyLatticeIndex and override _multiview so a
gear is GENUINELY absent from the postings (not just down-weighted). Measures
recall@k and nDCG@10 on held-out scifact test queries. Also a morphology/typo
stress test lives in _audit_stage1_morph.py.
"""
import sys, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words, _GW, _GT, _GP
from scripts.bench_supervised_bridges import load, ndcg10


class GearIndex(AppendOnlyLatticeIndex):
    """AppendOnlyLatticeIndex with a configurable subset of {word,tri,prefix} gears."""
    def set_gears(self, gears):
        self._use_word = "word" in gears
        self._use_tri = "tri" in gears
        self._use_prefix = "prefix" in gears
        return self

    def _multiview(self, text, positional=False):
        bag = {}
        bget = bag.get
        pos_head, pos_boost = self.pos_head, self.pos_boost
        for i, w in enumerate(words(text)):
            pos_w = pos_boost if (positional and i < pos_head) else 1.0
            if self._use_word:
                kw = ("w", w)
                bag[kw] = bget(kw, 0.0) + _GW * pos_w
            if self._use_tri:
                p = "^" + w + "$"
                for j in range(len(p) - 2):
                    k = ("3", p[j:j + 3])
                    bag[k] = bget(k, 0.0) + _GT
            if self._use_prefix:
                pk = ("p", w[:4])
                bag[pk] = bget(pk, 0.0) + _GP
        return bag


def build_index(corpus, gears):
    idx = GearIndex().set_gears(gears)
    for d, txt in corpus.items():
        idx.add(d, txt)
    return idx


def recall_at_k(ranked, rel, k):
    rels = {d for d, s in rel.items() if s > 0}
    if not rels:
        return None
    return len(set(ranked[:k]) & rels) / len(rels)


def main():
    corpus, queries, train_q, test_q = load("scifact")
    test_ids = [q for q in test_q if q in queries and any(v > 0 for v in test_q[q].values())]
    print(f"scifact: {len(corpus)} docs, {len(test_ids)} test queries with positive rels")

    configs = {
        "word":            ["word"],
        "word+tri":        ["word", "tri"],
        "word+prefix":     ["word", "prefix"],
        "word+tri+prefix": ["word", "tri", "prefix"],
        "tri-only":        ["tri"],
    }
    ks = [10, 20, 100]
    results = {}
    for name, gears in configs.items():
        idx = build_index(corpus, gears)
        rec = {k: [] for k in ks}
        ndcgs = []
        t0 = time.time()
        for qid in test_ids:
            ranked = idx.search(queries[qid], k=max(ks))
            ndcgs.append(ndcg10(ranked, test_q[qid]))
            for k in ks:
                r = recall_at_k(ranked, test_q[qid], k)
                if r is not None:
                    rec[k].append(r)
        dt = time.time() - t0
        results[name] = {
            "ndcg10": sum(ndcgs) / len(ndcgs),
            **{f"recall@{k}": sum(rec[k]) / len(rec[k]) for k in ks},
            "vocab": len(idx.token_prime),
            "qps": len(test_ids) / dt,
        }
        r = results[name]
        print(f"  {name:18s} nDCG@10={r['ndcg10']:.4f}  R@10={r['recall@10']:.4f}  "
              f"R@20={r['recall@20']:.4f}  R@100={r['recall@100']:.4f}  "
              f"vocab={r['vocab']:6d}  {r['qps']:.0f}q/s")
    return results


if __name__ == "__main__":
    main()
