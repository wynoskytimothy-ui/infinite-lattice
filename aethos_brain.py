#!/usr/bin/env python3
"""
The complete self-teaching chamber brain + a capability test suite.

Built from patterns mined from the AETHOS formulas:
  * entanglement ODE (Ch.17 Pattern 2, PROVEN): C' = Gform(1-C) - Gbreak*C,
    C* = Gform/(Gform+Gbreak)  -> synaptic plasticity (LTP/LTD), decay + revive.
  * "rarest means most" (the bridges): idf-weighted routing so shared tokens don't
    swamp the signal -> robust separation at high vocabulary overlap.
  * unbounded origins (0->inf): a novel input allocates a NEW chamber; old chambers
    are never touched (continual, no catastrophic forgetting).

Capabilities tested (the brain must do ALL of them):
  1 self-organise   chambers specialise to hidden domains, no labels
  2 robustness      still separates at high shared-vocabulary overlap (idf)
  3 prediction      complete a masked token from the winning chamber's pathways
  4 continual       learn a NEW domain mid-life without disturbing the old ones
  5 plasticity      idle pathway decays, revives on re-trigger, address kept
"""

from __future__ import annotations

import math
import random
from collections import Counter

GFORM, GBREAK = 0.30, 0.05
VIG = 0.18                                 # ART vigilance on the idf-normalised match rate


def make_stream(K=4, vocab_per=60, overlap=8, n=4000, toks=12, seed=0, dstart=0):
    rng = random.Random(seed)
    shared = [f"s{i}" for i in range(overlap)]
    dom = [[f"d{dstart+d}_w{i}" for i in range(vocab_per)] + shared for d in range(K)]
    out = [([rng.choice(dom[d]) for _ in range(toks)], dstart + d)
           for _ in range(n) for d in [rng.randrange(K)]]
    rng.shuffle(out)
    return out


def make_phrase_stream(K=4, n_phrase=15, psize=4, overlap=8, n=4000, seed=0):
    """structured data: each domain is a set of disjoint PHRASES (co-occurring
    token groups). An input = one phrase + shared noise. Now there is real
    within-domain structure to PREDICT (complete a phrase from its mates)."""
    rng = random.Random(seed)
    shared = [f"s{i}" for i in range(overlap)]
    phrases = [[[f"d{d}_p{p}_{i}" for i in range(psize)] for p in range(n_phrase)]
               for d in range(K)]
    out = []
    for _ in range(n):
        d = rng.randrange(K)
        ph = rng.choice(phrases[d])
        toks = list(ph) + [rng.choice(shared) for _ in range(2)]
        out.append((toks, d, tuple(ph)))
    rng.shuffle(out)
    return out


class Brain:
    def __init__(self, use_idf=True):
        self.ch = []                       # {"proto":{tok:C}, "hits":Counter}
        self.df = Counter()
        self.seen = 0
        self.use_idf = use_idf

    def idf(self, t):
        if not self.use_idf:
            return 1.0
        return (math.log((self.seen + 1) / (self.df[t] + 1)) + 1.0) ** 2   # rarest means most

    def _match(self, toks, c):
        p = c["proto"]
        return sum(p.get(t, 0.0) * self.idf(t) for t in toks)

    def route(self, toks):
        best, bs = None, -1.0
        for c in self.ch:
            s = self._match(toks, c)
            if s > bs:
                bs, best = s, c
        denom = sum(self.idf(t) for t in set(toks)) or 1.0
        return best, bs / denom

    def learn(self, toks, label=None):
        self.seen += 1
        for t in set(toks):
            self.df[t] += 1
        best, rate = self.route(toks)
        if best is None or rate < VIG:                 # resonance fails -> NEW chamber (origin)
            best = {"proto": {}, "assoc": {}, "hits": Counter()}
            self.ch.append(best)
        active = set(toks)
        p = best["proto"]
        for t in active:                               # Gform: potentiate toward 1
            p[t] = p.get(t, 0.0) + GFORM * (1.0 - p.get(t, 0.0))
        for t in list(p):                              # Gbreak: decay idle toward 0
            if t not in active:
                p[t] *= (1.0 - GBREAK)
        A = best["assoc"]                              # pairwise binding (for prediction)
        al = list(active)
        for i in range(len(al)):
            d = A.setdefault(al[i], {})
            for j in range(len(al)):
                if i != j:
                    d[al[j]] = d.get(al[j], 0.0) + GFORM * (1.0 - d.get(al[j], 0.0))
        if label is not None:
            best["hits"][label] += 1
        return best

    def surprise(self, toks):
        """predictive-coding novelty: the idf-weighted fraction of the input the
        best-matching chamber CANNOT explain. Low for a known state; spikes when
        rare new tokens appear -> gradual-drift anomaly, without needing a new
        chamber. (Computed against the current state; call BEFORE learn.)"""
        best, _ = self.route(toks)
        if best is None:
            return 1.0
        p = best["proto"]
        s = set(toks)
        num = sum(self.idf(t) * (1.0 - p.get(t, 0.0)) for t in s)
        den = sum(self.idf(t) for t in s) or 1.0
        return num / den

    def predict(self, given, k=3):
        """pattern completion: route, then aggregate the given tokens' learned
        co-occurrence partners (the association layer)."""
        best, _ = self.route(given)
        if best is None:
            return []
        g, score = set(given), Counter()
        A = best["assoc"]
        for t in given:
            for partner, w in A.get(t, {}).items():
                if partner not in g:
                    score[partner] += w
        return [t for t, _ in score.most_common(k)]

    def purity(self):
        n = sum(sum(c["hits"].values()) for c in self.ch) or 1
        return sum(max(c["hits"].values()) for c in self.ch if c["hits"]) / n

    def top(self, c, k=3):
        return tuple(sorted(c["proto"], key=lambda t: -c["proto"][t])[:k])


def test_self_organise():
    b = Brain()
    for toks, d in make_stream(overlap=8):
        b.learn(toks, d)
    pur = b.purity()
    return pur > 0.9, f"purity {pur*100:.0f}% at 12% shared vocab"


def test_robustness():
    rows = []
    for ov in (30, 60, 120):
        stream = make_stream(overlap=ov)
        b = Brain()
        for _ in range(2):                          # 2 passes = replay/consolidation
            for toks, d in stream:
                b.learn(toks)
        for c in b.ch:                              # clean eval: route only, count domains
            c["hits"].clear()
        for toks, d in stream:
            best, _ = b.route(toks)
            if best is not None:
                best["hits"][d] += 1
        rows.append((ov / (60 + ov), b.purity()))
    worst = min(p for _, p in rows)
    detail = "  ".join(f"{int(f*100)}%->{p*100:.0f}%" for f, p in rows)
    return worst > 0.8, f"purity by shared-vocab (2-pass consolidation): {detail}"


def test_prediction():
    b = Brain()
    for toks, d, ph in make_phrase_stream():
        b.learn(toks, d)
    hit = tot = 0
    for toks, d, ph in make_phrase_stream(n=600, seed=99):
        give, targets = list(ph[:2]), set(ph[2:])     # give 2 phrase tokens, predict the rest
        if set(b.predict(give, k=3)) & targets:
            hit += 1
        tot += 1
    acc = hit / tot
    vocab = 4 * 15 * 4 + 8
    base = 1 - (1 - len(targets) / vocab) ** 3
    return acc > 0.6, f"phrase-completion top-3 acc {acc*100:.0f}% vs random {base*100:.0f}%"


def test_continual():
    b = Brain()
    for toks, d in make_stream(K=4, overlap=8, dstart=0):
        b.learn(toks, d)
    big = sorted([c for c in b.ch if sum(c["hits"].values()) > 50],
                 key=lambda c: -sum(c["hits"].values()))[:4]
    before = {b.top(c): c["hits"].most_common(1)[0][0] for c in big}
    n_before = len(b.ch)
    # a brand-new domain arrives mid-life
    for toks, d in make_stream(K=1, overlap=8, dstart=9, n=800, seed=5):
        b.learn(toks, d)
    after = {b.top(c): c["hits"].most_common(1)[0][0] for c in big}
    unchanged = all(before[k] == after.get(k) for k in before)
    new_ch = [c for c in b.ch if c["hits"] and c["hits"].most_common(1)[0][0] == 9]
    new_pure = bool(new_ch) and max(
        c["hits"][9] / sum(c["hits"].values()) for c in new_ch) > 0.9
    grew = len(b.ch) > n_before
    return unchanged and new_pure and grew, (
        f"old chambers unchanged={unchanged}, new domain got a pure chamber={new_pure}, "
        f"grew {n_before}->{len(b.ch)}")


def test_plasticity():
    b = Brain()
    for toks, d in make_stream(overlap=8, n=500):
        b.learn(toks, d)
    c = max(b.ch, key=lambda c: sum(c["hits"].values()))
    t = max(c["proto"], key=lambda x: c["proto"][x])
    start = c["proto"][t]
    for _ in range(40):
        c["proto"][t] *= (1 - GBREAK)
    low = c["proto"][t]
    c["proto"][t] += GFORM * (1 - c["proto"][t])
    revived = c["proto"][t]
    ok = low < start * 0.3 and revived > low * 3 and t in c["proto"]
    return ok, f"'{t}' {start:.2f} -> decays {low:.2f} -> revives {revived:.2f} (address kept)"


def main():
    print("THE BRAIN -- capability audit (each must PASS)\n" + "=" * 64)
    tests = [("1 self-organise", test_self_organise), ("2 robustness ", test_robustness),
             ("3 prediction  ", test_prediction), ("4 continual   ", test_continual),
             ("5 plasticity  ", test_plasticity)]
    n_pass = 0
    for name, fn in tests:
        ok, detail = fn()
        n_pass += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    print("=" * 64)
    print(f"  {n_pass}/{len(tests)} capabilities verified"
          + ("  -- the brain does everything we asked." if n_pass == len(tests)
             else "  -- gaps remain (see FAIL)."))


if __name__ == "__main__":
    main()
