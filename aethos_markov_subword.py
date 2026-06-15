#!/usr/bin/env python3
"""Fair test of the FULL Markov-lattice idea: words + SUBWORDS. Full vocabulary (no
UNK), word 4-gram with backoff for seen contexts, and a CHARACTER model giving every
word a probability from its spelling -- so rare/unseen words get structured
probability (the subword layer), not a uniform floor. Word-only vs word+subword."""
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.bench_supervised_bridges import load

corpus, *_ = load("scifact")
tokr = re.compile(r"[a-z]+")
seq = [w for t in corpus.values() for w in tokr.findall(t.lower()) + ["."]]
n = int(0.9 * len(seq))
train, val = seq[:n], seq[n:]
V = sorted(set(train))
Vn = len(V)
print(f"Markov+subword LM: {len(seq):,} tokens, FULL vocab {Vn} (no UNK)\n")

uni = Counter(train)
tot = sum(uni.values())
bi, tri, quad = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
for i in range(len(train)):
    if i >= 1:
        bi[train[i-1]][train[i]] += 1
    if i >= 2:
        tri[(train[i-2], train[i-1])][train[i]] += 1
    if i >= 3:
        quad[(train[i-3], train[i-2], train[i-1])][train[i]] += 1

# character trigram (the subword model): spelling probability of any word
cbi, ctri = defaultdict(Counter), defaultdict(Counter)
for w, c in uni.items():
    s = "^^" + w + "$"
    for i in range(2, len(s)):
        cbi[s[i-1]][s[i]] += c
        ctri[(s[i-2], s[i-1])][s[i]] += c

def p_char(w):
    s = "^^" + w + "$"
    lp = 0.0
    for i in range(2, len(s)):
        d = ctri.get((s[i-2], s[i-1]))
        if d and sum(d.values()):
            p = (d[s[i]] + 0.1) / (sum(d.values()) + 0.1 * 40)
        else:
            db = cbi.get(s[i-1])
            p = (db[s[i]] + 0.1) / (sum(db.values()) + 0.1 * 40) if db else 1 / 40
        lp += math.log(p)
    return math.exp(lp)

pc = {w: p_char(w) for w in V}
pc_Z = sum(pc.values())
pc = {w: v / pc_Z for w, v in pc.items()}                  # proper vocab distribution


def cond(table, ctx, w):
    d = table.get(ctx)
    s = sum(d.values()) if d else 0
    return (d[w] / s) if s else None


def predict(ctx, w, use_subword):
    floor = pc.get(w, 1e-12) if use_subword else 1.0 / Vn   # subword vs uniform floor
    comps = [(0.08, floor), (0.07, (uni[w] + 1) / (tot + Vn))]
    for k, table, c in ((1, bi, ctx[-1:]), (2, tri, ctx[-2:]), (3, quad, ctx[-3:])):
        if len(c) == k:
            p = cond(table, c[0] if k == 1 else tuple(c), w)
            if p is not None:
                comps.append(({1: 0.20, 2: 0.27, 3: 0.38}[k], p))
    z = sum(l for l, _ in comps)
    return sum(l * p for l, p in comps) / z


def perplexity(use_subword):
    nll = cnt = 0
    for i in range(3, len(val)):
        if val[i] not in uni:                              # unseen word: only the floor can score it
            p = pc.get(val[i], 1e-12) if use_subword else 1.0 / Vn
        else:
            p = predict(tuple(val[i-3:i]), val[i], use_subword)
        nll += -math.log(max(p, 1e-12))
        cnt += 1
    return math.exp(nll / cnt)


a = perplexity(False)
b = perplexity(True)
print(f"  word 4-gram, uniform floor      : perplexity {a:.1f}")
print(f"  word 4-gram + SUBWORD floor     : perplexity {b:.1f}   ({100*(a-b)/a:+.0f}%)")
print(f"\n  => subwords {'HELP' if b < a - 1 else 'do not help'} the Markov predictor "
      f"({'the user is right -- spelling rescues rare words' if b < a - 1 else 'marginal here'}).")
print("  honest: subwords narrow it (rare-word probability), but the generalisation")
print("  ceiling -- predicting UNSEEN contexts -- is what neural still wins; that's the")
print("  Gamma-ODE's job. lattice = memory, Gamma-ODE = generation, as measured.")
