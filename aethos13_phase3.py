#!/usr/bin/env python3
"""AETHOS-13 PHASE 3 -- Semantic Imputation: a starved triple point triggers the LLM, absorbs bridge words, and
becomes a permanent Bridge Node. Built on the validated Phase 1/2 substrate (aethos13_lattice.py).

THE MECHANISM (master 3.9a, and the reach result measured this session):
  1  Each corpus TERM is a value. A document OCCUPIES the triple-point address of every 3-term subset it contains
     (Phase 1: state([tA,tB], tC) == state([tA,tC], tB) == state([tB,tC], tA) -- the meet, order-blind, no k-way fn).
  2  A query's rare 3-way term intersection is an ADDRESS. Documents at that address = the meet's posting list.
  3  STARVED = the address is empty/near-empty: no document ties those three terms together. The corpus has a void
     exactly where the query needs a bridge.
  4  IMPUTATION: Qwen writes synthetic text exploring the three terms. It naturally emits the BRIDGE words -- the
     technical twins (caffeine, cardiovascular, vasodilatory) that co-occur with the lay query terms.
  5  INGEST: tokenise the synthetic doc, add it to the corpus. The void's address now has an occupant, AND new
     correlate edges are born (coffee<->caffeine, heart<->cardiovascular) that did not exist before.
  6  TRANSFORM: the void is now a permanent Bridge Node. The measurable payoff: a REAL gold document that uses the
     bridge vocabulary (caffeine/cardiovascular) instead of the query's words -- previously unreachable -- is now
     reached through the new edges. This is the targeted, permanent version of the +0.0519 reach measured earlier.

Honest test: the synthetic doc is NOT the gold doc. We measure whether the bridge words it introduces connect the
query to the REAL gold, and whether the starved address genuinely fills. Aggregated over many queries, with a
CONTROL (ingest a random synthetic doc of the same length) so "adding any text" cannot masquerade as "bridging".
usage: python aethos13_phase3.py [n_queries]
"""
import os, sys, json, math, collections, csv, itertools, random, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from beir_data_root import resolve_beir_root
from aethos13_lattice import state, triple_point            # the validated substrate

NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 40
HERE = os.path.dirname(os.path.abspath(__file__)); CORPUS = "nfcorpus"; STARVED_MAX = 1
STOP = set("a an the of and or in on for to with is are was were be been by as at from that this it we our you your "
           "what how why when where which who i me my do does did can could should would will if there their they "
           "them then than so such about into over under between after before have has had not no yes any all".split())
def raw(s): return ''.join(c.lower() if c.isalnum() else ' ' for c in s).split()
def tok(s): return [w for w in raw(s) if w not in STOP and len(w) > 2]

d = os.path.join(resolve_beir_root(), CORPUS); corpus = {}; queries = {}
for line in open(os.path.join(d, "corpus.jsonl"), encoding="utf-8"):
    o = json.loads(line); corpus[o["_id"]] = (o.get("title", "") + " " + o.get("text", "")).strip()
for line in open(os.path.join(d, "queries.jsonl"), encoding="utf-8"):
    o = json.loads(line); queries[o["_id"]] = o["text"]
qr = collections.defaultdict(dict)
r = csv.reader(open(os.path.join(d, "qrels", "test.tsv"), encoding="utf-8"), delimiter="\t"); next(r, None)
for row in r:
    if len(row) >= 3 and int(row[2]) > 0: qr[row[0]][row[1]] = int(row[2])
docs = list(corpus); N0 = len(docs)
SET = {x: set(tok(corpus[x])) for x in docs}
post = collections.defaultdict(set)
for x in docs:
    for w in SET[x]: post[w].add(x)
df = collections.Counter({w: len(p) for w, p in post.items()})
IDF = {w: math.log(1 + (N0 - df[w] + 0.5) / (df[w] + 0.5)) for w in post}
# term -> integer value for the lattice (rank by df so the substrate sees stable integers)
VALUE = {w: i + 1 for i, w in enumerate(sorted(post, key=lambda w: (-df[w], w)))}

print("=" * 100); print(f"AETHOS-13 PHASE 3 -- SEMANTIC IMPUTATION | {CORPUS} | {N0:,} docs"); print("=" * 100, flush=True)

# ---- load Qwen ----
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
DEV = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()
QTOK = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
QM = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()
print(f"  Qwen2.5-1.5B loaded in {time.time()-t0:.0f}s on {DEV}", flush=True)
def generate(terms):
    msg = [{"role": "user", "content":
            f"Write three sentences from a medical/nutrition research abstract that connect these topics: "
            f"{', '.join(terms)}. Use precise technical and physiological terminology."}]
    p = QTOK.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    i = QTOK(p, return_tensors="pt").to(DEV)
    with torch.no_grad():
        o = QM.generate(**i, max_new_tokens=110, do_sample=True, temperature=0.7, top_p=0.9,
                        pad_token_id=QTOK.eos_token_id)
    return QTOK.decode(o[0][i.input_ids.shape[1]:], skip_special_tokens=True)

def meet_docs(terms, POST):
    """documents occupying the triple point of these terms = docs containing all of them (the meet's posting list)."""
    if len(terms) < 2: return set()
    s = set(POST.get(terms[0], set()))
    for t in terms[1:]: s &= POST.get(t, set())
    return s

# ---- pick queries whose rare 3-way meet is STARVED and whose gold is not directly reachable ----
ids = [q for q in qr if q in queries and any(qr[q].get(x, 0) > 0 for x in qr[q])]
random.Random(0).shuffle(ids)
def bm25_rank(qt, POST, TF, DLm, avg, Nn, IDFm, gold):
    sc = collections.defaultdict(float)
    for t in qt:
        w = IDFm.get(t, 0.0)
        for x in POST.get(t, ()):
            f = TF[x].get(t, 0); sc[x] += w * (f * 1.9) / (f + 0.9 * (0.6 + 0.4 * DLm[x] / avg))
    order = [x for x, _ in sorted(sc.items(), key=lambda z: -z[1])]
    return min([order.index(g) + 1 for g in gold if g in order], default=10**9)

TF = {x: collections.Counter(tok(corpus[x])) for x in docs}
DLm = {x: sum(TF[x].values()) for x in docs}; AVG = float(np.mean(list(DLm.values())))
def bm25_score(qt):
    sc = collections.defaultdict(float)
    for t in qt:
        w = IDF.get(t, 0.0)
        for x in post.get(t, ()):
            f = TF[x].get(t, 0); sc[x] += w * (f * 1.9) / (f + 0.9 * (0.6 + 0.4 * DLm[x] / AVG))
    return sc
print("\n  scanning for STARVED triple points with unreachable gold ...", flush=True)
cases = []
for q in ids:
    gold = {x for x in qr[q] if qr[q].get(x, 0) > 0 and x in corpus}
    qt = [t for t in dict.fromkeys(tok(queries[q])) if t in post]
    if len(qt) < 3 or not gold: continue
    tri = sorted(qt, key=lambda t: -IDF[t])[:3]                 # the rarest 3 -- the most specific meet
    starved = len(meet_docs(tri, post)) <= STARVED_MAX
    # FLAW-1 FIX: require gold that is genuinely UNREACHABLE (rank>100), else there is no reach headroom to recover.
    unreachable = [g for g in gold if g not in set(x for x, _ in sorted(
        bm25_score(qt).items(), key=lambda z: -z[1])[:100])]
    if starved and unreachable:
        cases.append((q, qt, tri, set(unreachable)))
    if len(cases) >= NQ: break
print(f"  found {len(cases)} STARVED cases with UNREACHABLE gold (the regime where a bridge can help)", flush=True)

# ---- ONE case, fully inspectable ----
def inspect(case):
    q, qt, tri, gold = case
    print("\n" + "-" * 100)
    print(f"  QUERY: {queries[q]}")
    print(f"  rarest 3-way triple point: {tri}  (values {[VALUE[t] for t in tri]})")
    addr = state([VALUE[tri[0]], VALUE[tri[1]]], VALUE[tri[2]])
    print(f"  lattice address (triple point): state = {addr.as_tuple()}  "
          f"[{'CONVERGES' if triple_point(VALUE[tri[0]], VALUE[tri[1]], VALUE[tri[2]]) else 'ERR'}]")
    print(f"  meet occupancy BEFORE: {len(meet_docs(tri, post))} documents  <- STARVED (a geometric void)")
    gt = tok(' '.join(corpus[g] for g in gold))
    gold_bridge = [t for t in set(gt) if t not in qt and t in post and df[t] >= 5]
    print(f"  gold uses these non-query terms (the bridge target): {sorted(gold_bridge, key=lambda t:-IDF[t])[:8]}")
    syn = generate(tri)
    print(f"\n  Qwen SYNTHETIC bridge text:\n    {syn[:260].strip()}")
    syn_terms = set(tok(syn))
    absorbed = [t for t in syn_terms if t in post and t not in qt and df[t] >= 3]
    hit_gold = [t for t in syn_terms if t in gold_bridge]
    print(f"\n  bridge words ABSORBED (in-corpus, not in query): {sorted(absorbed, key=lambda t:-IDF.get(t,0))[:10]}")
    print(f"  of those, words that ALSO appear in the real gold: {hit_gold}  <- the bridge that was missing")
    return syn

# ---- aggregate: ingest each synthetic doc, measure void-fill + gold reach, vs random-text control ----
def ingest_and_measure(cases, mode):
    POST = {w: set(p) for w, p in post.items()}
    TFn = dict(TF); DLn = dict(DLm); SETn = dict(SET)
    filled = 0; reach_before = 0; reach_after = 0; bridge_absorbed = 0; n = 0
    rng = random.Random(7); vocab = list(post)
    for (q, qt, tri, gold) in cases:
        n += 1
        before = bm25_rank(qt, POST, TFn, DLn, AVG, len(SETn), IDF, gold)
        if mode == "impute":
            syn = generate(tri)
        else:                                                   # CONTROL: random in-vocab text, same length
            syn = ' '.join(rng.sample(vocab, 60))
        sid = f"__syn_{n}"
        st = tok(syn); SETn[sid] = set(st); TFn[sid] = collections.Counter(st); DLn[sid] = len(st)
        for w in SETn[sid]: POST.setdefault(w, set()).add(sid)
        after_meet = len(meet_docs(tri, POST))
        if after_meet > STARVED_MAX: filled += 1
        # FLAW-2 FIX: do NOT dump all synthetic terms into the query (that is blind expansion, known to dilute).
        # The synthetic doc created BRIDGE EDGES: query term -> synthetic doc -> its OTHER terms. Follow those
        # edges ONE hop and keep only bridge terms that are RARE (discriminating twins, not generic filler),
        # each added at a small weight. This is "reach through the bridge", not "expand with everything".
        bridge_terms = collections.Counter()
        for t in qt:
            for sd in POST.get(t, ()):
                if sd.startswith("__syn"):
                    for b in SETn[sd]:
                        if b not in qt and b in post and df[b] >= 3 and IDF[b] > IDF.get(t, 0) * 0.7:
                            bridge_terms[b] += 1
        bridge_kept = [b for b, _ in bridge_terms.most_common(4)]
        bridge_absorbed += len(bridge_kept)
        sc = bm25_score(qt)                                     # base query score on ORIGINAL corpus
        for b in bridge_kept:                                   # add the discriminating bridge terms, low weight
            w = 0.5 * IDF[b]
            for x in post.get(b, ()):
                f = TF[x].get(b, 0); sc[x] += w * (f * 1.9) / (f + 0.9 * (0.6 + 0.4 * DLm[x] / AVG))
        order = [x for x, _ in sorted(sc.items(), key=lambda z: -z[1])]
        after = min([order.index(g) + 1 for g in gold if g in order], default=10**9)
        reach_before += (before <= 100); reach_after += (after <= 100)
    return dict(n=n, filled=filled, reach_before=reach_before, reach_after=reach_after,
                bridge=bridge_absorbed / max(1, n))

if cases:
    print("\n" + "=" * 100); print("  ONE CASE, FULLY INSPECTABLE"); print("=" * 100)
    inspect(cases[0])
    print("\n" + "=" * 100); print(f"  AGGREGATE over {len(cases)} starved cases (impute vs random-text control)")
    print("=" * 100, flush=True)
    imp = ingest_and_measure(cases, "impute")
    ctl = ingest_and_measure(cases, "control")
    print(f"\n  {'metric':40s} {'IMPUTE':>10} {'CONTROL':>10}")
    print(f"  {'starved voids FILLED (meet now populated)':40s} {imp['filled']:>7}/{imp['n']} {ctl['filled']:>7}/{ctl['n']}")
    print(f"  {'bridge words absorbed per doc':40s} {imp['bridge']:>10.1f} {ctl['bridge']:>10.1f}")
    print(f"  {'gold reachable BEFORE (top-100)':40s} {imp['reach_before']:>7}/{imp['n']} {'--':>10}")
    print(f"  {'gold reachable AFTER  (top-100)':40s} {imp['reach_after']:>7}/{imp['n']} {ctl['reach_after']:>7}/{ctl['n']}")
    print(f"\n  IMPUTE recovered {imp['reach_after']-imp['reach_before']} previously-unreachable gold docs; "
          f"control recovered {ctl['reach_after']-ctl['reach_before']}.")
    print(f"  VERDICT: {'the void became a BRIDGE NODE that reaches real gold -- above the random-text control' if imp['reach_after']-imp['reach_before'] > ctl['reach_after']-ctl['reach_before'] else 'imputation did not beat adding random text'}")
    json.dump({"corpus": CORPUS, "cases": len(cases), "impute": imp, "control": ctl},
              open(os.path.join(HERE, "_phase3.json"), "w"), indent=2)
    print("\nPHASE3_JSON written")
else:
    print("  no starved cases found at this threshold")
