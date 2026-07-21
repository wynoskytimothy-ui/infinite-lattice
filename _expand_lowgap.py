#!/usr/bin/env python3
"""DOES LLM-EXPANSION GAIN TRACK THE VOCABULARY GAP? — testing the law where it should FAIL.

Timothy said early today: "if we literally look up this 3-way intersection on the internet, the doc's main words will
be what's missing." I measured that idea's ceiling at 8.1% and called it not-the-main-lever. I measured it ON FIQA,
which has a **7.3%** vocabulary gap -- the corpus with almost the least room for it. That was a flattening.

Per-corpus VOCAB_GAP (share of gold sharing ZERO query terms, `_profile_*.json`):
  nfcorpus 61.2% · arguana 53.8% · scidocs 18.6% · fiqa 7.3% · scifact 2.2%
LLM expansion is exactly the "look it up outside" step, and on nfcorpus (61.2%) it gained **+0.1868 nDCG**.

THE LAW UNDER TEST: expansion gain should TRACK the vocabulary gap. So it must be SMALL on fiqa (7.3%) and NEAR ZERO
or NEGATIVE on scifact (2.2%) -- where the gold already restates the claim's vocabulary and added terms only dilute.
A law that only predicts wins is not a law, so this run is aimed at the corpora where it should underperform.

PROVENANCE: expansions written by Claude from QUERY TEXT ONLY -- no gold doc, no qrels, no corpus document consulted.
Frozen in-file. Same protocol and same scorer as `_llm_expand.py` so the numbers are directly comparable.
usage: python _expand_lowgap.py
"""
import os, sys, json, math, collections, csv, itertools, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beir_data_root import resolve_beir_root

FIQA = {
 "5125": "takeover acquisition bid shareholder approval regulatory network broadcaster media australia foreign investment board control stake",
 "42": "business expense deduction home office equipment depreciation section 179 capital asset write off schedule self employed irs",
 "504": "cash flow credit score bad credit lender underwriting collateral secured loan factoring line of credit revenue based",
 "7377": "vanguard fund return quote total nav expense ratio benchmark index annualized trailing distribution yield",
 "10039": "google finance stock quote delayed real time data feed exchange retail investor screener api yahoo",
 "6612": "debt payoff mortgage interest rate rent versus buy housing market equity down payment refinance leverage amortization",
 "687": "budgeting app bank aggregation categorization mint plaid transaction sync spending category personal finance software",
 "8121": "williams percent r oscillator technical indicator overbought oversold high low close period momentum formula",
 "9598": "index fund tracking replication basket shares creation redemption expense ratio passive weighted market cap",
 "10994": "mutual fund net loss capital gain distribution shareholder pass through carryforward taxable account nav",
 "1198": "irs reclassification independent contractor employee worker status payroll tax penalty withholding backup form ss-8",
 "4942": "fund holdings composition portfolio disclosure prospectus etf constituent weight sec filing n-q holdings list",
 "4946": "mitsubishi financial statement accounting discrepancy consolidated subsidiary reporting currency yen restatement",
 "7702": "bond etf individual bond maturity duration interest rate risk yield ladder liquidity nav premium discount",
 "1676": "w2 employee 1099 contractor payroll withholding self employment tax benefits classification irs status",
 "3612": "day trading pattern rule settlement t+2 free ride margin account buying power same day sell",
 "8537": "options account approval level brokerage margin covered call spread permission suitability derivative trading",
 "9548": "mutual fund selection roth ira expense ratio morningstar rating allocation target date diversification screening",
 "6479": "brokerage exchange direct access order routing commission market maker clearing custody execution",
 "2903": "contractor tax filing schedule c self employment estimated quarterly deduction 1099 net income",
 "1994": "commute expense deduction irs mileage transportation work travel business commuting personal nondeductible",
 "10975": "roth ira income limit phase out backdoor conversion 401k contribution magi traditional nondeductible",
 "5080": "utma custodial account age majority transfer minor gift tax basis termination beneficiary control",
 "2076": "veterinary expense deduction canada tax medical pet service animal cra eligible claim",
 "4678": "car finance lease purchase cash residual depreciation interest total cost ownership monthly payment buyout",
 "7345": "futures contract quote tick size notional expiration month settlement margin open interest volume symbol",
}
SCIFACT = {
 "501": "headache migraine cognitive impairment memory dementia association cohort neuropsychological testing",
 "237": "clpc sporulation bacillus subtilis mutant defect protease chaperone spore formation efficiency",
 "13": "perinatal mortality low birth weight neonatal death infant proportion attributable risk",
 "146": "mesenchymal stem cell autologous transplantation rejection interleukin-2 receptor antibody induction kidney graft",
 "982": "growth cone protein synthesis ubiquitination local translation axon degradation proteasome neuron",
 "513": "cardiopulmonary fitness exercise capacity mortality cohort vo2 max cardiorespiratory risk",
 "1332": "tumor necrosis factor interleukin-1 proinflammatory cytokine il-6 il-10 induction regulation",
 "1290": "statin hip fracture bone mineral density osteoporosis risk cohort inverse association",
 "248": "chenodeoxycholic acid bile energy expenditure brown adipose thermogenesis metabolic rate",
 "421": "molecular flexibility rigidity steric hindrance tumor microenvironment penetration nanoparticle diffusion",
 "507": "helminth coinfection macrophage il-4 alternative activation mycobacterium tuberculosis replication immunity",
 "700": "pin1 auxin arabidopsis embryo vps9a localization trafficking polarity",
 "821": "n-terminal cleavage transcription start site identification proteomics peptide mapping accuracy",
 "1363": "venule arteriole smooth muscle layer vessel wall microcirculation structure",
 "1163": "ddrb deinococcus radiodurans single stranded dna binding protein ssb radiation repair",
 "728": "ly6c monocyte subset inflammatory capacity cytokine differentiation macrophage",
 "715": "mir-7a microrna target repression ovary germline expression function",
 "57": "apoe4 ipsc neuron amyloid beta tau phosphorylation gaba interneuron degeneration alzheimer",
 "852": "noninvasive ventilation respiratory failure conventional treatment response intubation outcome",
 "1024": "ctcf anchor site recurrent mutation oncogene chromatin loop topological domain",
 "660": "ivermectin onchocerciasis river blindness microfilariae treatment mass administration",
 "384": "noncommunicable disease burden low income country epidemiology prevalence transition mortality",
 "674": "ldl cholesterol cardiovascular disease atherosclerosis risk causal lipid lowering",
 "587": "sox2 promoter green fluorescent protein transgenic mice reporter cell percentage expression",
 "533": "fibrinogen hyperfibrinogenemia femoropopliteal bypass graft thrombosis patency",
 "216": "cx3cr1 th2 cell survival apoptosis chemokine receptor fractalkine",
}
JOBS = [("fiqa", FIQA, 7.3), ("scifact", SCIFACT, 2.2)]

ROOT = resolve_beir_root(); HERE = os.path.dirname(os.path.abspath(__file__))
POOL = 800
STOP = set("a an the of and or in on for to with is are was were be been by as at from that this it we our you your "
           "what how why when where which who i me my do does did can could should would will if there their they "
           "them then than so such about into over under between after before have has had not no yes any all".split())
def stem(w, k=6): return w[:k]
def tok(s):
    return [stem(w) for w in ''.join(c.lower() if c.isalnum() else ' ' for c in s).split()
            if len(w) > 2 and w not in STOP]
def nd10(rk, rel):
    dcg = sum(rel.get(x, 0) / math.log2(i + 2) for i, x in enumerate(rk[:10]))
    idc = sum(v / math.log2(i + 2) for i, v in enumerate(sorted(rel.values(), reverse=True)[:10]))
    return dcg / idc if idc else 0.0
def rec(rk, rel, k=100):
    g = {x for x, v in rel.items() if v > 0}
    return len(set(rk[:k]) & g) / len(g) if g else 0.0

print("=" * 100); print("LLM EXPANSION ON LOW-VOCAB-GAP CORPORA — testing the law where it should FAIL")
print("=" * 100, flush=True)
print(f"  reference: nfcorpus VOCAB_GAP 61.2% -> LLM expansion gained +0.1868 nDCG")
print(f"  PREDICTION: fiqa (7.3%) small gain; scifact (2.2%) near zero or negative\n", flush=True)
out = {"nfcorpus": {"vocab_gap": 61.2, "gain": 0.1868}}
print(f"  {'corpus':10s} {'VGAP':>6} {'n':>4} {'orig nDCG':>10} {'+LLM nDCG':>10} {'gain':>9} {'origR100':>9} {'+LLM R100':>10}")
for name, EXPD, vgap in JOBS:
    d = os.path.join(ROOT, name); corpus = {}; queries = {}
    for line in open(os.path.join(d, "corpus.jsonl"), encoding="utf-8"):
        o = json.loads(line); corpus[o["_id"]] = (o.get("title", "") + " " + o.get("text", "")).strip()
    for line in open(os.path.join(d, "queries.jsonl"), encoding="utf-8"):
        o = json.loads(line); queries[o["_id"]] = o["text"]
    qr = collections.defaultdict(dict)
    r = csv.reader(open(os.path.join(d, "qrels", "test.tsv"), encoding="utf-8"), delimiter="\t"); next(r, None)
    for row in r:
        if len(row) >= 3 and int(row[2]) > 0: qr[row[0]][row[1]] = int(row[2])
    doc_ids = list(corpus); N = len(doc_ids)
    toks = {x: set(tok(t)) for x, t in corpus.items()}
    post = collections.defaultdict(set)
    d2i = {x: i for i, x in enumerate(doc_ids)}
    for x, ts in toks.items():
        for t in ts: post[t].add(d2i[x])
    idf = {t: math.log(N / len(p)) for t, p in post.items()}
    tfd = {x: collections.Counter(tok(corpus[x])) for x in doc_ids}
    dl = {x: sum(tfd[x].values()) for x in doc_ids}; avgdl = float(np.mean(list(dl.values())))
    ids = [q for q in EXPD if q in queries and q in qr and any(qr[q].get(x, 0) > 0 for x in qr[q])]

    def run(use_llm):
        nd = []; rc = []; nt = []
        for q in ids:
            base = [t for t in dict.fromkeys(tok(queries[q])) if t in post]
            qt = list(dict.fromkeys(base + [t for t in tok(EXPD[q]) if t in post])) if use_llm else base
            qt = sorted(qt, key=lambda t: -idf[t])[:20]
            if not qt: continue
            nt.append(len(qt))
            sc = collections.defaultdict(float)
            for t in qt:
                w = idf[t]
                for di in post[t]:
                    x = doc_ids[di]; f = tfd[x].get(t, 0)
                    sc[di] += w * (f * 1.9) / (f + 0.9 * (0.25 + 0.75 * dl[x] / avgdl))
            if not sc: continue
            order = [di for di, _ in sorted(sc.items(), key=lambda x: -x[1])][:100]
            rk = [doc_ids[i] for i in order]
            nd.append(nd10(rk, qr[q])); rc.append(rec(rk, qr[q]))
        return float(np.mean(nd)), float(np.mean(rc)), float(np.mean(nt))
    o_nd, o_rc, o_nt = run(False)
    l_nd, l_rc, l_nt = run(True)
    print(f"  {name:10s} {vgap:>5.1f}% {len(ids):>4} {o_nd:>10.4f} {l_nd:>10.4f} {l_nd-o_nd:>+9.4f} "
          f"{o_rc:>9.4f} {l_rc:>10.4f}", flush=True)
    out[name] = {"vocab_gap": vgap, "n": len(ids), "orig_nDCG": round(o_nd, 4), "llm_nDCG": round(l_nd, 4),
                 "gain": round(l_nd - o_nd, 4), "orig_R100": round(o_rc, 4), "llm_R100": round(l_rc, 4),
                 "terms_orig": round(o_nt, 2), "terms_llm": round(l_nt, 2)}

g = [(v["vocab_gap"], v["gain"]) for v in out.values()]
g.sort()
print(f"\n  VOCAB_GAP vs LLM-expansion gain:")
for vg, gn in g: print(f"     {vg:>5.1f}%  ->  {gn:+.4f}")
mono = all(g[i][1] <= g[i + 1][1] for i in range(len(g) - 1))
print(f"\n  monotone in vocab gap? {'YES -- the law holds across 3 corpora' if mono else 'NO -- the law does NOT hold'}")
json.dump(out, open(os.path.join(HERE, "_expand_lowgap.json"), "w"), indent=2)
print("\nLOWGAP_JSON written")
