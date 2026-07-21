#!/usr/bin/env python3
"""LLM QUERY EXPANSION vs PMI EXPANSION vs RANDOM — does OUTSIDE knowledge behave like NATURAL query length?

_length_causal.py established:
  - length is CAUSAL within-corpus (arguana truncation: margin +0.0689 @3 terms -> +0.2132 @14 terms, monotone)
  - PMI expansion RAISES the convergence margin 15x (nfcorpus +0.0041 -> +0.0638) and beats a random-term control
    by +0.0408, so the correlates add INFORMATION not just tokens
  - BUT it does not convert: nDCG 0.2856 -> 0.2639 (precision lost) while R@100 0.2573 -> 0.2853 (recall gained)

Diagnosis: PMI correlates are DERIVED FROM THE QUERY, so they cannot contribute evidence the query did not already
have. Natural length (arguana) lifted nDCG 4.6x because those terms carry INDEPENDENT information.
An LLM is the one expansion source that adds OUTSIDE knowledge. If the diagnosis is right, LLM expansion should
behave more like natural length than PMI expansion does.

PROVENANCE OF THE EXPANSIONS BELOW: written by Claude (the assistant) from THE QUERY TEXT ONLY, plus general
domain knowledge of nutrition/medical abstract vocabulary. NO gold document, no qrels, and no corpus document was
consulted for any expansion. They are frozen in this file so the run is reproducible and auditable.
usage: python _llm_expand.py
"""
import os, sys, json, math, collections, csv, itertools, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beir_data_root import resolve_beir_root

# query_id -> hypothetical-answer vocabulary an actual relevant abstract would plausibly use
EXP = {
 "PLAIN-1568": "maple syrup sap sucrose polyphenol antioxidant glycemic index mineral manganese zinc quebec grade sweetener refined sugar substitute inflammation",
 "PLAIN-1299": "growth promoter antibiotic livestock ractopamine hormone residue poultry swine feed efficiency antimicrobial resistance withdrawal meat consumer exposure",
 "PLAIN-3442": "vegetable antioxidant capacity orac cruciferous kale spinach nutrient density phytochemical carotenoid vitamin cancer cell proliferation inhibition assay",
 "PLAIN-2520": "caloric restriction plant based diet longevity igf-1 methionine protein restriction lifespan aging biomarker vegan mortality metabolic",
 "PLAIN-1679": "myelopathy spinal cord vitamin b12 deficiency subacute combined degeneration demyelination nitrous oxide neurologic cobalamin methylmalonic",
 "PLAIN-2281": "turnip brassica glucosinolate cruciferous root vegetable greens nitrate sulforaphane isothiocyanate cancer chemoprevention",
 "PLAIN-1611": "mesquite pod flour glycemic legume prosopis fiber protein arid desert traditional food galactomannan blood glucose",
 "PLAIN-3261": "herbalife supplement hepatotoxicity liver injury weight loss product case report jaundice hepatitis contamination multilevel marketing",
 "PLAIN-2430": "brain atrophy homocysteine b vitamin folate cognitive decline mild impairment mri gray matter supplementation trial elderly dementia",
 "PLAIN-1320": "physicians health study randomized trial multivitamin cancer incidence cardiovascular male cohort follow up placebo harvard supplementation",
 "PLAIN-3074": "abdominal aortic aneurysm rupture screening ultrasound smoking atherosclerosis elastin collagen degradation risk factor diet fruit vegetable",
 "PLAIN-1590": "medical ethics conflict of interest disclosure industry funding informed consent research integrity publication bias pharmaceutical",
 "PLAIN-913": "cinnamon cassia coumarin blood glucose insulin sensitivity type 2 diabetes hba1c hepatotoxicity ceylon polyphenol postprandial",
 "PLAIN-1506": "leucine branched chain amino acid mtor muscle protein synthesis sarcopenia supplementation insulin resistance signaling elderly",
 "PLAIN-1109": "endocrine disruptor bisphenol phthalate xenoestrogen hormone receptor exposure urinary metabolite reproductive developmental toxicity",
 "PLAIN-2197": "sweetener aspartame sucralose saccharin stevia artificial noncaloric glucose tolerance microbiome appetite weight gain",
 "PLAIN-2650": "turmeric curcumin osteoarthritis knee pain womac inflammation nsaid randomized bioavailability piperine cartilage joint",
 "PLAIN-1172": "fenugreek trigonella seed galactomannan blood glucose lipid testosterone lactation saponin diosgenin supplementation",
 "PLAIN-2910": "phytosterol plant stanol ester ldl cholesterol lowering dose response margarine absorption sitosterol meta analysis",
 "PLAIN-892": "chickpea legume pulse fiber satiety glycemic hummus protein cooking phytate mineral bioavailability cardiovascular",
 "PLAIN-2177": "sulfur amino acid methionine cysteine hydrogen sulfide sulfate dietary intake protein restriction colonic bacteria",
 "PLAIN-2009": "rhabdomyolysis creatine kinase myoglobinuria renal failure statin exercise muscle breakdown electrolyte",
 "PLAIN-3085": "vitamin d recommendation serum 25-hydroxyvitamin deficiency supplementation dose iu bone mineral density sunlight rda intake",
 "PLAIN-291": "stool bulk fecal weight breast cancer estrogen enterohepatic circulation fiber intake bowel transit excretion",
 "PLAIN-1579": "mastitis mammary gland inflammation lactation somatic cell count bovine milk antibiotic infection breastfeeding",
 "PLAIN-1897": "polypropylene plastic container leaching migration food contact microplastic additive heat exposure polymer",
 "PLAIN-2540": "ldl particle size small dense cholesterol subfraction cardiovascular risk apolipoprotein pattern lipoprotein atherogenic",
 "PLAIN-1741": "nut consumption almond walnut cardiovascular mortality cohort lipid ldl weight satiety unsaturated fatty acid",
 "PLAIN-1275": "goji berry lycium barbarum wolfberry antioxidant zeaxanthin polysaccharide macular supplementation carotenoid",
 "PLAIN-2840": "fenugreek seed supplementation glucose lipid galactomannan trigonelline randomized diabetic rodent lactation",
 "PLAIN-956": "cooking method boiling steaming frying acrylamide heterocyclic amine nutrient retention antioxidant loss temperature",
 "PLAIN-2890": "snacking meal frequency eating episodes weight energy intake satiety glycemic grazing obesity",
 "PLAIN-383": "paleolithic diet jenkins ancestral hunter gatherer fiber intake legume grain cholesterol dietary portfolio",
 "PLAIN-143": "dental radiograph x-ray ionizing radiation dose thyroid meningioma cancer risk imaging exposure",
 "PLAIN-1225": "fructose high fructose corn syrup hepatic lipogenesis uric acid insulin resistance triglyceride sugar sweetened beverage",
 "PLAIN-1249": "genetic modification transgenic crop gmo glyphosate herbicide tolerant safety assessment labeling",
 "PLAIN-660": "bean legume pulse fiber resistant starch glycemic satiety cardiovascular cholesterol lowering phytate",
 "PLAIN-1940": "prolactin pituitary hormone secretion hyperprolactinemia lactation dopamine breast tissue serum level",
 "PLAIN-3191": "fish oil distillation pcb dioxin mercury contaminant omega-3 supplement purity oxidation rancidity",
 "PLAIN-2690": "taeniasis cysticercosis pork tapeworm neurocysticercosis headache seizure larval cyst brain imaging",
 "PLAIN-711": "blood clot thrombosis coagulation fibrinogen platelet aggregation deep vein embolism anticoagulant diet",
 "PLAIN-1419": "insect entomophagy edible protein cricket mealworm sustainability feed conversion allergen chitin",
 "PLAIN-603": "arkansas poultry industry farming rural water contamination litter environmental",
 "PLAIN-133": "angiogenesis tumor blood supply vegf inhibitor antiangiogenic phytochemical starvation cancer growth endothelial",
 "PLAIN-280": "mercury methylmercury fish consumption pregnancy hair level fetal neurodevelopment testing exposure advisory",
}

ROOT = resolve_beir_root(); HERE = os.path.dirname(os.path.abspath(__file__))
EXPAND = 12; POOL = 800
STOP = set("a an the of and or in on for to with is are was were be been by as at from that this it we our you your "
           "what how why when where which who i me my do does did can could should would will if there their they "
           "them then than so such about into over under between after before have has had not no yes any all".split())
def stem(w, k=6): return w[:k]
def tok(s):
    return [stem(w) for w in ''.join(c.lower() if c.isalnum() else ' ' for c in s).split()
            if len(w) > 2 and w not in STOP]

d = os.path.join(ROOT, "nfcorpus"); corpus = {}; queries = {}
for line in open(os.path.join(d, "corpus.jsonl"), encoding="utf-8"):
    o = json.loads(line); corpus[o["_id"]] = (o.get("title", "") + " " + o.get("text", "")).strip()
for line in open(os.path.join(d, "queries.jsonl"), encoding="utf-8"):
    o = json.loads(line); queries[o["_id"]] = o["text"]
qr = collections.defaultdict(dict)
r = csv.reader(open(os.path.join(d, "qrels", "test.tsv"), encoding="utf-8"), delimiter="\t"); next(r, None)
for row in r:
    if len(row) >= 3 and int(row[2]) > 0: qr[row[0]][row[1]] = int(row[2])
doc_ids = list(corpus); N = len(doc_ids); d2i = {x: i for i, x in enumerate(doc_ids)}
toks = {x: set(tok(t)) for x, t in corpus.items()}
post = collections.defaultdict(set)
for x, ts in toks.items():
    for t in ts: post[t].add(d2i[x])
idf = {t: math.log(N / len(p)) for t, p in post.items()}
tfd = {x: collections.Counter(tok(corpus[x])) for x in doc_ids}
dl = {x: sum(tfd[x].values()) for x in doc_ids}; avgdl = float(np.mean(list(dl.values())))
KEEP = {t for t, p in post.items() if 3 <= len(p) <= N * 0.10}
co = collections.defaultdict(collections.Counter)
for x in doc_ids:
    ts = [t for t in toks[x] if t in KEEP]
    if len(ts) > 50: ts = sorted(ts, key=lambda t: -idf[t])[:50]
    for a, b in itertools.combinations(ts, 2):
        co[a][b] += 1; co[b][a] += 1
CORR = {}
for a, cnt in co.items():
    da = len(post[a]); s = []
    for b, c in cnt.items():
        if c < 2: continue
        p = math.log((c * N) / (da * len(post[b])) + 1e-12)
        if p > 0: s.append((p, b))
    s.sort(reverse=True)
    if s: CORR[a] = s[:EXPAND]
ALLC = sorted({t for a in CORR for _, t in CORR[a]} | set(CORR))

ids = [x for x in EXP if x in queries and x in qr and any(qr[x].get(z, 0) > 0 for z in qr[x])]
print("=" * 104); print(f"LLM EXPANSION | nfcorpus | {len(ids)} queries with frozen LLM expansions"); print("=" * 104)
print(f"  expansions written from QUERY TEXT ONLY -- no gold doc, no qrels, no corpus doc consulted", flush=True)

def esym(v, kmax):
    e = [1.0] + [0.0] * kmax
    for x in v:
        for k in range(min(kmax, len(v)), 0, -1): e[k] += x * e[k - 1]
    return e[1:]
def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    if y.sum() == 0 or y.sum() == len(y): return float("nan")
    o = np.argsort(s); rr = np.empty(len(s), float); rr[o] = np.arange(1, len(s) + 1)
    _, inv, c = np.unique(s, return_inverse=True, return_counts=True)
    sm = np.zeros(len(c)); np.add.at(sm, inv, rr); rr = (sm / c)[inv]
    a = y.sum(); b = len(y) - a
    return float((rr[y == 1].sum() - a * (a + 1) / 2) / (a * b))

def run(fn, label):
    rng = random.Random(21); conv = []; ctrl = []; nd = []; rc = []; nt = []
    for q in ids:
        gold = {d2i[x] for x in qr[q] if qr[q].get(x, 0) > 0 and x in d2i}
        if not gold: continue
        base = [t for t in dict.fromkeys(tok(queries[q])) if t in post]
        base = sorted(base, key=lambda t: -idf[t])
        qt = fn(q, base, rng)
        if len(qt) < 2: continue
        nt.append(len(qt))
        sc = collections.defaultdict(float)
        for t in qt:
            if t not in post: continue
            w = idf[t]
            for di in post[t]:
                x = doc_ids[di]; f = tfd[x].get(t, 0)
                sc[di] += w * (f * 1.9) / (f + 0.9 * (0.25 + 0.75 * dl[x] / avgdl))
        if not sc: continue
        order = [di for di, _ in sorted(sc.items(), key=lambda x: -x[1])]
        rr = [doc_ids[i] for i in order[:100]]; rel = qr[q]
        dcg = sum(rel.get(x, 0) / math.log2(i + 2) for i, x in enumerate(rr[:10]))
        idc = sum(v / math.log2(i + 2) for i, v in enumerate(sorted(rel.values(), reverse=True)[:10]))
        nd.append(dcg / idc if idc else 0.0)
        gs = {x for x, v in rel.items() if v > 0}
        rc.append(len(set(rr[:100]) & gs) / len(gs) if gs else 0.0)
        groups = [[t] + [b for _, b in CORR.get(t, [])] for t in qt[:10]]
        rg = [[t] + rng.sample(ALLC, min(EXPAND, len(ALLC))) for t in qt[:10]]
        pool = list(dict.fromkeys(order[:POOL] + [g for g in gold if g in sc]))
        if len(pool) < 20 or not (set(pool) & gold): continue
        y = [1 if di in gold else 0 for di in pool]
        ev = []; ec = []
        for di in pool:
            dt = toks[doc_ids[di]]
            ev.append(esym([sum(1 for w in g if w in dt) for g in groups], 3)[2])
            ec.append(esym([sum(1 for w in g if w in dt) for g in rg], 3)[2])
        a = auc(ev, y); c = auc(ec, y)
        if not math.isnan(a) and not math.isnan(c): conv.append(a); ctrl.append(c)
    m = float(np.mean(conv)) - float(np.mean(ctrl))
    print(f"   {label:32s} {np.mean(nt):>7.2f} {m:>+9.4f} {np.mean(nd):>9.4f} {np.mean(rc):>8.4f}", flush=True)
    return {"label": label, "terms": round(float(np.mean(nt)), 2), "margin": round(m, 4),
            "nDCG@10": round(float(np.mean(nd)), 4), "R@100": round(float(np.mean(rc)), 4)}

def pmi_pad(q, base, rng):
    out = list(base)
    for t in base:
        for _, b in CORR.get(t, []):
            if len(out) >= 12: break
            if b not in out: out.append(b)
        if len(out) >= 12: break
    return out
def llm(q, base, rng):
    return list(dict.fromkeys(base + [t for t in tok(EXP[q]) if t in post]))
def llm_only(q, base, rng):
    return [t for t in dict.fromkeys(tok(EXP[q])) if t in post]
def rand_pad(q, base, rng):
    out = list(base)
    while len(out) < 12:
        b = rng.choice(ALLC)
        if b not in out: out.append(b)
    return out

print(f"\n   {'arm':32s} {'terms':>7} {'margin':>9} {'nDCG@10':>9} {'R@100':>8}")
res = [run(lambda q, b, r: b, "ORIGINAL query"),
       run(pmi_pad, "+ PMI correlates"),
       run(rand_pad, "+ RANDOM control"),
       run(llm, "+ LLM expansion"),
       run(llm_only, "LLM expansion ONLY (no orig)")]
o, p_, rd, l, lo = res
print(f"\n   vs ORIGINAL   nDCG      R@100     margin")
for x in res[1:]:
    print(f"   {x['label']:28s} {x['nDCG@10']-o['nDCG@10']:>+7.4f} {x['R@100']-o['R@100']:>+9.4f} {x['margin']-o['margin']:>+9.4f}")
print(f"\n   LLM vs PMI: nDCG {l['nDCG@10']-p_['nDCG@10']:+.4f}  R@100 {l['R@100']-p_['R@100']:+.4f}")
print(f"   VERDICT: {'LLM expansion CONVERTS where PMI did not' if l['nDCG@10'] > o['nDCG@10'] else 'LLM expansion also fails to lift nDCG -- expansion is a RECALL lever, not a precision one'}")
json.dump(res, open(os.path.join(HERE, "_llm_expand.json"), "w"), indent=2)
print("\nLLM_EXPAND_JSON written")
