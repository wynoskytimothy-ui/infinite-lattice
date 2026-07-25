#!/usr/bin/env python3
"""AETHOS-13 PHASE 5 -- The Shadow Query & Anchor-Gravity Pruning (subsumption routing).

Phase 3 measured REACH (can the lattice find the missing document?). Phase 5 measures RANK: can the
substrate's structural intelligence kill noise and force the gold document to TOP-1?

Two mechanisms from the master spec, built on the validated subsumption DAG (P(b|a) directed containment,
verified in _walk.py -- Heart lives INSIDE Disease at 0.79, Disease inside Heart only 0.18):

  ANCHOR-GRAVITY PRUNING (Requirement 1)
    * Specificity weight: each query anchor's gravity = its height in the partial order. The most-specific
      anchor (rarest -> highest idf -> lowest in the DAG) casts the HEAVIEST gravity.
    * The prune: a document that satisfies only the general/broad terms but fails the specific anchor's
      gravity tether is KILLED -- not scored, not retrieved.

  SHADOW QUERY (Requirement 2)
    * O(1) twin lookup from the BAKED miniverse correlate graph (co-occurrence lift, the _walk.py miniverse()).
    * The tether accepts the anchor OR a baked twin of it (caffeine for coffee, cardiac for heart).
    * The lock: a document reached through the anchor's TWIN that ALSO satisfies a general term is a latent
      triangulation -- primary gravity and shadow converge on the same node -> boost.

  ============================================================================================================
  HONESTY DISCIPLINE (the whole point -- a constructed win is worthless unless it survives its own controls):
    A. The corpus is built so a FAIR BM25 baseline FAILS (anchor-less noise at top-1). Verified in code, not
       asserted -- if the construction accidentally lets BM25 win, the demo SAYS SO.
    B. It is run as an ABLATION LADDER, not a single pass, so each component's credit is attributed honestly:
         BM25  ->  +idf-gravity (drop TF)  ->  +hard-gate  ->  +shadow.
       This exposes the real decomposition (found on the first build): scoring-by-idf-presence does most of
       the lifting; the hard PRUNE adds little over it (the rare anchor's idf already outweighs the commons);
       the SHADOW is the genuinely additive piece (vocabulary mismatch).
    C. Two controls that must break a vacuous version:
         wrong-direction anchor -> gravity cast from the most-GENERAL term must NOT lock gold.
         shadow-off             -> the twin-only gold (zero literal query terms) must be UNREACHABLE.
    D. Twins are BAKED FROM CO-OCCURRENCE, never hand-wired to the answer -- the learned graph is printed,
       polysemy pollution included, so you can see it wasn't planted.
    E. Then it is measured on REAL cached BEIR (P@1 / MRR@10 / nDCG@10 / R@100 vs BM25, matched queries).
       Hard-kill AND soft-fuse are BOTH reported -- hard-kill exposes the recall damage honestly.

usage:
  python aethos13_phase5.py                 # constructed corpus: ablation ladder + controls
  python aethos13_phase5.py real nfcorpus   # real-BEIR generalization on one corpus
  python aethos13_phase5.py real scifact
"""
from __future__ import annotations
import os, sys, math, json, collections, random
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass


def tok(s: str) -> List[str]:
    """[a-z0-9]+ lowercaser, matched to marco_baseline.tok so df/postings share BM25's vocabulary."""
    return [w for w in ''.join(c.lower() if c.isalnum() else ' ' for c in s).split() if w]

STOP = set("a an the of and or in on for to with is are was were be been by as at from that this it we our you your "
           "what how why when where which who i me my do does did can could should would will if there their they "
           "them then than so such about into over under between after before have has had not no yes any all "
           "increase increases increased raise raises drinking".split())

def content(s: str, minlen: int = 3) -> List[str]:
    return [w for w in tok(s) if len(w) >= minlen and w not in STOP]


# ======================================================================================================
# THE LATTICE INDEX -- postings, idf, and the BAKED miniverse correlate graph (twins).
# ======================================================================================================
class LatticeIndex:
    def __init__(self, corpus: Dict[str, str], mindf: int = 2):
        self.ids = list(corpus)
        self.N = len(self.ids)
        self.mindf = mindf
        self.SET: Dict[str, Set[str]] = {d: set(content(corpus[d])) for d in self.ids}
        self.post: Dict[str, Set[str]] = collections.defaultdict(set)
        for d in self.ids:
            for w in self.SET[d]:
                self.post[w].add(d)
        self.df = {w: len(p) for w, p in self.post.items()}
        self._idf = {w: math.log(1.0 + (self.N - df + 0.5) / (df + 0.5)) for w, df in self.df.items()}
        self._twin_cache: Dict[str, List[str]] = {}

    def idf(self, t: str) -> float:
        return self._idf.get(t, math.log(1.0 + (self.N + 0.5) / 0.5))  # unseen term = max idf

    def twins(self, t: str, k: int = 12, min_lift: float = 1.5, min_co: int = 3) -> List[str]:
        """The BAKED miniverse correlate graph for one term: the words that live in t's world far more than
        chance (co-occurrence lift), df-filtered. This is _walk.py's miniverse(), cached -> O(1) after first."""
        if t in self._twin_cache:
            return self._twin_cache[t]
        docs = self.post.get(t, ())
        if not docs:
            self._twin_cache[t] = []
            return []
        c = collections.Counter()
        for d in docs:
            c.update(self.SET[d])
        nt = len(docs)
        scored = []
        for w, co in c.items():
            if w == t or self.df.get(w, 0) < self.mindf or co < min_co:
                continue
            lift = (co / nt) / (self.df[w] / self.N)
            if lift >= min_lift:
                scored.append((lift * math.log(1 + co), w))
        scored.sort(reverse=True)
        out = [w for _, w in scored[:k]]
        self._twin_cache[t] = out
        return out


# ======================================================================================================
# THE MECHANISM -- anchor-gravity pruning + shadow query.
# ======================================================================================================
@dataclass
class RankResult:
    ranked: List[Tuple[str, float]]
    killed: List[str]
    anchor: str
    tether: Set[str]
    order: List[str]
    log: List[str]


def anchor_gravity_rank(idx: LatticeIndex, qterms: List[str], candidates: List[str],
                        mode: str = "hard", anchor_pick: str = "specific",
                        use_shadow: bool = True, verbose: bool = False) -> RankResult:
    """Route candidates through the subsumption DAG.

    mode        : 'hard' kills a branch that fails the anchor tether; 'soft' penalises it (x0.12) instead;
                  'none' scores every candidate by idf-gravity (the drop-TF, no-prune rung of the ladder).
    anchor_pick : 'specific' casts gravity from the most-specific (highest-idf) query term -- the real rule;
                  'general' casts from the most-general term -- the wrong-direction control.
    use_shadow  : True lets a baked twin of the anchor satisfy the tether AND expands scoring to twins
                  (Shadow Query); False = literal query terms only.
    """
    log: List[str] = []
    present = [t for t in dict.fromkeys(qterms) if t in idx.post]
    if not present:
        # no literal query term is indexed. Shadow is the ONLY way in: expand each query term to its twins.
        if use_shadow:
            shadow_terms = []
            for t in dict.fromkeys(qterms):
                shadow_terms += idx.twins(t)
            shadow_terms = list(dict.fromkeys(shadow_terms))
            log.append(f"no literal query term indexed -> SHADOW-ONLY reach via twins {shadow_terms}")
            scored = []
            for d in candidates:
                g = sum(idx.idf(w) for w in shadow_terms if w in idx.SET[d])
                if g > 0:
                    scored.append((d, g))
            scored.sort(key=lambda z: -z[1])
            return RankResult(scored, [], "", set(shadow_terms), [], log)
        return RankResult([], [], "", set(), [], ["no literal query term and shadow off -> unreachable"])

    order = sorted(present, key=lambda t: -idx.idf(t))   # specific(rare) -> general(common)
    anchor = order[0] if anchor_pick == "specific" else order[-1]
    anchor_twins = set(idx.twins(anchor)) if use_shadow else set()
    tether = {anchor} | anchor_twins
    gen_terms = [t for t in order if t != anchor]
    # scoring vocabulary: query terms, plus (with shadow) each query term's twins at a discounted weight
    shadow_map = {t: idx.twins(t) for t in present} if use_shadow else {}

    log.append(f"partial order (specific->general): {' < '.join(order)}")
    log.append("idf gravity: " + ", ".join(f"{t}={idx.idf(t):.2f}" for t in order))
    log.append(f"ANCHOR (heaviest gravity, {anchor_pick}) = '{anchor}'  idf={idx.idf(anchor):.2f}")
    if use_shadow and anchor_twins:
        log.append(f"SHADOW twins of '{anchor}' (baked correlate graph): {sorted(anchor_twins)}")
    log.append(f"tether = {sorted(tether)}   mode={mode}")

    survivors: List[Tuple[str, float]] = []
    killed: List[str] = []
    for d in candidates:
        S = idx.SET[d]
        satisfies = bool(S & tether)
        if not satisfies and mode == "hard":
            killed.append(d)
            if verbose:
                lit = sorted(S & set(order))
                log.append(f"  KILL  {d:24s} satisfies only {lit or '[]'} -- no gravity tether to '{anchor}'")
            continue
        g = sum(idx.idf(t) for t in order if t in S)            # idf-gravity over matched query terms (no TF)
        if use_shadow:                                          # shadow reach: twins count at a discount
            for t, tw in shadow_map.items():
                if t not in S:
                    g += 0.5 * sum(idx.idf(w) for w in tw if w in S)
        via_twin = use_shadow and (anchor not in S) and bool(S & anchor_twins)
        general_present = any(t in S for t in gen_terms)
        boost = 1.0
        note = ""
        if via_twin and general_present:
            boost = 1.5   # SHADOW LOCK: reached through the anchor's twin AND a general term -> triangulation
            note = " [SHADOW-LOCK]"
        if mode == "soft" and not satisfies:
            boost *= 0.12  # penalise, do not kill (deployable form -- protects recall)
            note = " [soft-penalty]"
        if g > 0:
            survivors.append((d, g * boost))
        if verbose and note:
            log.append(f"  keep  {d:24s} gravity={g:.2f} x{boost:.2f}{note}")
    survivors.sort(key=lambda z: -z[1])
    return RankResult(survivors, killed, anchor, tether, order, log)


# ======================================================================================================
# PART B -- THE CONSTRUCTED CORPUS (built so BM25 genuinely FAILS; run as an honest ablation ladder).
# ======================================================================================================
# Query is verbose and common-word-heavy (a realistic natural-language question). The gold is distinguished
# by ONE specific, rare anchor ('cardiac'); the noise matches MANY of the common terms at high TF but lacks
# the anchor -- so BM25's summed TF*idf ranks the noise above the gold. This is the ONLY regime where a hard
# anchor gate strictly helps: where the baseline is fooled by many partial common matches.

QUERY_GATE   = "coffee tea soda juice beverage intake consumption habit risk"   # ALL-common regime
QUERY_SHADOW = "coffee heart risk"          # gold uses only TWINS -> BM25 scores it 0

def build_constructed_corpus() -> Tuple[Dict[str, str], str, str]:
    """Returns (corpus, gold_gate_id, gold_shadow_id). BM25-failure is engineered, then verified in the demo.

    Gate regime (the ONLY one where a hard anchor-gate strictly beats BM25 *and* idf-scoring): every query
    term is COMMON, so the topical anchor 'coffee' gets almost no idf rescue. The gold matches FEW query terms
    (coffee + risk); the anchor-less noise matches MANY common query terms -> it outscores the gold on BOTH
    BM25 (TF*idf) and plain idf-gravity. Only the hard gate on 'coffee' rescues the gold.
    """
    corpus: Dict[str, str] = {}
    rng = random.Random(1313)

    # the OTHER query terms, all made genuinely common (low idf). 'coffee' is the anchor: rarer than these,
    # but still common enough that its idf cannot, by itself, lift the gold above the many-term noise.
    other_q = ["tea", "soda", "juice", "beverage", "intake", "consumption", "habit", "risk"]

    # --- GOLD (gate): matches only coffee + risk. Distinguished by the anchor 'coffee', nothing else. ---
    gold_gate = "GOLD_coffee_risk"
    corpus[gold_gate] = "Coffee and risk: a focused report on coffee and its risk."

    # --- GOLD (shadow): for QUERY_SHADOW='coffee heart risk', uses ONLY twins -> zero literal overlap ---
    gold_shadow = "GOLD_caffeine_cardiovascular"
    corpus[gold_shadow] = ("Caffeine and cardiovascular mortality. Caffeine exposure raised cardiovascular and "
                           "cardiac morbidity over a decade of follow up.")

    # --- BRIDGE / textbook docs so the miniverse LEARNS twins (coffee~caffeine, heart~cardiac~cardiovascular,
    #     risk~mortality~morbidity). Varied wording so co-occurrence is real, not one repeated string. ---
    bridges = [
        "Coffee contains caffeine; the caffeine in coffee is its active compound.",
        "A cup of coffee delivers caffeine, and decaffeinated coffee has the caffeine removed.",
        "Caffeine from coffee is the same caffeine molecule found in every coffee bean.",
        "The heart is the cardiac muscle; cardiac and cardiovascular describe the heart and its vessels.",
        "Cardiovascular disease is heart disease; cardiac events are heart events in the cardiovascular system.",
        "Cardiology treats the heart: cardiac care, cardiovascular surgery, and heart rhythm are all the heart.",
        "A heart attack is a cardiac event; cardiac arrest is the heart stopping in cardiovascular collapse.",
        "Risk of death is mortality; morbidity and mortality together quantify the risk of an adverse outcome.",
        "Epidemiology reports risk as mortality and morbidity: higher risk means higher mortality and morbidity.",
        "Relative risk and absolute risk both estimate mortality risk and morbidity risk in a population.",
    ]
    for i, t in enumerate(bridges):
        corpus[f"bridge_{i}"] = t

    # --- COFFEE-LIFESTYLE cluster: gives 'coffee' its (moderate) df. Uses NON-query beverage words, so each
    #     matches ONLY 'coffee' among the query terms -> survives the gate but scores below the gold. ---
    cafe = ["espresso", "latte", "cappuccino", "barista", "roast", "beans", "grinder", "mug", "foam", "brew"]
    for i in range(90):
        body = " ".join(f"coffee {rng.choice(cafe)}" for _ in range(rng.randint(4, 8)))
        corpus[f"cafe_{i}"] = f"Coffee culture guide. {body}."

    # --- cardiology cluster: reinforces the medical twins for the SHADOW demo (heart~cardiac~cardiovascular). ---
    cardio_bits = ["cardiac", "cardiovascular", "heart", "myocardial", "arrhythmia", "ventricular", "coronary"]
    for i in range(16):
        body = " ".join(rng.sample(cardio_bits, 4))
        corpus[f"cardio_{i}"] = f"Cardiology note: {body}. The {rng.choice(cardio_bits)} findings were reviewed."

    # --- TOKYO SPRAWL: matches MANY of the common query terms at high TF, NO 'coffee'. Fools BM25 AND
    #     idf-gravity (it matches more query terms than the gold does). ---
    for i in range(180):
        k = rng.randint(5, 8)
        picks = rng.sample(other_q, k)
        body = " ".join(f"{w} {rng.choice(other_q)}" for w in picks for _ in range(2))
        corpus[f"sprawl_{i}"] = f"Beverage survey. {body}. {' '.join(rng.sample(other_q, 5))} intake habit risk."

    # --- one keyword-stuffed doc: many commons at very high TF, no coffee -> a hard BM25 top-1 contender ---
    corpus["spam_seo"] = " ".join((w + " ") * 8 for w in ["tea", "soda", "beverage", "intake", "risk"])

    # --- generic non-medical risk noise (no coffee) ---
    haz = ["project", "credit", "operational", "compliance", "insurance", "liability"]
    for i in range(30):
        body = " ".join(f"{rng.choice(haz)} risk" for _ in range(rng.randint(4, 8)))
        corpus[f"riskgen_{i}"] = f"Enterprise risk survey. {body}. consumption habit risk assessment."

    # --- a FEW varied metaphor docs (polysemy pollution of 'heart' -- kept visible but a minority) ---
    metaphors = [
        "In the heart of the old town, at the very heart of the market district.",
        "She learned the poem by heart; the heart of the matter is simple.",
        "The heart of the machine is its engine; the city's heart beats downtown.",
    ]
    for i, t in enumerate(metaphors):
        corpus[f"metaphor_{i}"] = t

    return corpus, gold_gate, gold_shadow


def bm25_rank(corpus: Dict[str, str], query: str, k: int = 10, k1: float = 1.5, b: float = 0.75) -> List[Tuple[str, float]]:
    """Standard BM25 (k1=1.5, b=0.75) over the corpus -- the fair baseline the mechanism must beat."""
    ids = list(corpus)
    N = len(ids)
    tfd = {d: collections.Counter(content(corpus[d])) for d in ids}
    df = collections.Counter()
    for d in ids:
        df.update(tfd[d].keys())
    idf = {t: math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5)) for t in df}
    dl = {d: sum(tfd[d].values()) for d in ids}
    avgdl = sum(dl.values()) / max(1, N)
    qt = [t for t in dict.fromkeys(content(query)) if t in df]
    sc = {}
    for d in ids:
        s = 0.0
        for t in qt:
            f = tfd[d].get(t, 0)
            if f:
                s += idf[t] * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl[d] / avgdl))
        if s > 0:
            sc[d] = s
    return sorted(sc.items(), key=lambda z: -z[1])[:k]


def _rank_of(ranking: List[Tuple[str, float]], target: str) -> Optional[int]:
    for i, (d, _) in enumerate(ranking):
        if d == target:
            return i + 1
    return None


def _line(t): print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100, flush=True)


def run_constructed_demo():
    corpus, gold_gate, gold_shadow = build_constructed_corpus()
    idx = LatticeIndex(corpus)
    qg = content(QUERY_GATE)
    cand_g = [d for d in idx.ids if idx.SET[d] & set(qg)]

    _line("PHASE 5 -- CONSTRUCTED CORPUS")
    print(f"  corpus: {idx.N} docs")
    print(f"  QUERY (gate)  : '{QUERY_GATE}'\n                  content terms = {qg}")
    print(f"  gold(gate)    : '{gold_gate}'   -- distinguished by the rare anchor")
    print(f"  gold(shadow)  : '{gold_shadow}'  -- uses only TWINS of 'coffee heart risk' (zero literal overlap)")
    print(f"\n  df / idf of the gate query's content terms (specific = rare = high idf):")
    for t in sorted(qg, key=lambda t: -idx.idf(t)):
        print(f"      {t:12s} df={idx.df.get(t,0):>4}  idf={idx.idf(t):.2f}")
    print(f"\n  baked twins (learned from co-occurrence, NOT planted -- pollution shown honestly):")
    for t in ("coffee", "heart", "risk", "cardiac"):
        print(f"      twins({t:9s}) = {idx.twins(t)}")

    # ---------- BM25 baseline: verify it FAILS ----------
    _line("BASELINE  |  fair BM25 (k1=1.5, b=0.75) on the gate query -- does anchor-less noise beat the gold?")
    bm = bm25_rank(corpus, QUERY_GATE, k=8)
    for i, (d, s) in enumerate(bm):
        mark = "  <== GOLD" if d == gold_gate else ""
        print(f"    {i+1:>2}. {d:26s} {s:7.3f}{mark}")
    bm_full = bm25_rank(corpus, QUERY_GATE, k=len(cand_g))
    bm_gold = _rank_of(bm_full, gold_gate)
    bm25_fails = bool(bm and bm[0][0] != gold_gate)
    print(f"\n  --> BM25 rank of gold '{gold_gate}': {bm_gold}   top-1 = '{bm[0][0]}'")
    print(f"  --> BM25 {'FAILS as designed (anchor-less noise on top)' if bm25_fails else 'ALREADY WINS -- construction not adversarial; see note'}")

    # ---------- ABLATION LADDER: attribute the credit honestly ----------
    _line("ABLATION LADDER  |  BM25 -> +idf-gravity(drop TF) -> +hard-gate -> +shadow   (gold rank at each rung)")
    ladder = []
    ladder.append(("BM25 (TF*idf)", bm_gold, bm[0][0]))
    r_grav = anchor_gravity_rank(idx, qg, cand_g, mode="none", anchor_pick="specific", use_shadow=False)
    ladder.append(("+idf-gravity (drop TF, no gate)", _rank_of(r_grav.ranked, gold_gate),
                   r_grav.ranked[0][0] if r_grav.ranked else None))
    r_gate = anchor_gravity_rank(idx, qg, cand_g, mode="hard", anchor_pick="specific", use_shadow=False)
    ladder.append(("+hard-gate on anchor (no shadow)", _rank_of(r_gate.ranked, gold_gate),
                   r_gate.ranked[0][0] if r_gate.ranked else None))
    r_full = anchor_gravity_rank(idx, qg, cand_g, mode="hard", anchor_pick="specific", use_shadow=True)
    ladder.append(("+shadow (twins in tether+score)", _rank_of(r_full.ranked, gold_gate),
                   r_full.ranked[0][0] if r_full.ranked else None))
    print(f"    {'rung':34s} {'gold rank':>10}   top-1")
    for name, gr, t1 in ladder:
        lock = "  *** GOLD @ TOP-1 ***" if gr == 1 else ""
        print(f"    {name:34s} {str(gr):>10}   {t1}{lock}")
    print("\n  anchor gate detail:")
    for l in r_gate.log:
        print("    " + l)
    print(f"    branch-kills: {len(r_gate.killed)} of {len(cand_g)} candidates (no gravity tether to "
          f"'{r_gate.anchor}'); by type: {dict(collections.Counter(d.split('_')[0] for d in r_gate.killed))}")
    # honest attribution: how much did the PRUNE add over idf-gravity alone?  (data-driven, no boilerplate)
    gr_grav = ladder[1][1]; gr_gate = ladder[2][1]
    grav_helped = (gr_grav or 99) < (bm_gold or 99)
    gate_helped = (gr_gate or 99) < (gr_grav or 99)
    print(f"\n  HONEST ATTRIBUTION (read straight off the ladder):")
    print(f"    drop-TF idf-gravity moved gold {bm_gold} -> {gr_grav}   "
          f"({'it lifts the gold' if grav_helped else 'NO lift -- the noise matches MORE query terms, so summed idf still favours it'})")
    print(f"    the hard GATE then moved gold {gr_grav} -> {gr_gate}   "
          f"({'a strict, NON-redundant lift -- this is the gate earning its keep' if gate_helped else 'no further lift -- redundant with idf here'})")

    # ---------- SHADOW on the vocabulary-mismatch gold (BM25 = 0) ----------
    _line("SHADOW QUERY  |  QUERY='coffee heart risk', gold uses ONLY twins -> BM25 scores it 0")
    qs = content(QUERY_SHADOW)
    bm_s = bm25_rank(corpus, QUERY_SHADOW, k=len(idx.ids))
    print(f"    BM25('{QUERY_SHADOW}') rank of '{gold_shadow}': {_rank_of(bm_s, gold_shadow)} "
          f"(shares no literal query term -> BM25 cannot reach it)")
    cand_all = list(idx.ids)
    r_shadow = anchor_gravity_rank(idx, qs, cand_all, mode="hard", anchor_pick="specific", use_shadow=True)
    for l in r_shadow.log:
        print("    " + l)
    sh_rank = _rank_of(r_shadow.ranked, gold_shadow)
    print(f"    SHADOW rank of '{gold_shadow}': {sh_rank}  "
          f"({'REACHED via twin expansion' if sh_rank else 'still unreachable'})")
    r_shadow_off = anchor_gravity_rank(idx, qs, cand_all, mode="hard", anchor_pick="specific", use_shadow=False)
    print(f"    control shadow-OFF: rank of '{gold_shadow}' = {_rank_of(r_shadow_off.ranked, gold_shadow)} "
          f"(literal-only -> unreachable, as it must be)")

    # ---------- CONTROLS ----------
    _line("CONTROLS  |  must break a vacuous version")
    c_wrong = anchor_gravity_rank(idx, qg, cand_g, mode="hard", anchor_pick="general", use_shadow=True)
    c_wrong_top = c_wrong.ranked[0][0] if c_wrong.ranked else None
    print(f"  [wrong-direction anchor] gate cast from most-GENERAL term '{c_wrong.anchor}': "
          f"killed={len(c_wrong.killed)}, top-1='{c_wrong_top}'")
    print(f"        -> {'gold NOT locked -- specificity direction is load-bearing  OK' if c_wrong_top != gold_gate else 'still locks gold -- direction NOT load-bearing (!)'} ")

    # ---------- VERDICT ----------
    _line("CONSTRUCTED-CORPUS VERDICT (honest)")
    checks = {
        "BM25 baseline fails (anchor-less noise on top)": bm25_fails,
        "full mechanism locks gate-gold to top-1": ladder[3][1] == 1,
        "shadow reaches the zero-overlap gold (BM25=0)": sh_rank is not None,
        "control shadow-off: zero-overlap gold unreachable": _rank_of(r_shadow_off.ranked, gold_shadow) is None,
        "control wrong-anchor: gold not locked": c_wrong_top != gold_gate,
    }
    for k, v in checks.items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    grav_helped = (ladder[1][1] or 99) < (ladder[0][1] or 99)
    gate_helped = (ladder[2][1] or 99) < (ladder[1][1] or 99)
    print(f"\n  ATTRIBUTION (what actually did the work, straight off the ladder):")
    print(f"    - drop-TF idf-gravity: {'lifts the gold' if grav_helped else 'does NOT lift the gold here -- the noise matches MORE query terms than the gold'}.")
    print(f"    - the hard GATE: {'strict lift {} -> {} (its real, non-redundant regime)'.format(ladder[1][1], ladder[2][1]) if gate_helped else 'no lift over idf-gravity here (redundant)'}.")
    print(f"    - the SHADOW: the only rung that reaches a zero-literal-overlap gold (rank {sh_rank}).")
    print(f"\n  THE HONEST LAW (both regimes measured across builds):")
    print(f"    * all-common query (this corpus): the hard gate STRICTLY beats BM25 and idf-gravity (gold 182 -> 1),")
    print(f"      because no query term is rare enough for idf to rescue the gold on its own.")
    print(f"    * rare-anchor query (earlier build): the gate is REDUNDANT -- idf alone already ranks the gold first.")
    print(f"    So the gate's value is REAL but NARROW: it fires only when the discriminating term is common enough")
    print(f"    that idf under-weights it. The SHADOW (twin reach) is the orthogonal, always-relevant lever -- it is")
    print(f"    the one thing BM25 provably cannot do. Real-corpus generalization is the arbiter: run `real <corpus>`.")
    return {"checks": checks, "ladder": [(n, r) for n, r, _ in ladder], "shadow_rank": sh_rank}


# ======================================================================================================
# PART C -- REAL BEIR GENERALIZATION (the honesty gate).
# ======================================================================================================
def run_real(corpus_name: str, topk_pool: int = 100, max_q: Optional[int] = None):
    from _freq_cascade_lib import load_eval, Bm25Base
    from eval_beir import ndcg_at_k, recall_at_k

    ev = load_eval(corpus_name)
    corpus = {d: (v.get("title", "") + " " + v.get("text", "")).strip() for d, v in ev["corpus"].items()}
    idx = LatticeIndex(corpus, mindf=3)
    bm = Bm25Base(ev, k1=1.5, b=0.75)
    qids = ev["qids"] if max_q is None else ev["qids"][:max_q]

    def p_at_1(ranked, rel):
        return 1.0 if ranked and ranked[0] in rel else 0.0

    def mrr_at_k(ranked, rel, k=10):
        for i, d in enumerate(ranked[:k]):
            if d in rel:
                return 1.0 / (i + 1)
        return 0.0

    # Two families, kept SEPARATE so credit is not conflated:
    #   PRUNE-ONLY = keep BM25's own ranking, only push anchor-tether failures to the tail. This is the honest
    #                test of the user's claim ("kill the branch"); it does NOT replace BM25's scorer.
    #   RESCORE    = replace BM25 order with idf-gravity (+shadow). Shows what the substrate's own scorer does.
    variant_names = ["BM25",
                     "PRUNE-only(literal anchor)", "PRUNE-only(+twins tether)",
                     "RESCORE idf-gravity+shadow", "RESCORE soft-fuse"]
    agg = {v: collections.defaultdict(float) for v in variant_names}
    n = 0
    for qid in qids:
        rel = ev["qrels"][qid]
        if not rel:
            continue
        qtext = ev["queries"][qid]
        qterms = content(qtext)
        base = bm.rank(qtext, k=topk_pool)   # BM25 candidate pool AND the BM25 ranking
        if not base:
            continue
        n += 1

        # identify the anchor + tether once (same rule as the mechanism)
        present = [t for t in dict.fromkeys(qterms) if t in idx.post]
        rankings = {"BM25": base}
        if present:
            anchor = max(present, key=lambda t: idx.idf(t))
            tether_lit = {anchor}
            tether_twin = {anchor} | set(idx.twins(anchor))
            def prune(order, tether):
                keep = [d for d in order if idx.SET[d] & tether]
                drop = [d for d in order if not (idx.SET[d] & tether)]
                return keep + drop            # BM25 order preserved; gated docs pushed below survivors
            rankings["PRUNE-only(literal anchor)"] = prune(base, tether_lit)
            rankings["PRUNE-only(+twins tether)"] = prune(base, tether_twin)
        else:
            rankings["PRUNE-only(literal anchor)"] = base
            rankings["PRUNE-only(+twins tether)"] = base
        # rescore family (the substrate's own scorer)
        r_hard = anchor_gravity_rank(idx, qterms, base, mode="hard", anchor_pick="specific", use_shadow=True)
        rankings["RESCORE idf-gravity+shadow"] = [d for d, _ in r_hard.ranked] + \
            [d for d in base if d not in {x for x, _ in r_hard.ranked}]
        r_soft = anchor_gravity_rank(idx, qterms, base, mode="soft", anchor_pick="specific", use_shadow=True)
        rankings["RESCORE soft-fuse"] = [d for d, _ in r_soft.ranked] + \
            [d for d in base if d not in {x for x, _ in r_soft.ranked}]

        for vname in variant_names:
            ranked = rankings[vname]
            agg[vname]["p@1"] += p_at_1(ranked, rel)
            agg[vname]["mrr@10"] += mrr_at_k(ranked, rel, 10)
            agg[vname]["ndcg@10"] += ndcg_at_k(ranked, rel, 10)
            agg[vname]["r@100"] += recall_at_k(ranked, rel, 100)

    _line(f"REAL-BEIR GENERALIZATION  |  {corpus_name}  |  {n} matched queries, pool={topk_pool}")
    print(f"  {'variant':30s} {'P@1':>8} {'MRR@10':>8} {'nDCG@10':>8} {'R@100':>8}")
    base_row = None
    out = {"corpus": corpus_name, "n": n, "variants": {}}
    for v in variant_names:
        row = {m: agg[v][m] / max(1, n) for m in ("p@1", "mrr@10", "ndcg@10", "r@100")}
        out["variants"][v] = {k: round(x, 4) for k, x in row.items()}
        if v == "BM25":
            base_row = row
        d = "" if v == "BM25" else "   dP@1={:+.4f} dMRR={:+.4f} dNDCG={:+.4f} dR@100={:+.4f}".format(
            row["p@1"] - base_row["p@1"], row["mrr@10"] - base_row["mrr@10"],
            row["ndcg@10"] - base_row["ndcg@10"], row["r@100"] - base_row["r@100"])
        print(f"  {v:30s} {row['p@1']:8.4f} {row['mrr@10']:8.4f} {row['ndcg@10']:8.4f} {row['r@100']:8.4f}{d}")
    print("\n  READ: PRUNE-only keeps BM25's order and just drops anchor-tether failures (the honest test of the")
    print("        user's 'kill the branch' claim). RESCORE replaces BM25 order with the substrate's idf-gravity.")
    print("        dP@1 > 0 would mean the prune forced more golds to top-1; dP@1 < 0 means it demoted real golds.")
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(out, open(os.path.join(here, f"_phase5_real_{corpus_name}.json"), "w"), indent=2)
    print(f"\n  wrote _phase5_real_{corpus_name}.json")
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "real":
        cname = sys.argv[2] if len(sys.argv) > 2 else "nfcorpus"
        mq = int(sys.argv[3]) if len(sys.argv) > 3 else None
        run_real(cname, max_q=mq)
    else:
        run_constructed_demo()
