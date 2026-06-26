"""
Lattice-native lexical scoring — no BM25 denominator.

Maps the append-only multi-view index to formula pieces:
  - TF saturation:  tf / (tf + a)     Zeno-blocked, one address per term
  - Length norm:    (avg_kappa_card / doc_kappa_card) ** lpow
  - IDF:            log df on prime postings (+ optional pi-depth on word gear)
  - Pair meets:     bonus when query word-prime pairs co-occur in doc
  - Plane bands:    4 |z| bands x 8 wings (32 total) — band alignment bonus

BM25 is optional (lexical_mode='bm25') for A/B only.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Literal

from aethos_append_index import AppendOnlyLatticeIndex
from aethos_lattice import BranchKind
from aethos_physics import SpacetimeCell
from aethos_words import letter_to_prime
from pipeline.bit_01_word_cell import DEFAULT_ANCHOR_N
from pipeline.bit_05_z_band import band_profile_for_cell, wing_band_map


class LexicalMode(str, Enum):
    BM25 = "bm25"
    APPEND_INDEX = "append_index"  # multiview BM25 — hybrid L0 floor (~0.7 SciFact)
    LATTICE_PURE = "lattice_pure"
    LATTICE_PLANE = "lattice_plane"  # pure + 32-wing band alignment


# Four |z| bands — slots on the 3D plane (BIT 5); each hosts 8 wings = 32 addresses.
PLANE_BAND_ROLE: dict[int, str] = {
    0: "va1_hub_anchor",
    1: "va2_spring_pair",
    2: "va3_branch",
    3: "va4_branch",
}


def word_letter_chain(word: str) -> tuple[int, ...]:
    primes: set[int] = set()
    for c in word.lower():
        if c.isalpha():
            try:
                primes.add(letter_to_prime(c))
            except ValueError:
                continue
    return tuple(sorted(primes))


def word_band_profile(
    word: str,
    *,
    n: float = DEFAULT_ANCHOR_N,
    wing: int = 1,
) -> tuple[int, float]:
    """Band id (0..3) and |z| modulus for a word via letter-prime chain on the plane."""
    chain = word_letter_chain(word)
    if not chain:
        return 0, 0.0
    cell = SpacetimeCell.at(chain, n, BranchKind.VA1, wing)
    prof = band_profile_for_cell(cell, wing=wing)
    return prof.band_id, prof.z_modulus


@dataclass
class LatticeLexicalConfig:
    mode: LexicalMode = LexicalMode.LATTICE_PLANE
    sat_a: float = 1.0
    lpow: float = 0.35
    pi_alpha: float = 0.5
    pair_w: float = 0.3
    band_w: float = 0.12
    anchor_n: float = DEFAULT_ANCHOR_N
    wing: int = 1


@dataclass
class LatticeLexicalScorer:
    """Score documents from prime postings using lattice formula — not BM25."""

    idx: AppendOnlyLatticeIndex
    config: LatticeLexicalConfig = field(default_factory=LatticeLexicalConfig)
    _doc_card: dict[str, int] = field(default_factory=dict)
    _avg_card: float = 1.0
    _word_band: dict[str, int] = field(default_factory=dict)
    _p2w: dict[int, str] = field(default_factory=dict)
    _wing_band: dict[int, int] = field(default_factory=dict)
    _dense_query: str | None = field(default=None, repr=False)
    _dense_vec: object | None = field(default=None, repr=False)

    def bind(self) -> LatticeLexicalScorer:
        N = max(1, len(self.idx.alive))
        doc_card: Counter[str] = Counter()
        for p, plist in self.idx.postings.items():
            for d in plist:
                if d in self.idx.alive:
                    doc_card[d] += 1
        self._doc_card = dict(doc_card)
        self._avg_card = sum(doc_card.values()) / N if doc_card else 1.0

        self._p2w = {
            p: tok[1]
            for tok, p in self.idx.token_prime.items()
            if tok[0] == "w"
        }

        if self.config.mode == LexicalMode.LATTICE_PLANE:
            for w in self._p2w.values():
                if w not in self._word_band:
                    bid, _ = word_band_profile(
                        w, n=self.config.anchor_n, wing=self.config.wing,
                    )
                    self._word_band[w] = bid
            # reference 32-wing → band map at anchor (glass-box constant)
            ref_chain = word_letter_chain("anchor") or (2, 3, 5)
            self._wing_band = wing_band_map(ref_chain, self.config.anchor_n)

        return self

    def _pi_depth(self, tok: tuple[str, str]) -> float:
        if tok[0] != "w":
            return 2.0
        return float(len(set(tok[1])))

    def _idf(self, p: int, N: int, tok: tuple[str, str]) -> float:
        dfp = self.idx.df.get(p, 0)
        if dfp == 0:
            return 0.0
        idf = math.log(1 + (N - dfp + 0.5) / (dfp + 0.5))
        if self.config.pi_alpha and tok[0] == "w":
            idf *= 1.0 + self.config.pi_alpha * self._pi_depth(tok) / 10.0
        return idf

    def score_bm25(self, query: str) -> dict[str, float]:
        raw = self.idx._score(query)
        return dict(raw)

    def cache_dense_scores(self, query: str) -> None:
        """One dense BM25 vector per query — reused by pool union + score_pool."""
        if self.idx._dense_ready:
            self._dense_vec = self.idx._dense_score_array(query)
            self._dense_query = query
        else:
            self._dense_vec = None
            self._dense_query = None

    def append_top_k(self, k: int) -> list[str]:
        """Top-k from cached dense vector; requires ``cache_dense_scores`` first."""
        if self._dense_vec is not None and self.idx._dense_ready:
            return self.idx.top_k_dense(self._dense_vec, k)
        return []

    def score_bm25_pool(
        self, query: str, cand: set[str] | frozenset[str]
    ) -> dict[str, float]:
        """Multiview BM25 on a bounded candidate set — O(query_terms × |cand|)."""
        if not cand:
            return {}
        if (
            self.idx._dense_ready
            and self._dense_query == query
            and self._dense_vec is not None
        ):
            return self.idx.pool_scores_dense(self._dense_vec, cand)
        idx = self.idx
        N = max(1, len(idx.alive))
        avgdl = idx._total_len / N
        qbag = idx._multiview(query)
        k1, b = idx.k1, idx.b
        A, Bc, k1p1 = k1 * (1 - b), k1 * b / avgdl, k1 + 1
        df, postings, doc_len = idx.df, idx.postings, idx.doc_len
        tri_cap = idx.tri_df_frac * N if idx.tri_df_frac < 1.0 else None
        delta_base = idx.bm25_delta
        scores: dict[str, float] = {}
        cand_list = list(cand)
        for tok, qwt in qbag.items():
            p = idx.token_prime.get(tok)
            if p is None:
                continue
            dfp = df.get(p, 0)
            if dfp == 0:
                continue
            if tri_cap is not None and tok[0] == "3" and dfp > tri_cap:
                continue
            idf = math.log(1 + (N - dfp + 0.5) / (dfp + 0.5))
            delta = delta_base if tok[0] == "w" else 0.0
            pl = postings[p]
            cf = qwt * idf
            for d in cand_list:
                tf = pl.get(d)
                if not tf:
                    continue
                norm = tf * k1p1 / (tf + A + Bc * doc_len[d])
                scores[d] = scores.get(d, 0.0) + cf * (norm + delta)
        return scores

    def score_lattice(self, query: str) -> dict[str, float]:
        cfg = self.config
        idx = self.idx
        N = max(1, len(idx.alive))
        qbag = idx._multiview(query)
        scores: dict[str, float] = defaultdict(float)
        q_word_prims: list[int] = []
        q_bands: Counter[int] = Counter()

        if cfg.mode == LexicalMode.LATTICE_PLANE:
            for tok, _ in qbag.items():
                if tok[0] == "w":
                    bid = self._word_band.get(tok[1])
                    if bid is not None:
                        q_bands[bid] += 1

        for tok, qwt in qbag.items():
            p = idx.token_prime.get(tok)
            if p is None or idx.df.get(p, 0) == 0:
                continue
            idf = self._idf(p, N, tok)
            if tok[0] == "w":
                q_word_prims.append(p)
            for d, tf in idx.postings[p].items():
                if d not in idx.alive:
                    continue
                sat = tf / (tf + cfg.sat_a)
                card = self._doc_card.get(d, 1)
                lennorm = (self._avg_card / card) ** cfg.lpow if card else 1.0
                scores[d] += qwt * idf * sat * lennorm
                if cfg.mode == LexicalMode.LATTICE_PLANE and tok[0] == "w":
                    bid = self._word_band.get(tok[1])
                    if bid is not None and q_bands[bid]:
                        scores[d] += cfg.band_w * idf * q_bands[bid]

        if cfg.pair_w and len(q_word_prims) >= 2:
            for a, b in combinations(set(q_word_prims), 2):
                da = idx.postings.get(a, {})
                db = idx.postings.get(b, {})
                bonus = cfg.pair_w * (idx._idf(a, N) + idx._idf(b, N)) / 2.0
                for d in da.keys() & db.keys():
                    if d in idx.alive:
                        scores[d] += bonus

        return dict(scores)

    def score_pool(self, query: str, cand: set[str] | frozenset[str]) -> dict[str, float]:
        """Score a bounded candidate set — O(query_terms × |cand|)."""
        if self.config.mode in (LexicalMode.BM25, LexicalMode.APPEND_INDEX):
            return self.score_bm25_pool(query, cand)
        if not cand:
            return {}
        cfg = self.config
        idx = self.idx
        N = max(1, len(idx.alive))
        qbag = idx._multiview(query)
        scores: dict[str, float] = {}
        cand_set = set(cand)
        cand_list = list(cand_set)
        q_word_prims: list[int] = []
        q_bands: Counter[int] = Counter()

        if cfg.mode == LexicalMode.LATTICE_PLANE:
            for tok, _ in qbag.items():
                if tok[0] == "w":
                    bid = self._word_band.get(tok[1])
                    if bid is not None:
                        q_bands[bid] += 1

        for tok, qwt in qbag.items():
            p = idx.token_prime.get(tok)
            if p is None or idx.df.get(p, 0) == 0:
                continue
            idf = self._idf(p, N, tok)
            if tok[0] == "w":
                q_word_prims.append(p)
            pl = idx.postings.get(p, {})
            for d in cand_list:
                tf = pl.get(d)
                if not tf:
                    continue
                sat = tf / (tf + cfg.sat_a)
                card = self._doc_card.get(d, 1)
                lennorm = (self._avg_card / card) ** cfg.lpow if card else 1.0
                scores[d] = scores.get(d, 0.0) + qwt * idf * sat * lennorm
                if cfg.mode == LexicalMode.LATTICE_PLANE and tok[0] == "w":
                    bid = self._word_band.get(tok[1])
                    if bid is not None and q_bands[bid]:
                        scores[d] += cfg.band_w * idf * q_bands[bid]

        if cfg.pair_w and len(q_word_prims) >= 2:
            for a, b in combinations(set(q_word_prims), 2):
                da = idx.postings.get(a, {})
                db = idx.postings.get(b, {})
                bonus = cfg.pair_w * (idx._idf(a, N) + idx._idf(b, N)) / 2.0
                for d in da.keys() & db.keys():
                    if d in cand_set and d in idx.alive:
                        scores[d] = scores.get(d, 0.0) + bonus

        return scores

    def score(self, query: str) -> dict[str, float]:
        if self.config.mode in (LexicalMode.BM25, LexicalMode.APPEND_INDEX):
            return self.score_bm25(query)
        return self.score_lattice(query)

    def top_k(self, query: str, k: int = 10) -> list[str]:
        scores = self.score(query)
        if not scores:
            return []
        return sorted(scores, key=lambda d: scores[d], reverse=True)[:k]

    def explain_query(self, query: str) -> dict:
        """Glass-box: which plane bands and formula terms fire for a query."""
        qbag = self.idx._multiview(query)
        terms = []
        for tok, qwt in qbag.items():
            if tok[0] != "w":
                continue
            w = tok[1]
            bid, zmod = word_band_profile(
                w, n=self.config.anchor_n, wing=self.config.wing,
            )
            terms.append({
                "word": w,
                "weight": round(qwt, 3),
                "band_id": bid,
                "band_role": PLANE_BAND_ROLE.get(bid, "unknown"),
                "z_modulus": round(zmod, 4),
            })
        return {
            "lexical_mode": self.config.mode.value,
            "plane_bands": PLANE_BAND_ROLE,
            "wing_band_slots": len(self._wing_band),
            "query_word_bands": terms,
        }


def lattice_lexical_scorer(
    idx: AppendOnlyLatticeIndex,
    *,
    mode: Literal["bm25", "append_index", "lattice_pure", "lattice_plane"] = "lattice_plane",
    **kwargs,
) -> LatticeLexicalScorer:
    cfg = LatticeLexicalConfig(mode=LexicalMode(mode))
    for key, val in kwargs.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return LatticeLexicalScorer(idx=idx, config=cfg).bind()
