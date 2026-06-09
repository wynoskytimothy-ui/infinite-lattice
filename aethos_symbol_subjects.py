"""
31 subject chambers + 1 master (k=0) — sub-quadrant correlation routing.

k=0   Master / subconscious — all correlations, cross-reference audit
k=1..31  Subject slices — same primes, rotated chamber-local edges

Usage::

    from aethos_symbol_subjects import (
        MASTER_CHAMBER,
        subjects_for_dataset,
        infer_doc_subjects,
        vote_query_chambers,
    )

    tags = subjects_for_dataset("scifact")   # {1, 9, 10}
    idx.ingest_corpus(corpus, tags)
    chambers = vote_query_chambers("apple phone chip")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from aethos_lattice import BranchKind

if TYPE_CHECKING:
    from aethos_symbol_knowledge import CrossLink, SymbolKnowledgeIndex

_TOKEN_RE = re.compile(r"[a-z]+")

MASTER_CHAMBER = 0
NUM_CHAMBERS = 32
SUBJECT_CHAMBER_MIN = 1
SUBJECT_CHAMBER_MAX = 31

# k -> (branch 1..4, wing 1..8, name)
SUBJECT_REGISTRY: dict[int, tuple[int, int, str]] = {
    1: (1, 1, "physics_astronomy"),
    2: (1, 2, "chemistry_materials"),
    3: (1, 3, "mathematics_statistics"),
    4: (1, 4, "computing_software"),
    5: (1, 5, "engineering_technology"),
    6: (1, 6, "earth_climate"),
    7: (1, 7, "energy_industry"),
    8: (1, 8, "measurement_instrumentation"),
    9: (2, 1, "biology_genetics"),
    10: (2, 2, "medicine_clinical"),
    11: (2, 3, "nutrition_diet"),
    12: (2, 4, "pharmacology_biochemistry"),
    13: (2, 5, "neuroscience_psychology"),
    14: (2, 6, "agriculture_food"),
    15: (2, 7, "ecology_environment"),
    16: (2, 8, "veterinary_animal"),
    17: (3, 1, "law_regulation"),
    18: (3, 2, "economics_finance"),
    19: (3, 3, "business_management"),
    20: (3, 4, "politics_government"),
    21: (3, 5, "history_archaeology"),
    22: (3, 6, "education_pedagogy"),
    23: (3, 7, "sociology_culture"),
    24: (3, 8, "religion_ethics"),
    25: (4, 1, "linguistics_grammar"),
    26: (4, 2, "literature_writing"),
    27: (4, 3, "arts_media"),
    28: (4, 4, "music_performance"),
    29: (4, 5, "sports_recreation"),
    30: (4, 6, "geography_travel"),
    31: (4, 7, "daily_life_consumer"),
}

DATASET_SUBJECTS: dict[str, frozenset[int]] = {
    "scifact": frozenset({1, 9, 10}),
    "nfcorpus": frozenset({10, 11, 14}),
    "fiqa": frozenset({18, 19}),
    "trec-covid": frozenset({10, 9, 15}),
    "arguana": frozenset({17, 20, 24}),
    "quora": frozenset({25, 23, 31}),
}

# Lightweight keyword hints per chamber (first match wins in tie-break)
_SUBJECT_KEYWORDS: dict[int, frozenset[str]] = {
    1: frozenset({"quantum", "particle", "gravity", "astronomy", "cosmic", "photon"}),
    2: frozenset({"chemical", "molecule", "catalyst", "polymer", "reaction"}),
    3: frozenset({"theorem", "algebra", "statistics", "probability", "equation"}),
    4: frozenset({"software", "algorithm", "computer", "chip", "processor", "phone", "silicon"}),
    5: frozenset({"engineer", "mechanical", "circuit", "device", "technical"}),
    6: frozenset({"climate", "weather", "geology", "seismic", "atmosphere"}),
    7: frozenset({"energy", "power", "fuel", "battery", "electric"}),
    8: frozenset({"sensor", "measurement", "instrument", "calibration"}),
    9: frozenset({"gene", "dna", "rna", "cell", "organism", "species"}),
    10: frozenset({"clinical", "patient", "trial", "disease", "cancer", "therapy", "diagnosis"}),
    11: frozenset({"nutrition", "diet", "vitamin", "calorie", "food", "pie", "fruit", "orchard"}),
    12: frozenset({"drug", "pharma", "biochemistry", "enzyme", "metabolism"}),
    13: frozenset({"brain", "neural", "cognitive", "psychology", "mental"}),
    14: frozenset({"crop", "farm", "agriculture", "harvest", "soil"}),
    15: frozenset({"ecology", "ecosystem", "wildlife", "pollution", "biodiversity"}),
    16: frozenset({"animal", "veterinary", "livestock", "pet"}),
    17: frozenset({"law", "legal", "court", "statute", "regulation"}),
    18: frozenset({"finance", "stock", "market", "investment", "bank", "loan", "fiqa"}),
    19: frozenset({"business", "company", "management", "corporate", "revenue"}),
    20: frozenset({"political", "government", "policy", "election", "democracy"}),
    21: frozenset({"history", "ancient", "archaeology", "century", "war"}),
    22: frozenset({"education", "student", "school", "learning", "pedagogy"}),
    23: frozenset({"culture", "society", "social", "community", "ethnic"}),
    24: frozenset({"religion", "ethics", "moral", "faith", "spiritual"}),
    25: frozenset({"grammar", "syntax", "linguistic", "language", "phoneme"}),
    26: frozenset({"literature", "novel", "poetry", "author", "writing"}),
    27: frozenset({"art", "film", "media", "cinema", "design"}),
    28: frozenset({"music", "melody", "concert", "instrument", "song"}),
    29: frozenset({"sport", "athlete", "game", "fitness", "team"}),
    30: frozenset({"geography", "travel", "city", "region", "country"}),
    31: frozenset({"consumer", "daily", "home", "shopping", "recipe"}),
}


def normalize_subjects(subjects: int | Sequence[int] | set[int] | None) -> frozenset[int]:
    """Validate subject ids (1..31); empty set allowed."""
    if subjects is None:
        return frozenset()
    if isinstance(subjects, int):
        subjects = [subjects]
    out: set[int] = set()
    for k in subjects:
        if k == MASTER_CHAMBER:
            continue
        if not SUBJECT_CHAMBER_MIN <= k <= SUBJECT_CHAMBER_MAX:
            raise ValueError(f"subject chamber must be 1..31, got {k}")
        out.add(k)
    return frozenset(out)


def chambers_for_ingest(subjects: int | Sequence[int] | set[int] | None) -> frozenset[int]:
    """Subject tags plus master — always train k=0."""
    return normalize_subjects(subjects) | {MASTER_CHAMBER}


def chamber_to_branch_wing(k: int) -> tuple[BranchKind, int]:
    """Map flat chamber k (0..31) to (branch, wing) for plane rotation."""
    if k == MASTER_CHAMBER:
        return BranchKind.VA1, 1
    k = max(SUBJECT_CHAMBER_MIN, min(SUBJECT_CHAMBER_MAX, k))
    branch_idx = (k - 1) // 8 + 1
    wing = (k - 1) % 8 + 1
    return BranchKind(branch_idx), wing


def subject_name(k: int) -> str:
    if k == MASTER_CHAMBER:
        return "master_subconscious"
    entry = SUBJECT_REGISTRY.get(k)
    return entry[2] if entry else f"subject_{k}"


def subjects_for_dataset(name: str) -> frozenset[int]:
    """Default subject tags for a BEIR / corpus name (empty if unknown)."""
    key = name.lower().split("+")[0].strip()
    return DATASET_SUBJECTS.get(key, frozenset())


def infer_doc_subjects(
    text: str,
    *,
    fallback: frozenset[int] | None = None,
    max_subjects: int = 3,
) -> frozenset[int]:
    """Keyword vote → 1..3 subject chambers for a single document."""
    tokens = set(_TOKEN_RE.findall(text.lower()))
    scores: dict[int, int] = {}
    for k, keywords in _SUBJECT_KEYWORDS.items():
        hit = len(tokens & keywords)
        if hit:
            scores[k] = hit
    if not scores:
        return fallback or frozenset()
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    top = frozenset(k for k, _ in ranked[:max_subjects])
    return top


def vote_query_chambers(
    words: Sequence[str],
    *,
    max_chambers: int = 3,
    min_score: int = 1,
) -> frozenset[int]:
    """
    Pick subject chambers from query tokens (brain routing).

    Returns subject ids only (not master). Caller adds MASTER_CHAMBER as fallback.
    """
    tokens = set(w.lower() for w in words if len(w) >= 2)
    scores: dict[int, int] = {}
    for k, keywords in _SUBJECT_KEYWORDS.items():
        hit = len(tokens & keywords)
        if hit >= min_score:
            scores[k] = hit
    if not scores:
        return frozenset()
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return frozenset(k for k, _ in ranked[:max_chambers])


@dataclass(frozen=True)
class ChamberConflict:
    anchor: str
    chamber_a: int
    chamber_b: int
    neighbors_a: tuple[str, ...]
    neighbors_b: tuple[str, ...]
    overlap: tuple[str, ...]
    kind: str  # "disambiguated" | "smear" | "orphan_global"


def _top_neighbors(
    knowledge: SymbolKnowledgeIndex,
    word: str,
    chamber: int,
    *,
    limit: int = 8,
    min_strength: float = 1.0,
) -> list[str]:
    knowledge._ensure_chamber(chamber)
    nbrs: list[tuple[str, float]] = []
    for lk in knowledge.neighbors(word, chamber=chamber, kinds={"direct"}):
        other = lk.right if lk.left == word.lower() else lk.left
        if lk.strength >= min_strength:
            nbrs.append((other, lk.strength))
    nbrs.sort(key=lambda x: -x[1])
    return [w for w, _ in nbrs[:limit]]


def master_audit(
    knowledge: SymbolKnowledgeIndex,
    *,
    min_direct_strength: float = 2.0,
    smear_overlap_min: int = 2,
) -> dict[str, object]:
    """
    Master chamber cross-reference — detect mis-correlations across subjects.

    Flags anchors whose rare neighbors overlap heavily in incompatible chambers.
    """
    subject_chambers = sorted(knowledge.subject_chambers_active())
    conflicts: list[dict[str, object]] = []
    disambiguated: list[dict[str, object]] = []
    orphans: list[dict[str, object]] = []

    anchors: set[str] = set()
    for k in subject_chambers:
        knowledge._ensure_chamber(k)
        for key in knowledge.chamber_links.get(k, {}):
            anchors.add(key[0])
            anchors.add(key[1])
    if not anchors:
        for ev in getattr(knowledge, "_doc_evidence", {}).values():
            if ev.chambers & set(subject_chambers):
                for key in ev.pairs:
                    anchors.add(key[0])
                    anchors.add(key[1])

    for anchor in sorted(anchors):
        chamber_nbrs: dict[int, list[str]] = {}
        for k in subject_chambers:
            nbrs = _top_neighbors(
                knowledge, anchor, k, min_strength=min_direct_strength,
            )
            if nbrs:
                chamber_nbrs[k] = nbrs

        if len(chamber_nbrs) < 2:
            continue

        keys = sorted(chamber_nbrs.keys())
        for i, ka in enumerate(keys):
            for kb in keys[i + 1 :]:
                na = set(chamber_nbrs[ka])
                nb = set(chamber_nbrs[kb])
                overlap = tuple(sorted(na & nb))
                entry = {
                    "anchor": anchor,
                    "chamber_a": ka,
                    "chamber_b": kb,
                    "name_a": subject_name(ka),
                    "name_b": subject_name(kb),
                    "neighbors_a": chamber_nbrs[ka],
                    "neighbors_b": chamber_nbrs[kb],
                    "overlap": list(overlap),
                }
                if len(overlap) >= smear_overlap_min:
                    entry["kind"] = "smear"
                    conflicts.append(entry)
                else:
                    entry["kind"] = "disambiguated"
                    disambiguated.append(entry)

        master_nbrs = _top_neighbors(
            knowledge, anchor, MASTER_CHAMBER, min_strength=min_direct_strength,
        )
        subj_union: set[str] = set()
        for nbr_list in chamber_nbrs.values():
            subj_union.update(nbr_list)
        if master_nbrs and not subj_union:
            orphans.append({
                "anchor": anchor,
                "master_neighbors": master_nbrs,
                "kind": "orphan_global",
            })

    return {
        "n_subject_chambers": len(subject_chambers),
        "n_anchors_checked": len(anchors),
        "n_disambiguated": len(disambiguated),
        "n_smear_conflicts": len(conflicts),
        "n_orphan_global": len(orphans),
        "smear_conflicts": conflicts[:100],
        "disambiguated_samples": disambiguated[:30],
        "orphan_samples": orphans[:30],
    }


def write_master_audit(
    knowledge: SymbolKnowledgeIndex,
    path: str | Path | None = None,
) -> Path:
    """Run master_audit and write logs/chamber_conflicts.json."""
    report = master_audit(knowledge)
    out = Path(path) if path else (
        Path(__file__).resolve().parent / "logs" / "chamber_conflicts.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


def chamber_neighbors_for_query(
    knowledge: SymbolKnowledgeIndex,
    words: Sequence[str],
    *,
    include_master: bool = True,
    max_neighbors: int = 12,
) -> dict[int, list[CrossLink]]:
    """
    Query router: vote chambers from query, return neighbors per active chamber.

    Falls back to master when no subject vote.
    """
    voted = vote_query_chambers(words)
    if not voted:
        chambers = {MASTER_CHAMBER} if include_master else set()
    else:
        chambers = set(voted)
        if include_master:
            chambers.add(MASTER_CHAMBER)

    routed: dict[int, list[CrossLink]] = {}
    for w in words:
        wl = w.lower()
        if len(wl) < 2:
            continue
        for k in chambers:
            nbrs = knowledge.neighbors(wl, chamber=k)[:max_neighbors]
            if nbrs:
                routed.setdefault(k, []).extend(nbrs)
    return routed
