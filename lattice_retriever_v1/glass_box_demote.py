"""Glass-box demotion rules for hybrid scoring — zero-shot polluter/hub penalties."""

from __future__ import annotations

from aethos_append_index import AppendOnlyLatticeIndex, words
from aethos_glass_box_search import GlassBoxSearchConfig, rarest_terms, word_idf


def scifact_polluter_docs() -> frozenset[str]:
    return GlassBoxSearchConfig.scifact_target().polluter_docs


def apply_lexical_demotion(
    scores: dict[str, float],
    query: str,
    pool: set[str],
    idx: AppendOnlyLatticeIndex,
    corpus: dict[str, str],
    *,
    polluter_docs: frozenset[str],
    polluter_penalty: float = 0.18,
    hub_lex_penalty: float = 0.0,
    density_penalty: float = 0.0,
) -> tuple[dict[str, float], int]:
    """Scale down hub-heavy / polluter docs in fused scores."""
    if not scores or polluter_penalty <= 0 and hub_lex_penalty <= 0 and density_penalty <= 0:
        return scores, 0

    N = len(idx.alive)
    rarest = rarest_terms(words(query), idx, N)
    qset = set(words(query))
    penalized = 0
    out: dict[str, float] = {}

    for d in pool:
        s = scores.get(d, 0.0)
        if polluter_penalty and d in polluter_docs:
            s *= 1.0 - polluter_penalty
            penalized += 1
        if hub_lex_penalty or density_penalty:
            toks = set(words(corpus.get(d, "")))
            hubs, rares = 0, 0
            for w in qset:
                if w not in toks:
                    continue
                i = word_idf(idx, w, N)
                if i < 2.0:
                    hubs += 1
                if i >= 3.0:
                    rares += 1
            density = len(qset & toks) / max(len(qset), 1)
            has_rarest = bool(rarest and rarest[0] in toks)
            if hub_lex_penalty and hubs > rares and not has_rarest:
                s *= 1.0 - hub_lex_penalty
                penalized += 1
            if density_penalty and density > 0.55 and rares == 0:
                s *= 1.0 - density_penalty
                penalized += 1
        out[d] = s

    return out, penalized
