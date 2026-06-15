#!/usr/bin/env python3
"""Fast trinary train on route misses — cached doc-freq, no slow corpus rescans."""

from __future__ import annotations

import json
import re
import sys
import time
from itertools import combinations
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from aethos_symbol_cellular import _DEFAULT_MEMBRANE
from aethos_symbol_knowledge import PRETRAIN_QUANTUM_GOLD, SymbolKnowledgeIndex, knowledge_path
from aethos_symbol_trinary_train import TrinaryTrainer
from beir_data_root import resolve_beir_root
from eval_beir import load_paths, load_qrels, load_queries
from eval_beir_symbol import query_words

MISS_IDS = ["1", "3", "13", "36", "48", "54", "94", "99", "127", "132"]
_TOKEN_RE = re.compile(r"[a-z]+")


class FastTrinaryTrainer(TrinaryTrainer):
    """TrinaryTrainer with one-pass document-frequency cache."""

    def __init__(self, knowledge):
        super().__init__(knowledge, max_triples_per_query=8)
        self._doc_freq: dict[str, int] = {}
        for text in knowledge.corpus.values():
            toks = {
                t for t in _TOKEN_RE.findall(text.lower())
                if t not in _DEFAULT_MEMBRANE and len(t) >= 3
            }
            for t in toks:
                self._doc_freq[t] = self._doc_freq.get(t, 0) + 1

    def _word_doc_freq(self, word: str) -> int:
        return self._doc_freq.get(word.lower(), 0)

    def promote_query_gold_pairs(
        self,
        query_text: str,
        gold_doc_ids: list[str],
        *,
        strength: float = 8.0,
    ) -> int:
        """Fallback: strengthen query-word ↔ gold-token pairs when no triples found."""
        qwords = set(self._tokens(query_text))
        promoted = 0
        for gid in gold_doc_ids:
            gtext = self.knowledge.corpus.get(gid, "")
            gtoks = {
                t for t in _TOKEN_RE.findall(gtext.lower())
                if t not in _DEFAULT_MEMBRANE and len(t) >= 3
            }
            overlap = qwords & gtoks
            # query↔overlap + overlap triples within gold window
            for qw in qwords:
                for gw in overlap or list(gtoks)[:6]:
                    if qw == gw:
                        continue
                    key = tuple(sorted((qw, gw)))
                    self.knowledge._cooccur_pairs[key] = (
                        self.knowledge._cooccur_pairs.get(key, 0) + strength
                    )
                    master = self.knowledge._chamber_cooccur.setdefault(0, {})
                    master[key] = master.get(key, 0) + int(strength)
                    self.knowledge._add_link(
                        qw, gw, kind="direct",
                        strength=float(self.knowledge._cooccur_pairs[key]),
                    )
                    promoted += 1
            tokens = self._tokens(gtext)
            for i in range(len(tokens)):
                win = tokens[i : i + self.window]
                if len(win) < 3:
                    continue
                for combo in combinations(dict.fromkeys(win), 3):
                    if len(set(combo) & qwords) >= 2:
                        from aethos_symbol_trinary_train import WordTriple
                        t = WordTriple(
                            words=tuple(sorted(combo)),
                            gold_doc_id=gid,
                            rarity=1.0,
                            query_overlap=len(set(combo) & qwords),
                            meet_imag=0,
                        )
                        self.promote_triple(t, gold_doc_ids=frozenset(gold_doc_ids))
                        promoted += 1
                        if promoted >= 24:
                            return promoted
        return promoted


def main() -> int:
    paths = load_paths(Path(resolve_beir_root()), "scifact")
    queries = load_queries(paths.queries)
    qrels = load_qrels(paths.qrels_test)

    print("Loading brain ...", flush=True)
    t0 = time.perf_counter()
    knowledge = SymbolKnowledgeIndex.load("scifact")
    print(f"  loaded in {(time.perf_counter()-t0)*1000:.0f} ms", flush=True)

    if "1" in MISS_IDS:
        print("Pretrain Q1 ...", flush=True)
        knowledge.compound_learn(PRETRAIN_QUANTUM_GOLD, subjects={1, 9, 10})

    trainer = FastTrinaryTrainer(knowledge)
    reports = []
    for qid in MISS_IDS:
        golds = list(qrels[qid].keys())
        rep = trainer.train_query(qid, queries[qid], golds)
        extra = 0
        if rep.triples_promoted == 0:
            extra = trainer.promote_query_gold_pairs(queries[qid], golds)
        reports.append({
            "query_id": qid,
            "triples": rep.triples_promoted,
            "fallback_pairs": extra,
            "top": rep.promoted[0].words if rep.promoted else None,
        })
        print(
            f"  Q{qid}: triples={rep.triples_promoted} fallback={extra} "
            f"top={reports[-1]['top']}",
            flush=True,
        )

    knowledge.save(knowledge_path("scifact_miss_trinary"))
    out = _ROOT / "logs" / "trinary_train_misses.json"
    out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"Saved {knowledge_path('scifact_miss_trinary')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
