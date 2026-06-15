#!/usr/bin/env python3
"""
Train rare 3-way correlations from query + gold docs (SciFact qrels).

  python scripts/train_trinary_qrels.py --max-queries 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from aethos_symbol_knowledge import SymbolKnowledgeIndex, knowledge_path
from aethos_symbol_trinary_train import TrinaryTrainer, load_beir_qrels_train


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--knowledge", default="scifact_compound")
    p.add_argument("--max-queries", type=int, default=30)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    kpath = knowledge_path(args.knowledge)
    if not kpath.is_file():
        kpath = knowledge_path("scifact")
    print(f"Loading knowledge {kpath} ...")
    knowledge = SymbolKnowledgeIndex.load(args.knowledge, path=kpath)

    print("Loading qrels ...")
    queries, qrels = load_beir_qrels_train(args.dataset)

    trainer = TrinaryTrainer(knowledge=knowledge)
    reports = trainer.train_from_qrels(queries, qrels, max_queries=args.max_queries)

    total_promoted = sum(r.triples_promoted for r in reports)
    print(f"\nTrained {len(reports)} queries  promoted {total_promoted} triples")
    for r in reports[:5]:
        print(f"  [{r.query_id}] golds={len(r.gold_doc_ids)} rarest={r.rarest_gold_doc!r} "
              f"triples={r.triples_promoted}")
        if r.promoted:
            print(f"    top: {r.promoted[0].words}")

    knowledge.save(knowledge_path(f"{args.knowledge}_trinary"))
    out = Path(args.out or _ROOT / "logs" / "trinary_train_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "queries": len(reports),
        "triples_promoted": total_promoted,
        "reports": [
            {
                "query_id": r.query_id,
                "query": r.query_text[:80],
                "rarest_gold": r.rarest_gold_doc,
                "triples_promoted": r.triples_promoted,
                "top_triple": r.promoted[0].words if r.promoted else None,
            }
            for r in reports
        ],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved knowledge: {knowledge_path(f'{args.knowledge}_trinary')}")
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
