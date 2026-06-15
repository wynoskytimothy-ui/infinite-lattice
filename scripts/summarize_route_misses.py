#!/usr/bin/env python3
"""Summarize route_miss_audit_full_test.json."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
data = json.loads((_ROOT / "logs/route_miss_audit_full_test.json").read_text(encoding="utf-8"))

by_kind: dict[str, list] = defaultdict(list)
for r in data["reports"]:
    kind = r["gold_reports"][0]["failure_kind"]
    gr = r["gold_reports"][0]
    by_kind[kind].append({
        "id": r["query_id"],
        "words": r["n_words"],
        "keys": r["n_query_keys"],
        "overlap_n": len(gr["token_overlap"]),
        "k_overlap": gr["kappa_key_overlap"],
        "query": r["query"][:100],
    })

print(f"Route misses: {data['n_route_misses']} / {data['n_evaluated']} ({100*(1-data['route_recall']):.1f}%)")
print(f"Avg miss: {data['avg_miss_n_words']:.1f} words, {data['avg_miss_query_len']:.0f} chars, {data['avg_miss_n_keys']:.0f} keys\n")

for kind, rows in sorted(by_kind.items(), key=lambda x: -len(x[1])):
    wc = Counter(x["words"] for x in rows)
    ko = [x["k_overlap"] for x in rows]
    print(f"=== {kind}: {len(rows)} ===")
    print(f"  word counts: {dict(sorted(wc.items()))}")
    if ko:
        print(f"  kappa key overlap: min={min(ko)} max={max(ko)} avg={sum(ko)/len(ko):.1f}")
    for x in rows[:5]:
        print(f"  Q{x['id']}: overlap={x['overlap_n']} k_ov={x['k_overlap']} | {x['query']}")
    if len(rows) > 5:
        print(f"  ... +{len(rows)-5} more")
    print()
