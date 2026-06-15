#!/usr/bin/env python3
"""Print human summary of logs/query_audit_10h10m.json."""
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

data = json.loads(Path("logs/query_audit_10h10m.json").read_text())


def _clean(s: str) -> str:
    return re.sub(r"[^\x00-\x7f]+", "?", s)


def fmt_hit(h: dict) -> str:
    g = h["gold_ids"][0]
    lines = [
        f"Q{h['qid']}: {_clean(h['query'])}",
        f"  ndcg={h['ndcg@10']}  gold_in_route={h['gold_in_route']}  "
        f"kappa_overlap={h['gold_kappa_overlap'].get(g, 0)}",
        f"  GOLD {g}: {_clean(h['gold_snippet'][g][:110])}",
        f"  rare_triggers: {h['rare_triggers']}",
    ]
    for w in h["words"][:7]:
        corrs = w.get("correlations") or []
        corr_s = ", ".join(
            f"{c['other']}({c['strength']},{c['link_kind']})" for c in corrs[:4]
        )
        links = ", ".join(
            f"{l['other']}({l['kind']},{l['strength']})"
            for l in w.get("brain_links", [])[:3]
        )
        lines.append(
            f"  '{w['word']}' [{w.get('class')}] rare={w.get('is_rare_trigger')}  "
            f"corr_neighbors: {corr_s or '—'}  brain: {links or '—'}"
        )
    meets = ", ".join(
        f"{m['pair']} str={m.get('link_strength')} meet={m['indexed_meet']}"
        for m in h.get("meet_pairs", [])[:5]
    )
    lines.append(f"  pair_meets({h['n_pair_meets']}): {meets or '—'}")
    return "\n".join(lines)


def fmt_miss(m: dict) -> str:
    g = m["gold_ids"][0]
    t1 = m["top1_id"]
    lines = [
        f"Q{m['qid']}: {_clean(m['query'])}",
        f"  gold_in_route={m['gold_in_route']}  "
        f"kappa_overlap={m['gold_kappa_overlap'].get(g, 0)}",
        f"  GOT  {t1}: {_clean(m['top1_snippet'][:110])}",
        f"  WANT {g}: {_clean(m['gold_snippet'][g][:110])}",
        f"  top5: {m['top5']}",
        f"  rare_triggers: {m['rare_triggers']}",
    ]
    for w in m["words"][:7]:
        corrs = w.get("correlations") or []
        corr_s = ", ".join(
            f"{c['other']}({c['strength']},{c['link_kind']})" for c in corrs[:4]
        )
        links = ", ".join(
            f"{l['other']}({l['kind']},{l['strength']})"
            for l in w.get("brain_links", [])[:3]
        )
        lines.append(
            f"  '{w['word']}' [{w.get('class')}]  "
            f"corr_neighbors: {corr_s or '—'}  brain: {links or '—'}"
        )
    meets = ", ".join(
        f"{m['pair']} meet={m['indexed_meet']}" for m in m.get("meet_pairs", [])[:5]
    )
    lines.append(f"  pair_meets: {meets or '—'}")
    return "\n".join(lines)


print("=== 10 HITS (gold in top 10) ===\n")
for h in data["hits"]:
    print(fmt_hit(h))
    print()
print("=== 10 MISSES (gold not in top 10) ===\n")
for m in data["misses"]:
    print(fmt_miss(m))
    print()
print(f"pool: {data['n_hits_total']} hits / {data['n_miss_total']} misses of 294")
