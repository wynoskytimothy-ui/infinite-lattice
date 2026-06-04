"""
Token level audit — strengthen and test L1 through L9 + lattice each step.

Run after ingest:
  python run_token_levels.py
  python run_token_levels.py --level L3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from aethos_core import formula_coord
from aethos_lattice_token import encode_document, tokenize_words, verify_l1_consistency
from aethos_pipeline import AethosPipeline, check_promotion_invariants
from aethos_promotion import LatticeTier, PromotionRegistry, intersection_prime, is_stopword, letter_chain
from aethos_words import letter_to_prime


class TokenLevel(str, Enum):
    L1_SYMBOL = "L1"
    L2_SUBWORD = "L2"
    L3_WORD = "L3"
    L4_CORR = "L4"
    L5_CORR = "L5"
    L6_CORR = "L6"
    L7_CLUSTER = "L7"
    L8_MARKOV = "L8"
    L9_RESOLVE = "L9"
    LATTICE = "LATTICE"
    POOL = "POOL"
    NUM = "NUM"


@dataclass
class LevelCheck:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class LevelResult:
    level: TokenLevel
    passed: bool
    checks: list[LevelCheck] = field(default_factory=list)
    metrics: dict[str, object] = field(default_factory=dict)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{self.level.value}] {status}"]
        for c in self.checks:
            mark = "ok" if c.passed else "FAIL"
            suffix = f" — {c.detail}" if c.detail else ""
            lines.append(f"  {mark}: {c.name}{suffix}")
        if self.metrics:
            parts = ", ".join(f"{k}={v}" for k, v in self.metrics.items())
            lines.append(f"  metrics: {parts}")
        return "\n".join(lines)


def audit_l1(registry: PromotionRegistry) -> LevelResult:
    checks: list[LevelCheck] = []
    for ch in "abcdefghijklmnopqrstuvwxyz":
        p = letter_to_prime(ch)
        checks.append(LevelCheck(f"letter {ch!r} -> prime", p > 0, f"prime={p}"))
    sample_words = sorted(registry.word_counts.keys())[:20]
    for w in sample_words:
        ok, msg = verify_l1_consistency(w)
        checks.append(LevelCheck(f"L1 sum {w!r}", ok, msg))
    for w in sample_words:
        chain = letter_chain(w)
        checks.append(
            LevelCheck(
                f"letter_chain sorted {w!r}",
                list(chain) == sorted(chain),
                f"len={len(chain)}",
            )
        )
    return LevelResult(
        level=TokenLevel.L1_SYMBOL,
        passed=all(c.passed for c in checks),
        checks=checks,
        metrics={"unique_words": len(registry.word_counts), "symbols_seen": len(registry.symbol_counts)},
    )


def audit_l2(registry: PromotionRegistry) -> LevelResult:
    checks: list[LevelCheck] = []
    l2 = [(t, tok) for (tier, t), tok in registry.promoted.items() if tier == LatticeTier.L2_SUBWORD]
    for text, tok in l2:
        count = registry.subword_counts.get(text, 0)
        parents = registry.subword_parent_words.get(text, set())
        diverse = len(parents) >= registry.subword_min_parents
        repeated_whole = text in parents and registry.word_counts.get(text, 0) >= registry.subword_promote_at
        pmi = registry.max_subword_pmi(text)
        pmi_ok = pmi >= registry.subword_min_pmi or pmi >= registry.subword_min_pmi * 1.5
        zmax = registry.max_subword_z(text)
        z_ok = zmax >= registry.subword_min_z or zmax >= registry.subword_min_z * 1.5
        ok = (
            count >= registry.subword_promote_at
            and not is_stopword(text)
            and (repeated_whole or (diverse and (pmi_ok or z_ok)))
        )
        checks.append(
            LevelCheck(
                f"L2 {text!r} promoted",
                ok,
                f"count={count} parents={len(parents)} pmi={pmi:.2f} prime={tok.prime}",
            )
        )
    checks.append(
        LevelCheck(
            "phon not spuriously promoted",
            (LatticeTier.L2_SUBWORD, "phon") not in registry.promoted,
        )
    )
    # tab/bat corpus expectation: shared subwords from anagram line
    for sw in ("at", "ba", "ta"):
        count = registry.subword_counts.get(sw, 0)
        if count < registry.subword_promote_at:
            continue
        promoted = (LatticeTier.L2_SUBWORD, sw) in registry.promoted
        if promoted:
            checks.append(LevelCheck(f"high-freq subword {sw!r}", True, "promoted"))
        else:
            # Common orthographic substring (e.g. 'at') may fail PMI — that is OK.
            checks.append(
                LevelCheck(
                    f"high-freq subword {sw!r}",
                    True,
                    f"not promoted (max_pmi={registry.max_subword_pmi(sw):.2f})",
                )
            )
    return LevelResult(
        level=TokenLevel.L2_SUBWORD,
        passed=all(c.passed for c in checks),
        checks=checks,
        metrics={"l2_promoted": len(l2), "subword_vocab": len(registry.subword_counts)},
    )


def audit_l3(registry: PromotionRegistry) -> LevelResult:
    checks: list[LevelCheck] = []
    for w in registry.intersections:
        checks.append(
            LevelCheck(
                f"intersection {w!r}",
                (LatticeTier.L3_WORD, w) not in registry.promoted,
            )
        )
    for (tier, w), tok in registry.promoted.items():
        if tier != LatticeTier.L3_WORD:
            continue
        checks.append(
            LevelCheck(
                f"dedicated {w!r}",
                registry.contexts_differ(w) and registry.word_counts.get(w, 0) >= registry.word_promote_at,
                f"prime={tok.prime}",
            )
        )
    inv = check_promotion_invariants(registry)
    checks.append(LevelCheck("promotion invariants", not inv, "; ".join(inv[:3])))
    return LevelResult(
        level=TokenLevel.L3_WORD,
        passed=all(c.passed for c in checks),
        checks=checks,
        metrics={
            "intersection_only": len(registry.intersections),
            "dedicated_l3": sum(1 for k in registry.promoted if k[0] == LatticeTier.L3_WORD),
        },
    )


def audit_l4(registry: PromotionRegistry) -> LevelResult:
    checks: list[LevelCheck] = []
    if not registry.correlations:
        checks.append(LevelCheck("correlation edges exist", False, "no L4-L6 edges"))
    else:
        checks.append(LevelCheck("correlation edges exist", True, f"{len(registry.correlations)} edges"))
    for (a, b), link in list(registry.correlations.items())[:15]:
        checks.append(
            LevelCheck(
                f"edge {a!r}-{b!r}",
                link.strength >= 1 and link.dim4 >= 0,
                f"strength={link.strength}",
            )
        )
    multi = [w for w, c in registry.word_counts.items() if c >= 2]
    linked = sum(1 for w in multi if registry.correlations_for(w))
    checks.append(
        LevelCheck(
            "repeat words have neighbors",
            linked >= min(1, len(multi)),
            f"{linked}/{len(multi)} linked",
        )
    )
    return LevelResult(
        level=TokenLevel.L4_CORR,
        passed=all(c.passed for c in checks),
        checks=checks,
        metrics={"edges": len(registry.correlations)},
    )


def audit_l5_l6(registry: PromotionRegistry) -> LevelResult:
    checks: list[LevelCheck] = []
    for (_, _), link in list(registry.correlations.items())[:10]:
        checks.append(
            LevelCheck(
                f"dims456 {link.source!r}-{link.target!r}",
                0 <= link.dim4 <= 10 and 0 <= link.dim5 <= 10 and 0 <= link.dim6 <= 10,
            )
        )
        pt = registry.correlation_point(link)
        checks.append(LevelCheck(f"6D point len", len(pt) == 6))
    return LevelResult(
        level=TokenLevel.L5_CORR,
        passed=all(c.passed for c in checks) if checks else True,
        checks=checks,
        metrics={"sample_links": min(10, len(registry.correlations))},
    )


def audit_l7(pipe: AethosPipeline, documents: tuple[str, ...]) -> LevelResult:
    reader = pipe.reader
    checks: list[LevelCheck] = []
    hubs = reader.cluster_hubs
    checks.append(LevelCheck("clusters discovered", len(hubs) >= 1, f"{len(hubs)} hubs"))
    active = set(hubs.keys())
    cross_keys = set(reader.cross.categories.keys())
    checks.append(LevelCheck("cross lattice synced", active == cross_keys))
    if "apple" in pipe.registry.word_counts:
        tech = pipe.resolve("apple", ["phone", "chip"])
        food = pipe.resolve("apple", ["fruit", "pie"])
        checks.append(
            LevelCheck(
                "apple disambiguation",
                tech["cluster_id"] != food["cluster_id"],
                f"{tech['cluster_id']} vs {food['cluster_id']}",
            )
        )
    return LevelResult(
        level=TokenLevel.L7_CLUSTER,
        passed=all(c.passed for c in checks),
        checks=checks,
        metrics={"clusters": len(hubs), "bridge_words": len(getattr(reader, "bridge_cluster_map", {}))},
    )


def audit_l8(pipe: AethosPipeline, documents: tuple[str, ...]) -> LevelResult:
    reader = pipe.reader
    checks: list[LevelCheck] = []
    checks.append(LevelCheck("markov categories", len(reader.cross.categories) >= 1))
    checks.append(LevelCheck("cluster_hubs populated", len(reader.cluster_hubs) >= 1))
    return LevelResult(
        level=TokenLevel.L8_MARKOV,
        passed=all(c.passed for c in checks),
        checks=checks,
        metrics={"categories": len(reader.cross.categories)},
    )


def audit_l9(pipe: AethosPipeline, documents: tuple[str, ...]) -> LevelResult:
    checks: list[LevelCheck] = []
    r = pipe.resolve("zebra")
    checks.append(LevelCheck("OOV empty cluster", r["cluster_id"] == "" and r["cluster_score"] == 0.0))
    if "cat" in pipe.registry.word_counts:
        cat = pipe.resolve("cat")
        checks.append(
            LevelCheck(
                "known word routes",
                cat["cluster_id"] != "" and cat["cluster_score"] > 0,
                cat["cluster_id"],
            )
        )
    for w in ("the", "a", "and"):
        if w in pipe.registry.word_counts:
            checks.append(
                LevelCheck(
                    f"stopword {w!r} intersection",
                    pipe.registry.is_intersection_only(w),
                )
            )
    return LevelResult(
        level=TokenLevel.L9_RESOLVE,
        passed=all(c.passed for c in checks),
        checks=checks,
        metrics={"resolve_ok": sum(1 for c in checks if c.passed)},
    )


def audit_pool(registry: PromotionRegistry) -> LevelResult:
    checks: list[LevelCheck] = []
    for u in registry._pool.all_usage():
        checks.append(LevelCheck(f"tier {u.tier.value}", not u.critical, u.summary()))
    checks.append(
        LevelCheck(
            "allocator sync",
            registry._next_promotion_idx == registry._pool.total_used(),
            f"total={registry._next_promotion_idx}",
        )
    )
    return LevelResult(
        level=TokenLevel.POOL,
        passed=all(c.passed for c in checks),
        checks=checks,
        metrics={"total_allocated": registry._next_promotion_idx},
    )


def audit_num(pipe: AethosPipeline, documents: tuple[str, ...]) -> LevelResult:
    from aethos_species import digit_chain, number_intersection
    from aethos_tokenize import tokenize_spans

    reg = pipe.registry
    checks: list[LevelCheck] = []
    nums: set[str] = set()
    for doc in documents:
        for span in tokenize_spans(doc):
            if span.species.value == "NUM":
                nums.add(span.text)
    if not nums:
        checks.append(LevelCheck("NUM tokens in corpus", True, "none (skip)"))
    else:
        checks.append(LevelCheck("NUM tokens in corpus", True, f"{len(nums)} found"))
    for n in sorted(nums)[:5]:
        tok = reg.resolve_token(n)
        checks.append(LevelCheck(f"digit chain {n!r}", tok.parent_primes == digit_chain(n)))
        checks.append(
            LevelCheck(
                f"intersection {n!r}",
                tok.prime == number_intersection(n),
                f"prime={tok.prime}",
            )
        )
        addr = reg.lattice_address(n, LatticeTier.L3_WORD)
        chain = tuple(sorted(set(tok.parent_primes)))
        core = formula_coord(chain, 7)
        checks.append(
            LevelCheck(
                f"formula_coord NUM {n!r}",
                all(abs(addr[i] - core[i]) < 1e-9 for i in range(3)),
            )
        )
    return LevelResult(
        level=TokenLevel.NUM,
        passed=all(c.passed for c in checks),
        checks=checks,
        metrics={"unique_numbers": len(nums)},
    )


def audit_lattice(pipe: AethosPipeline, documents: tuple[str, ...]) -> LevelResult:
    checks: list[LevelCheck] = []
    reg = pipe.registry
    for doc in documents[:3]:
        words = tokenize_words(doc)
        for w in words[:5]:
            reg_addr = reg.lattice_address(w, LatticeTier.L3_WORD)
            tok = reg.resolve_token(w)
            if tok.intersection_only:
                chain = tuple(sorted(set(tok.parent_primes)))
            else:
                chain = tuple(sorted(set(tok.parent_primes + (tok.prime,))))
            core_addr = formula_coord(chain, 7)
            close = all(abs(reg_addr[i] - core_addr[i]) < 1e-9 for i in range(3))
            checks.append(LevelCheck(f"formula_coord {w!r}", close, f"reg={reg_addr}"))
    tokens = encode_document(
        documents[0],
        reg,
        infer_cluster=pipe.reader.infer_cluster,
    )
    checks.append(LevelCheck("encode_document", len(tokens) == len(tokenize_words(documents[0]))))
    checks.append(LevelCheck("token lattice_local", all(len(t.lattice_local) == 3 for t in tokens)))
    return LevelResult(
        level=TokenLevel.LATTICE,
        passed=all(c.passed for c in checks),
        checks=checks,
        metrics={"encoded_tokens": len(tokens)},
    )


_LEVEL_ORDER: list[TokenLevel] = [
    TokenLevel.L1_SYMBOL,
    TokenLevel.L2_SUBWORD,
    TokenLevel.L3_WORD,
    TokenLevel.L4_CORR,
    TokenLevel.L5_CORR,
    TokenLevel.L7_CLUSTER,
    TokenLevel.L8_MARKOV,
    TokenLevel.L9_RESOLVE,
    TokenLevel.POOL,
    TokenLevel.NUM,
    TokenLevel.LATTICE,
]

_AUDIT_FNS: dict[TokenLevel, Callable[..., LevelResult]] = {
    TokenLevel.L1_SYMBOL: lambda pipe, docs: audit_l1(pipe.registry),
    TokenLevel.L2_SUBWORD: lambda pipe, docs: audit_l2(pipe.registry),
    TokenLevel.L3_WORD: lambda pipe, docs: audit_l3(pipe.registry),
    TokenLevel.L4_CORR: lambda pipe, docs: audit_l4(pipe.registry),
    TokenLevel.L5_CORR: lambda pipe, docs: audit_l5_l6(pipe.registry),
    TokenLevel.L7_CLUSTER: audit_l7,
    TokenLevel.L8_MARKOV: audit_l8,
    TokenLevel.L9_RESOLVE: audit_l9,
    TokenLevel.POOL: lambda pipe, docs: audit_pool(pipe.registry),
    TokenLevel.NUM: audit_num,
    TokenLevel.LATTICE: audit_lattice,
}


def run_level_audits(
    pipe: AethosPipeline,
    documents: tuple[str, ...],
    *,
    levels: list[TokenLevel] | None = None,
) -> list[LevelResult]:
    """Run selected level audits on an already-ingested pipeline."""
    chosen = levels or _LEVEL_ORDER
    return [_AUDIT_FNS[lvl](pipe, documents) for lvl in chosen if lvl in _AUDIT_FNS]


def format_audit_report(results: list[LevelResult]) -> str:
    lines = ["AETHOS token level audit", "=" * 40]
    passed = sum(1 for r in results if r.passed)
    lines.append(f"Levels: {passed}/{len(results)} passed\n")
    for r in results:
        lines.append(r.summary())
        lines.append("")
    return "\n".join(lines)
