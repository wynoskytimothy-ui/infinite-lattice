"""
Stage 05 — Free token address = canonical P×Q (computed, never stored).

Order policy (locked here — Stages 06–08 inherit this):
  - meet_composite = min(P,Q) × max(P,Q)  — order-free FTA meet identity
  - invoke_order   = (P, Q) as passed       — symbol meet order (separate field)
  - quadrant       = 1..32 rotation index   — from Stage 03 when opening a corridor

P×Q = Q×P as integers; tas ≠ sat is NOT encoded in the product. Order re-enters
via invoke_order + quadrant, not by multiplying extra factors into the composite.

No registry row, no cache table — addresses are regenerated from primes + rotation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from lattice_retriever_v1.stage02_intersections import (
    DEFAULT_TRANSGRESSOR_N,
    lattice_signature,
    pair_composite,
)

CorridorKey = tuple[int, int, int]  # (meet_composite, quadrant, transgressor_n)
OrientedCorridorKey = tuple[int, int, int, int, int]  # meet, quadrant, n, from_p, to_p


def canonical_pair(p: int, q: int) -> tuple[int, int]:
    """Order-free factor pair with p <= q."""
    return (p, q) if p <= q else (q, p)


def meet_composite(p: int, q: int) -> int:
    """FTA meet identity — min(P,Q) × max(P,Q)."""
    return pair_composite(p, q)


def oriented_corridor_key(addr: FreeTokenAddress) -> OrientedCorridorKey:
    """Full oriented posting identity — order re-enters outside the FTA product."""
    from_p, to_p = addr.invoke_order
    return (addr.meet_composite, addr.quadrant, addr.transgressor_n, from_p, to_p)


def oriented_corridor_pin(addr: FreeTokenAddress) -> int:
    """
    Stable int pin for inverted-index postings on oriented 2-way corridors.

    apple→phone and phone→apple share meet_composite but differ in quadrant
    and invoke_order — separate buckets, separate doc routing.
    """
    h = 1469598103934665603
    for x in oriented_corridor_key(addr):
        h ^= int(x)
        h = (h * 1099511628211) & ((1 << 64) - 1)
    return h or 1


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def factor_pair_composite(composite: int) -> tuple[int, int]:
    """
    Exact factor-back: recover {P, Q} from a two-prime product.

    Raises ValueError if composite is not a semiprime (exactly two prime factors).
    """
    if composite < 4:
        raise ValueError(f"composite too small for pair factorization: {composite}")
    root = int(math.isqrt(composite))
    for d in range(2, root + 1):
        if composite % d != 0:
            continue
        q = composite // d
        if d * q != composite:
            continue
        if is_prime(d) and is_prime(q):
            return canonical_pair(d, q)
    raise ValueError(f"composite is not a two-prime product: {composite}")


@dataclass(frozen=True)
class FreeTokenAddress:
    """
    Corridor address: canonical meet product + rotation index + rail.

    lattice_signature is derived from the canonical pair (swap-invariant meet).
    corridor_key includes quadrant so tas/sat-style order lives outside the product.
    """

    p: int
    q: int
    invoke_order: tuple[int, int]
    quadrant: int
    transgressor_n: int

    @property
    def meet_composite(self) -> int:
        return self.p * self.q

    @property
    def lattice_signature(self) -> tuple[tuple[int, int, int], ...]:
        return lattice_signature((self.p, self.q), n=self.transgressor_n)

    @property
    def corridor_key(self) -> CorridorKey:
        return (self.meet_composite, self.quadrant, self.transgressor_n)

    def explain(self) -> dict:
        return {
            "meet_composite": self.meet_composite,
            "canonical_pair": [self.p, self.q],
            "invoke_order": list(self.invoke_order),
            "quadrant": self.quadrant,
            "transgressor_n": self.transgressor_n,
            "corridor_key": list(self.corridor_key),
            "order_in_product": False,
            "order_in_quadrant_and_invoke_order": True,
            "stored_row": False,
            "lattice_L01": self.lattice_signature[0],
            "n_lattices": len(self.lattice_signature),
        }


def free_token_address(
    p: int,
    q: int,
    *,
    quadrant: int = 1,
    transgressor_n: int = DEFAULT_TRANSGRESSOR_N,
    invoke_order: tuple[int, int] | None = None,
) -> FreeTokenAddress:
    """Build corridor address from two primes — nothing persisted."""
    low, high = canonical_pair(p, q)
    order = invoke_order if invoke_order is not None else (p, q)
    qid = max(1, min(32, int(quadrant)))
    return FreeTokenAddress(
        p=low,
        q=high,
        invoke_order=order,
        quadrant=qid,
        transgressor_n=transgressor_n,
    )


def regenerate_from_composite(
    composite: int,
    *,
    quadrant: int,
    transgressor_n: int = DEFAULT_TRANSGRESSOR_N,
) -> FreeTokenAddress:
    """
    Corridor-regeneration: factor composite → primes, rebuild address.
    No registry lookup — FTA + formula only.
    """
    p, q = factor_pair_composite(composite)
    return free_token_address(
        p,
        q,
        quadrant=quadrant,
        transgressor_n=transgressor_n,
        invoke_order=(p, q),
    )


def addresses_bit_identical(a: FreeTokenAddress, b: FreeTokenAddress) -> bool:
    """Full structural equality for corridor-regeneration gates."""
    return (
        a.p == b.p
        and a.q == b.q
        and a.meet_composite == b.meet_composite
        and a.quadrant == b.quadrant
        and a.transgressor_n == b.transgressor_n
        and a.lattice_signature == b.lattice_signature
        and a.corridor_key == b.corridor_key
    )


_CASE_LABELS: dict[int, str] = {
    1: "before first anchor",
    2: "crossed first anchor",
    3: "crossed second anchor",
}


def _prime_label(p: int) -> str:
    try:
        from aethos_words import prime_to_letter

        return prime_to_letter(p)
    except ValueError:
        return f"p{p}"


def resolve_prime_label(
    p: int,
    registry: object | None = None,
    *,
    text_hint: str | None = None,
) -> dict:
    """
    Decode one prime → human text without a vocab row.

    L1 letter-primes use the fixed alphabet; pool-primes use the append-only registry.
    Optional text_hint supplies the query/doc word when identity is a letter-product.
    """
    if text_hint is not None:
        return {
            "text": text_hint,
            "tier": "L3_WORD",
            "prime": p,
            "stored_row": False,
            "hint": "read-order label",
        }
    from aethos_words import LETTER_PRIMES, prime_to_letter

    if p in LETTER_PRIMES:
        return {
            "text": prime_to_letter(p),
            "tier": "L1_SYMBOL",
            "prime": p,
            "stored_row": False,
        }
    if registry is not None:
        promoted = getattr(registry, "promoted", None)
        if promoted is not None:
            for (tier, text), tok in promoted.items():
                if tok.prime == p:
                    return {
                        "text": text,
                        "tier": tier.name if hasattr(tier, "name") else str(tier),
                        "prime": p,
                        "intersection_only": tok.intersection_only,
                        "stored_row": not tok.intersection_only,
                    }
    return {
        "text": None,
        "tier": "unknown",
        "prime": p,
        "stored_row": False,
        "hint": f"identity prime {p} — pass text_hint or registry",
    }


def decode_corridor_address(
    addr: FreeTokenAddress,
    *,
    registry: object | None = None,
    from_text: str | None = None,
    to_text: str | None = None,
) -> dict:
    """Human-readable decode of a corridor address — labels + witness."""
    from_p, to_p = addr.invoke_order
    from_dec = resolve_prime_label(from_p, registry, text_hint=from_text)
    to_dec = resolve_prime_label(to_p, registry, text_hint=to_text)
    pair: tuple[str, str] | None = None
    if from_dec["text"] and to_dec["text"]:
        pair = (from_dec["text"], to_dec["text"])
    witness = corridor_witness_explain(addr, pair=pair)
    path = (
        f"{from_dec['text']} → {to_dec['text']}"
        if pair
        else witness["summary"]
    )
    return {
        "meet_composite": addr.meet_composite,
        "anchors": list(addr.invoke_order),
        "canonical_pair": [addr.p, addr.q],
        "from": from_dec,
        "to": to_dec,
        "path": path,
        "stored_row": False,
        "witness": witness,
    }


def decode_corridor(
    composite: int,
    *,
    invoke_order: tuple[int, int] | None = None,
    quadrant: int = 1,
    transgressor_n: int = DEFAULT_TRANSGRESSOR_N,
    registry: object | None = None,
) -> dict:
    """
    Glass-box decode: composite + orientation → human path + full witness.

    No vocab lookup for letter-prime corridors; pool-primes resolve via registry only.
    """
    p, q = factor_pair_composite(composite)
    order = invoke_order if invoke_order is not None else (p, q)
    addr = free_token_address(
        p,
        q,
        quadrant=quadrant,
        transgressor_n=transgressor_n,
        invoke_order=order,
    )
    return decode_corridor_address(addr, registry=registry)


def corridor_witness_explain(
    addr: FreeTokenAddress,
    *,
    pair: tuple[str, str] | None = None,
) -> dict:
    """
    Glass-box corridor path: invoke order, transgressor regime, wing readout.

    Explains which vector opened the corridor — e.g. TH vs HE via from/to primes.
    """
    from aethos_lattice import prime_pair_case

    from lattice_retriever_v1.stage03_rotation import wing_and_branch_from_quadrant

    lo, hi = addr.p, addr.q
    n = addr.transgressor_n
    case = prime_pair_case(lo, hi, n)
    from_p, to_p = addr.invoke_order
    wing, branch = wing_and_branch_from_quadrant(addr.quadrant)
    if pair is not None:
        label = "".join(pair)
    else:
        label = f"{_prime_label(from_p)}{_prime_label(to_p)}"
    summary = (
        f"{label.upper()} corridor, case {case} ({_CASE_LABELS[case]}), "
        f"transgressed from {_prime_label(from_p)}={from_p} to {_prime_label(to_p)}={to_p} "
        f"at n={n}, wing {wing} ({branch.name})"
    )
    return {
        "label": label,
        "pair": list(pair) if pair is not None else [_prime_label(from_p), _prime_label(to_p)],
        "invoke_order": list(addr.invoke_order),
        "invoke_order_letters": [_prime_label(from_p), _prime_label(to_p)],
        "from_prime": from_p,
        "from_letter": _prime_label(from_p),
        "to_prime": to_p,
        "to_letter": _prime_label(to_p),
        "canonical_pair": [lo, hi],
        "meet_composite": addr.meet_composite,
        "transgressor_n": n,
        "case": case,
        "case_label": _CASE_LABELS[case],
        "quadrant": addr.quadrant,
        "wing": wing,
        "branch": branch.name,
        "corridor_key": list(addr.corridor_key),
        "oriented_pin": oriented_corridor_pin(addr),
        "oriented_key": list(oriented_corridor_key(addr)),
        "lattice_L01": addr.lattice_signature[0],
        "summary": summary,
    }
