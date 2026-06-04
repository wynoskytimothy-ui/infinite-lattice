"""AETHOS core foundation — primes, L1 characters, L2 subwords."""

from core.l1_characters import (
    char_prime,
    intersection_prime,
    prime_to_char,
    word_letter_chain,
    word_letter_order,
)
from core.l2_subwords import (
    SubwordConfig,
    SubwordPromoter,
    SubwordStats,
    decompose,
    shared_l2_factors,
)
from core.phi_lattice import (
    compute_anchor,
    compute_coordinates,
    prime_factor_similarity,
    swap_meet,
    should_promote_intersection,
)
from core.primes import (
    LETTER_PRIMES,
    PROMOTION_POOL,
    PrimePool,
    PoolTier,
    chain_primes,
    product_unique,
)

__all__ = [
    "LETTER_PRIMES",
    "PROMOTION_POOL",
    "PrimePool",
    "PoolTier",
    "chain_primes",
    "product_unique",
    "char_prime",
    "prime_to_char",
    "word_letter_order",
    "word_letter_chain",
    "intersection_prime",
    "SubwordConfig",
    "SubwordStats",
    "SubwordPromoter",
    "decompose",
    "shared_l2_factors",
    "compute_coordinates",
    "compute_anchor",
    "prime_factor_similarity",
    "swap_meet",
    "should_promote_intersection",
]
