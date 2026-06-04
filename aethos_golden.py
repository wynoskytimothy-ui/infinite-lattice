"""
Frozen anchors for regression — odd-prime policy (2 is never in PRIMES chains).

Update only when the anchor policy intentionally changes.
"""

from __future__ import annotations

# First five odd primes from SequenceKind.PRIMES
PRIMES_CHAIN_5: tuple[int, ...] = (3, 5, 7, 11, 13)

# Letter a uses first slot of LETTER_PRIMES
LETTER_A_PRIME: int = 3
LETTER_Z_PRIME: int = 103  # 26th odd prime (a=3 .. z=103)

# Codec menu and promotion pool (after 26 letter slots)
PRIME_MENU_FIRST: int = 3
PROMOTION_POOL_FIRST: int = 107  # 27th odd prime (index 26 in full chain)
