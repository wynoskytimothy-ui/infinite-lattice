"""Stage 01 gate — symbols → primes."""

from aethos_words import LETTER_PRIMES, letter_to_prime
from lattice_retriever_v1.stage01_symbols import symbols_to_primes


def test_letter_primes_start_at_three():
    """In AETHOS, 2 is reserved; letter primes begin at the first chain prime (3)."""
    assert LETTER_PRIMES[0] == 3
    assert letter_to_prime("a") == 3


def test_deterministic_mapping():
    a = symbols_to_primes("task")
    b = symbols_to_primes("task")
    assert a.primes == b.primes
    assert a.explain() == b.explain()


def test_anagrams_differ_in_sequence():
    tas = symbols_to_primes("tas")
    sat = symbols_to_primes("sat")
    assert tas.primes != sat.primes
    # Sorted intersection chain is the same multiset; order differs (stage 03 uses order)
    assert sorted(tas.primes) == sorted(sat.primes)


def test_glass_box_explain():
    ex = symbols_to_primes("at").explain()
    assert ex["text"] == "at"
    assert len(ex["symbols"]) == 2
    assert ex["sequence"][0] == letter_to_prime("a")
