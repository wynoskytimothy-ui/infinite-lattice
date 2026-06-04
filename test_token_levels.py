"""Per-level token tests — strengthen and verify L1 through LATTICE."""

from __future__ import annotations

import unittest

from aethos_core import formula_coord
from aethos_lattice_token import (
    encode_corpus,
    encode_document,
    tokenize_words,
    verify_l1_consistency,
)
from aethos_pipeline import AethosPipeline, check_promotion_invariants
from aethos_promotion import LatticeTier, PromotionRegistry, intersection_prime, letter_chain
from aethos_token_levels import TokenLevel, run_level_audits
from aethos_species import TokenSpecies
from aethos_tokenize import clean_word_token, normalize_unicode, tokenize_with_raw, tokenize_words
from aethos_words import letter_to_prime, decode_word, encode_word_at_site
from diagnose_corpus import SMALL_CORPUS


def _pipe() -> AethosPipeline:
    pipe = AethosPipeline(rebuild_every=2)
    pipe.ingest(*SMALL_CORPUS)
    return pipe


class TestL1Symbol(unittest.TestCase):
    def test_letter_primes_deterministic(self) -> None:
        self.assertEqual(letter_to_prime("a"), letter_to_prime("A"))
        self.assertNotEqual(letter_to_prime("a"), letter_to_prime("b"))

    def test_letter_chain_sorted(self) -> None:
        chain = letter_chain("apple")
        self.assertEqual(chain, tuple(sorted(chain)))

    def test_intersection_is_sum_not_product(self) -> None:
        w = "cat"
        self.assertEqual(intersection_prime(w), sum(letter_chain(w)))

    def test_l1_verify_all_corpus_words(self) -> None:
        for doc in SMALL_CORPUS:
            for w in tokenize_words(doc):
                ok, msg = verify_l1_consistency(w)
                self.assertTrue(ok, msg)


class TestL2Subword(unittest.TestCase):
    def setUp(self) -> None:
        self.pipe = _pipe()
        self.reg = self.pipe.registry

    def test_promoted_subwords_have_parent_diversity(self) -> None:
        for (tier, text), tok in self.reg.promoted.items():
            if tier != LatticeTier.L2_SUBWORD:
                continue
            parents = self.reg.subword_parent_words.get(text, set())
            count = self.reg.subword_counts[text]
            diverse = len(parents) >= self.reg.subword_min_parents
            repeated_whole = text in parents and self.reg.word_counts.get(text, 0) >= self.reg.subword_promote_at
            pmi = self.reg.max_subword_pmi(text)
            z = self.reg.max_subword_z(text)
            pmi_ok = pmi >= self.reg.subword_min_pmi or pmi >= self.reg.subword_min_pmi * 1.5
            z_ok = z >= self.reg.subword_min_z or z >= self.reg.subword_min_z * 1.5
            self.assertTrue(repeated_whole or (diverse and (pmi_ok or z_ok)), f"{text} pmi={pmi:.2f} z={z:.2f}")
            self.assertGreaterEqual(count, self.reg.subword_promote_at)

    def test_phon_not_promoted_from_single_parent(self) -> None:
        self.assertNotIn((LatticeTier.L2_SUBWORD, "phon"), self.reg.promoted)

    def test_stopword_subwords_not_promoted(self) -> None:
        for sw in ("the", "and", "with"):
            self.assertNotIn((LatticeTier.L2_SUBWORD, sw), self.reg.promoted)

    def test_shared_subword_at_from_anagram_line(self) -> None:
        self.assertIn("at", self.reg.subword_counts)
        if (LatticeTier.L2_SUBWORD, "at") in self.reg.promoted:
            self.assertGreaterEqual(len(self.reg.subword_parent_words["at"]), 2)


class TestTokenizePolicy(unittest.TestCase):
    def test_apostrophe_contraction(self) -> None:
        self.assertEqual(tokenize_words("don't stop"), ["dont", "stop"])

    def test_nfkc_and_us_abbrev(self) -> None:
        self.assertEqual(tokenize_words("U.S. trade"), ["us", "trade"])

    def test_hyphen_join(self) -> None:
        self.assertEqual(tokenize_words("co-operate now"), ["cooperate", "now"])

    def test_unicode_nfkc_latin(self) -> None:
        raw = normalize_unicode("caf\u00e9")
        self.assertEqual(clean_word_token(raw), "café")
        self.assertEqual(letter_to_prime("é"), letter_to_prime("e"))

    def test_cyrillic_letter_gets_prime(self) -> None:
        word = clean_word_token("\u043f\u0440\u0438\u0432\u0435\u0442")
        self.assertTrue(word.isalpha())
        primes = tuple(letter_to_prime(c) for c in word)
        self.assertEqual(len(primes), len(word))
        self.assertEqual(len(set(primes)), len(primes))

    def test_tokenize_with_raw(self) -> None:
        pairs = tokenize_with_raw("can't wait")
        self.assertEqual(pairs, [("can't", "cant", TokenSpecies.WORD), ("wait", "wait", TokenSpecies.WORD)])

    def test_numeric_token(self) -> None:
        from aethos_tokenize import tokenize_spans

        spans = tokenize_spans("price $42 and 2024")
        texts = [(s.text, s.species.value) for s in spans]
        self.assertIn(("42", "NUM"), texts)
        self.assertIn(("2024", "NUM"), texts)


class TestPoolAndNum(unittest.TestCase):
    def test_pool_tiers_under_critical(self) -> None:
        pipe = _pipe()
        for u in pipe.registry._pool.all_usage():
            self.assertLess(u.ratio, 0.95, u.summary())

    def test_num_species_ingest(self) -> None:
        from aethos_species import digit_chain, number_intersection

        pipe = AethosPipeline(rebuild_every=2)
        pipe.ingest("sold 42 phones in 2024", "price is 99")
        self.assertIn("42", pipe.registry.number_counts)
        tok = pipe.registry.resolve_token("42")
        self.assertEqual(tok.parent_primes, digit_chain("42"))
        self.assertEqual(tok.prime, number_intersection("42"))
        tokens = pipe.encode_document("order 42 chips")
        num = next(t for t in tokens if t.text == "42")
        self.assertEqual(num.species, "NUM")
        self.assertEqual(num.tier, "num_intersection")

    def test_compression_strength_positive_for_phone(self) -> None:
        pipe = _pipe()
        self.assertGreater(pipe.registry.compression_strength("phone"), 0)


class TestL3Word(unittest.TestCase):
    def setUp(self) -> None:
        self.pipe = _pipe()
        self.reg = self.pipe.registry

    def test_stopwords_intersection_only(self) -> None:
        for w in ("the", "a", "and", "with"):
            if w in self.reg.word_counts:
                self.assertTrue(self.reg.is_intersection_only(w), w)

    def test_apple_has_dedicated_l3(self) -> None:
        self.assertIn((LatticeTier.L3_WORD, "apple"), self.reg.promoted)

    def test_promotion_invariants(self) -> None:
        self.assertEqual(check_promotion_invariants(self.reg), [])


class TestL4Correlations(unittest.TestCase):
    def setUp(self) -> None:
        self.pipe = _pipe()
        self.reg = self.pipe.registry

    def test_edges_exist(self) -> None:
        self.assertGreater(len(self.reg.correlations), 0)

    def test_phone_has_neighbors(self) -> None:
        corrs = self.reg.correlations_for("phone")
        self.assertGreater(len(corrs), 0)
        self.assertGreaterEqual(corrs[0].strength, 1)


class TestL7L9Resolve(unittest.TestCase):
    def setUp(self) -> None:
        self.pipe = _pipe()

    def test_apple_disambiguation(self) -> None:
        tech = self.pipe.resolve("apple", ["phone", "chip"])
        food = self.pipe.resolve("apple", ["fruit", "pie"])
        self.assertNotEqual(tech["cluster_id"], food["cluster_id"])

    def test_oov(self) -> None:
        r = self.pipe.resolve("zebra")
        self.assertEqual(r["cluster_id"], "")
        self.assertEqual(r["cluster_score"], 0.0)


class TestLatticeTokenEmitter(unittest.TestCase):
    def setUp(self) -> None:
        self.pipe = _pipe()

    def test_encode_document_length(self) -> None:
        doc = SMALL_CORPUS[0]
        tokens = self.pipe.encode_document(doc)
        self.assertEqual(len(tokens), len(tokenize_words(doc)))

    def test_token_fields(self) -> None:
        tokens = self.pipe.encode_document(SMALL_CORPUS[2])
        apple = next(t for t in tokens if t.text == "apple")
        self.assertEqual(apple.tier, "dedicated_l3")
        self.assertFalse(apple.intersection_only)
        self.assertEqual(len(apple.lattice_local), 3)
        self.assertNotEqual(apple.cluster_id, "")

    def test_lattice_matches_formula_coord(self) -> None:
        reg = self.pipe.registry
        w = "phone"
        tok = reg.resolve_token(w)
        chain = tok.parent_primes if tok.intersection_only else tuple(sorted(set(tok.parent_primes + (tok.prime,))))
        reg_addr = reg.lattice_address(w, LatticeTier.L3_WORD)
        core_addr = formula_coord(chain, 7)
        for i in range(3):
            self.assertAlmostEqual(reg_addr[i], core_addr[i])

    def test_encode_corpus_flat(self) -> None:
        tokens = encode_corpus(SMALL_CORPUS, self.pipe.registry, infer_cluster=self.pipe.reader.infer_cluster)
        self.assertGreater(len(tokens), 20)

    def test_word_dot_roundtrip(self) -> None:
        self.assertEqual(decode_word(encode_word_at_site("apple")), "apple")


class TestLevelAudits(unittest.TestCase):
    def test_all_levels_pass_on_small_corpus(self) -> None:
        pipe = _pipe()
        results = run_level_audits(pipe, SMALL_CORPUS)
        failed = [r for r in results if not r.passed]
        self.assertFalse(failed, "\n".join(r.summary() for r in failed))

    def test_single_level_l1(self) -> None:
        pipe = _pipe()
        results = run_level_audits(pipe, SMALL_CORPUS, levels=[TokenLevel.L1_SYMBOL])
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].passed)


if __name__ == "__main__":
    unittest.main()
