"""decode_corridor — factor-back + human labels without a vocab row."""

from aethos_words import letter_to_prime

from lattice_retriever_v1.stage05_free_token import (
    decode_corridor,
    decode_corridor_address,
    free_token_address,
    meet_composite,
)
from lattice_retriever_v1.stage07_semantic_light import build_demo_registry
from lattice_retriever_v1.stage08_retrieve import LatticeRetriever


def test_decode_letter_pair_no_registry():
    t, h = letter_to_prime("t"), letter_to_prime("h")
    comp = meet_composite(t, h)
    dec = decode_corridor(comp, invoke_order=(t, h), quadrant=2)
    assert dec["stored_row"] is False
    assert dec["from"]["text"] == "t"
    assert dec["to"]["text"] == "h"
    assert dec["path"] == "t → h"
    assert dec["witness"]["quadrant"] == 2
    assert dec["witness"]["meet_composite"] == comp


def test_decode_promoted_subword_via_registry():
    reg = build_demo_registry()
    th = reg.promoted_subword("th")
    ing = reg.promoted_subword("ing")
    assert th is not None and ing is not None
    comp = meet_composite(th.prime, ing.prime)
    dec = decode_corridor(
        comp,
        invoke_order=(th.prime, ing.prime),
        quadrant=5,
        registry=reg.registry,
    )
    assert dec["from"]["text"] == "th"
    assert dec["to"]["text"] == "ing"
    assert dec["from"]["stored_row"] is True
    assert dec["path"] == "th → ing"


def test_decode_corridor_address_roundtrip():
    t, h = letter_to_prime("t"), letter_to_prime("h")
    addr = free_token_address(t, h, quadrant=11, invoke_order=(t, h))
    dec = decode_corridor_address(addr, from_text="t", to_text="h")
    assert dec["anchors"] == [t, h]
    assert dec["path"] == "t → h"
    assert "TH corridor" in dec["witness"]["summary"]


def test_retriever_witness_includes_decode():
    r = LatticeRetriever()
    r.index_doc("d1", "cat purrs softly")
    trace = r.retrieve_with_trace("cat purrs", limit=1)
    w = trace.corridor_witnesses[0]
    assert "decode" in w
    assert w["decode"]["path"] == "cat → purrs"
