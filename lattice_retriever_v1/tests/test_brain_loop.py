"""Brain loop — synthetic miss → teach → re-retrieve round-trip."""

from lattice_retriever_v1.brain_loop import BrainLoop
from lattice_retriever_v1.stage08_retrieve import FIXTURE_CORPUS


def _build_miss_fixture() -> tuple[BrainLoop, str, str]:
    """Query term absent from corpus; gold holds bridge term only."""
    loop = BrainLoop()
    loop.index_corpus(
        {
            "gold": "therapy zzbrainbridge marker",
            "decoy": "therapy common filler the and or protein expression cells",
            "noise": "unrelated content the and or only common words here",
        }
    )
    query = "zzbrainmiss zzbrainbridge"
    return loop, query, "gold"


def test_brain_loop_miss_teach_reretrieve() -> None:
    loop, query, gold_id = _build_miss_fixture()

    hits_before = loop.retrieve(query, limit=5)
    trace_before = loop.explain_last()
    pool_before = {h.doc_id for h in hits_before}
    gold_in_pool_before = gold_id in pool_before or any(h.doc_id == gold_id for h in hits_before)

    teach = loop.teach_from_miss(query, gold_id, bridge_terms=("zzbrainbridge",))
    assert teach["taught"] is True
    assert "zzbrainmiss" in teach["supplement"]

    hits_after = loop.retrieve(query, limit=5)
    trace_after = loop.explain_last()
    pool_after_ids = {h.doc_id for h in hits_after}
    assert gold_id in pool_after_ids or any(h.doc_id == gold_id for h in hits_after)

    if not gold_in_pool_before:
        assert gold_id in pool_after_ids or hits_after[0].doc_id == gold_id


def test_brain_loop_explain_last_trace_fields() -> None:
    loop = BrainLoop()
    loop.index_corpus(FIXTURE_CORPUS)
    loop.retrieve("cat purrs", limit=3)
    ex = loop.explain_last()

    assert ex["query"] == "cat purrs"
    assert ex["route_mode"]
    assert ex["query_primes"]
    assert ex["filter_steps"] is not None
    assert ex["hits"]
    assert ex["hits"][0]["reasons"]


def test_brain_loop_wraps_registry_and_semantic() -> None:
    loop = BrainLoop()
    assert loop.semantic is loop.retriever.semantic
    assert loop.registry is loop.semantic.registry


def test_brain_loop_explain_last_neuron_room() -> None:
    loop = BrainLoop(enable_neuron_room=True)
    loop.index_corpus(FIXTURE_CORPUS)
    loop.retrieve("cancer mutation rare", limit=3)
    ex = loop.explain_last()

    assert "neuron_room" in ex
    nr = ex["neuron_room"]
    assert nr["status"] == "OPEN"
    assert nr["n_wings"] == 32
    assert nr["L01_coord"] is not None
    assert nr["k"] >= 2
