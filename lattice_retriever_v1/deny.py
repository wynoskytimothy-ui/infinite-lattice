"""Imports forbidden until Stage 08 unlocks them."""

FORBIDDEN_UNTIL_STAGE_08 = frozenset({
    "aethos_multi_corpus",
    "eval_beir",
    "eval_beir_symbol",
    "aethos_vocab_gap_router",
    "aethos_bridges",
    "rank_bm25",
    "sentence_transformers",
    "sklearn",
    "faiss",
})

FORBIDDEN_ALWAYS = frozenset({
    "cross_encoder",
    "openai",
})


def assert_allowed_import(module_name: str) -> None:
    """Raise ImportError if module is forbidden by the stage gate."""
    root = module_name.split(".", 1)[0]
    if root in FORBIDDEN_ALWAYS:
        raise ImportError(
            f"Import of '{module_name}' is forbidden always "
            f"(see lattice_retriever_v1/deny.py)."
        )
    if root in FORBIDDEN_UNTIL_STAGE_08:
        raise ImportError(
            f"Import of '{module_name}' is forbidden until Stage 08 unlocks it "
            f"(see lattice_retriever_v1/deny.py)."
        )


check_import = assert_allowed_import
