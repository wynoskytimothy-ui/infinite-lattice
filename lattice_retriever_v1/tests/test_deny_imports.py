"""Gate: deny.py import enforcement."""



import pytest



from lattice_retriever_v1.deny import (

    FORBIDDEN_ALWAYS,

    FORBIDDEN_UNTIL_STAGE_08,

    assert_allowed_import,

    check_import,

)





@pytest.mark.parametrize("module_name", sorted(FORBIDDEN_UNTIL_STAGE_08))

def test_assert_allowed_import_rejects_until_stage_08(module_name: str) -> None:

    with pytest.raises(ImportError, match="forbidden until Stage 08"):

        assert_allowed_import(module_name)





@pytest.mark.parametrize("module_name", sorted(FORBIDDEN_ALWAYS))

def test_assert_allowed_import_rejects_always(module_name: str) -> None:

    with pytest.raises(ImportError, match="forbidden always"):

        assert_allowed_import(module_name)





@pytest.mark.parametrize(

    "module_name",

    [

        "aethos_words",

        "aethos_promotion",

        "lattice_retriever_v1.stage08_retrieve",

        "stage08_retrieve",

    ],

)

def test_assert_allowed_import_allows_stage_stack(module_name: str) -> None:

    assert_allowed_import(module_name)





def test_check_import_is_alias() -> None:

    assert check_import is assert_allowed_import





def test_rejects_submodule_of_forbidden_root() -> None:

    with pytest.raises(ImportError, match="rank_bm25"):

        assert_allowed_import("rank_bm25.BM25Okapi")


