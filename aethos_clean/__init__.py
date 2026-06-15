"""
AETHOS Clean Pipeline — ground-up RAG product shell.

Phase 1: lean preset (BIT-4 router + hub rank on folder 3 engine).
"""

from aethos_clean.gates import (
    GateReport,
    CorpusGates,
    PresetGates,
    evaluate_gates,
    get_corpus_gates,
    get_preset,
    load_gates,
)
from aethos_clean.pipeline import CleanPipeline
from aethos_clean.types import CleanEvalResult, CleanQueryResult, StorageReport

__all__ = [
    "CleanPipeline",
    "CleanEvalResult",
    "CleanQueryResult",
    "StorageReport",
    "GateReport",
    "CorpusGates",
    "PresetGates",
    "evaluate_gates",
    "get_corpus_gates",
    "get_preset",
    "load_gates",
]
