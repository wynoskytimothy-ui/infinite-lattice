"""
AETHOS Clean Pipeline — single entry for index, query, evaluate.

Phase 1 (lean): folder (3) plane stack only — BIT-4 router + hub rank.
Builds on eval_beir internals; does not import legacy scatter paths.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from aethos_clean.gates import evaluate_gates, get_corpus_gates, get_preset
from aethos_clean.storage import storage_from_bundle
from aethos_clean.types import CleanEvalResult, CleanQueryResult, StorageReport
from beir_data_root import resolve_beir_root
from eval_beir import (
    _score_one_query,
    evaluate_dataset,
    load_paths,
    ndcg_at_k,
    recall_at_k,
)
from aethos_clean.composite_train import retrain_composites_on_bundle
from eval_checkpoint import EvalBundle, checkpoint_path, load_checkpoint, save_checkpoint


class CleanPipeline:
    """
    One-call RAG pipeline.

        pipe = CleanPipeline.from_beir("scifact", preset="lean")
        pipe.index()
        hits = pipe.query("Does aspirin reduce inflammation?", top_k=10)
        report = pipe.evaluate()
    """

    def __init__(
        self,
        dataset: str,
        *,
        preset: str = "lean",
        beir_root: str | Path | None = None,
        max_docs: int | None = None,
        max_queries: int | None = None,
        checkpoint: str | Path | None = None,
    ) -> None:
        self.dataset = dataset
        self.preset_name = preset
        self.preset = get_preset(preset)
        self.beir_root = Path(beir_root or resolve_beir_root())
        self.paths = load_paths(self.beir_root, dataset)
        self.max_docs = max_docs
        self.max_queries = max_queries
        self._checkpoint_path = Path(checkpoint) if checkpoint else None
        self._bundle: EvalBundle | None = None
        self._queries: dict[str, str] = {}
        self._qrels: dict[str, dict[str, int]] = {}
        self._qids: list[str] = []

    @classmethod
    def from_beir(
        cls,
        dataset: str,
        *,
        preset: str = "lean",
        beir_root: str | Path | None = None,
        max_docs: int | None = None,
        max_queries: int | None = None,
        checkpoint: str | Path | None = None,
    ) -> CleanPipeline:
        return cls(
            dataset,
            preset=preset,
            beir_root=beir_root,
            max_docs=max_docs,
            max_queries=max_queries,
            checkpoint=checkpoint,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        preset: str = "lean",
    ) -> CleanPipeline:
        bundle = load_checkpoint(path)
        pipe = cls(
            bundle.dataset,
            preset=preset,
            checkpoint=path,
        )
        pipe._bundle = bundle
        pipe._apply_query_window()
        return pipe

    @property
    def indexed(self) -> bool:
        return self._bundle is not None

    @property
    def bundle(self) -> EvalBundle:
        if self._bundle is None:
            raise RuntimeError("call index() first")
        return self._bundle

    def _apply_query_window(self) -> None:
        assert self._bundle is not None
        self._queries = dict(self._bundle.queries)
        self._qrels = dict(self._bundle.qrels)
        self._qids = list(self._bundle.qids)
        if self.max_queries is not None:
            self._qids = self._qids[: self.max_queries]

    def index(
        self,
        *,
        rebuild: bool = False,
        skip_training: bool = False,
        save: bool | str | Path = True,
        verbose: bool = False,
    ) -> StorageReport:
        """Build corpus index (ingest + hubs + BIT 3/4 + training)."""
        if self._bundle is not None and not rebuild:
            return self.storage_report()

        ckpt = self._checkpoint_path
        if ckpt is None and save:
            ckpt = checkpoint_path(self.dataset, self.preset.mode)

        if ckpt and ckpt.exists() and not rebuild:
            self._bundle = load_checkpoint(ckpt)
            self._apply_query_window()
            return self.storage_report()

        ephemeral: Path | None = None
        save_arg: bool | str | Path = True
        if save is False:
            tmp = tempfile.NamedTemporaryFile(suffix=".eval.pkl", delete=False)
            ephemeral = Path(tmp.name)
            tmp.close()
            save_arg = ephemeral
        elif isinstance(save, (str, Path)):
            save_arg = save
        elif ckpt is not None:
            save_arg = ckpt

        evaluate_dataset(
            self.paths,
            mode=self.preset.mode,
            max_docs=self.max_docs,
            max_queries=self.max_queries,
            kappa_candidate_cap=self.preset.pool_cap,
            lambda_kappa=self.preset.lambda_kappa if self.preset.kappa_scoring else None,
            save_checkpoint=save_arg,
            build_only=True,
            skip_training=skip_training,
            verbose=verbose,
            train_mode=self.preset.train_mode,
            max_composite_anchors=self.preset.max_composite_anchors,
            max_composite_meta=self.preset.max_composite_meta,
            max_composite_negatives=self.preset.max_composite_negatives,
            clear_bad_correlation=self.preset.clear_bad_correlation,
        )

        load_from = Path(save_arg) if isinstance(save_arg, (str, Path)) else checkpoint_path(
            self.dataset, self.preset.mode
        )
        self._bundle = load_checkpoint(load_from)
        self._apply_query_window()
        if ephemeral is not None:
            ephemeral.unlink(missing_ok=True)
        elif save is not False and isinstance(save_arg, (str, Path)):
            save_checkpoint(self._bundle, save_arg)
        return self.storage_report()

    def storage_report(self) -> StorageReport:
        return storage_from_bundle(self.bundle, pool_cap=self.preset.pool_cap)

    def retrain_composites(
        self,
        *,
        save: bool | str | Path = True,
        verbose: bool = True,
    ) -> int:
        """Fast composite-only train on an existing checkpoint (no full rebuild)."""
        if self._bundle is None:
            raise RuntimeError("load or index a checkpoint first")
        n = retrain_composites_on_bundle(
            self._bundle,
            dataset=self.dataset,
            mode=self.preset.mode,
            max_new_anchors=self.preset.max_composite_anchors,
            max_new_meta=self.preset.max_composite_meta,
            max_new_negatives=self.preset.max_composite_negatives,
            clear_bad_correlation=self.preset.clear_bad_correlation,
            verbose=verbose,
        )
        if save:
            out = (
                Path(save)
                if isinstance(save, (str, Path))
                else (self._checkpoint_path or checkpoint_path(self.dataset, self.preset.mode))
            )
            save_checkpoint(self._bundle, out)
            if verbose:
                print(f"  checkpoint updated: {Path(out).name}", flush=True)
        return n

    def _attractor_index(self):
        from pipeline.bit_03_doc_attractor_set import build_attractor_index_from_hub_signatures

        b = self.bundle
        if b.attractor_index is not None:
            return b.attractor_index
        return build_attractor_index_from_hub_signatures(b.pipe.registry, b.hub_sigs)

    def query(self, text: str, *, top_k: int = 10) -> CleanQueryResult:
        """Score one query against the indexed corpus."""
        b = self.bundle
        t0 = time.perf_counter()
        enable_kappa = self.preset.kappa_scoring and self.preset.lambda_kappa > 0
        result = _score_one_query(
            text,
            pipe=b.pipe,
            cidx=b.cidx,
            hub_sigs=b.hub_sigs,
            neighbor_map=b.neighbor_map,
            meet_index=b.meet_index,
            sub_comp_idx=b.sub_comp_idx,
            comp_idx=b.comp_idx,
            phrase_idx=b.phrase_idx,
            anchor_idx=b.anchor_idx,
            attractor_index=self._attractor_index(),
            kappa_candidate_cap=self.preset.pool_cap,
            enable_kappa_scoring=enable_kappa,
        )
        ms = (time.perf_counter() - t0) * 1000.0
        ranked = result.ranked[:top_k]
        return CleanQueryResult(
            query=text,
            ranked_ids=ranked,
            scores=None,
            latency_ms=ms,
            n_candidates=result.n_candidates,
            route_tier=result.route_tier,
            n_kappa_keys=result.n_kappa_keys,
        )

    def evaluate(
        self,
        *,
        check_gates: bool = True,
        verbose: bool = False,
    ) -> CleanEvalResult:
        """Run full query loop with metrics + optional gate check."""
        b = self.bundle
        attractor = self._attractor_index()
        enable_kappa = self.preset.kappa_scoring and self.preset.lambda_kappa > 0

        ndcgs: list[float] = []
        r10s: list[float] = []
        r100s: list[float] = []
        q_times: list[float] = []

        if verbose:
            print(
                f"  clean evaluate: {len(self._qids)} queries  "
                f"preset={self.preset_name}  pool_cap={self.preset.pool_cap}",
                flush=True,
            )

        for qi, qid in enumerate(self._qids):
            t0 = time.perf_counter()
            result = _score_one_query(
                self._queries[qid],
                pipe=b.pipe,
                cidx=b.cidx,
                hub_sigs=b.hub_sigs,
                neighbor_map=b.neighbor_map,
                meet_index=b.meet_index,
                sub_comp_idx=b.sub_comp_idx,
                comp_idx=b.comp_idx,
                phrase_idx=b.phrase_idx,
                anchor_idx=b.anchor_idx,
                attractor_index=attractor,
                kappa_candidate_cap=self.preset.pool_cap,
                enable_kappa_scoring=enable_kappa,
            )
            q_times.append((time.perf_counter() - t0) * 1000.0)
            rel = self._qrels[qid]
            ndcgs.append(ndcg_at_k(result.ranked, rel, 10))
            r10s.append(recall_at_k(result.ranked, rel, 10))
            r100s.append(recall_at_k(result.ranked, rel, 100))
            if verbose and (qi + 1) % 50 == 0:
                print(
                    f"  {qi + 1}/{len(self._qids)}  "
                    f"NDCG@10 avg={sum(ndcgs) / len(ndcgs):.4f}",
                    flush=True,
                )

        n_q = max(len(self._qids), 1)
        sorted_t = sorted(q_times) if q_times else [0.0]
        p50 = sorted_t[len(sorted_t) // 2]
        p99 = sorted_t[int(len(sorted_t) * 0.99)] if len(sorted_t) > 1 else p50
        ndcg_mean = sum(ndcgs) / n_q
        r10_mean = sum(r10s) / n_q
        r100_mean = sum(r100s) / n_q

        storage = self.storage_report()
        cg = get_corpus_gates(self.dataset)
        gate_passed = None
        gate_summary = None
        if check_gates:
            gate_report_obj = evaluate_gates(
                dataset=self.dataset,
                preset=self.preset_name,
                ndcg10=ndcg_mean,
                recall10=r10_mean,
                recall100=r100_mean,
                p50_query_ms=p50,
                p99_query_ms=p99,
                hot_bytes_per_doc=storage.hot_bytes_per_doc,
            )
            gate_passed = gate_report_obj.passed
            gate_summary = gate_report_obj.summary()

        return CleanEvalResult(
            dataset=self.dataset,
            preset=self.preset_name,
            mode=self.preset.mode,
            n_docs=b.n_docs,
            n_queries=len(self._qids),
            ndcg10=ndcg_mean,
            recall10=r10_mean,
            recall100=r100_mean,
            p50_query_ms=p50,
            p99_query_ms=p99,
            p50_ingest_ms=b.p50_ingest_ms,
            storage=storage,
            bm25_ref=cg.bm25_ref,
            gate_passed=gate_passed,
            gate_report=gate_summary,
        )
