"""Hot storage budget — cert + hub only (Phase 1)."""

from __future__ import annotations

from aethos_clean.types import StorageReport


def storage_from_bundle(bundle, *, pool_cap: int) -> StorageReport:
    fp = float(getattr(bundle, "bytes_per_doc", 0.0) or 0.0)
    hub = float(getattr(bundle, "hub_bytes_per_doc", 0.0) or 0.0)
    return StorageReport(
        fingerprint_bytes_per_doc=fp,
        hub_bytes_per_doc=hub,
        hot_bytes_per_doc=fp + hub,
        n_docs=int(getattr(bundle, "n_docs", 0) or 0),
        pool_cap=pool_cap,
    )
