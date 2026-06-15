"""
Quantized attractor keys κ(z, ζ) — BIT 2 geometry only.

No SpacetimeCell, no RAG, no registry.
"""

from __future__ import annotations

from typing import Iterable

from plane3d.psi import ComplexPlane3D

AttractorKey = tuple[int, int, int]

DEFAULT_QUANTIZE = 1.0


def kappa(
    z: complex,
    zeta: float,
    *,
    quantize: float = DEFAULT_QUANTIZE,
) -> AttractorKey:
    q = quantize if quantize > 0 else DEFAULT_QUANTIZE
    return (
        int(round(z.real / q)),
        int(round(z.imag / q)),
        int(round(zeta / q)),
    )


def kappa_psi(psi: ComplexPlane3D, *, quantize: float = DEFAULT_QUANTIZE) -> AttractorKey:
    return kappa(psi.z, psi.zeta, quantize=quantize)


def attractor_neighbors(key: AttractorKey, *, radius: int = 1) -> set[AttractorKey]:
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    rx, ry, rz = key
    out: set[AttractorKey] = set()
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                out.add((rx + dx, ry + dy, rz + dz))
    return out


def keys_from_psi(points: Iterable[ComplexPlane3D], *, quantize: float = DEFAULT_QUANTIZE) -> set[AttractorKey]:
    return {kappa_psi(p, quantize=quantize) for p in points}
