#!/usr/bin/env python3
"""Merge book/output/*.docx into one master file (order = reading order)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "book" / "output"

PARTS = [
    "00_Front_Matter.docx",
    "01_The_Lattice_of_Reality.docx",
    "02_The_Pi_Lattice.docx",
    "03_ThreeD_Complex_Plane.docx",
    "03_Particles_Electron_Proton_Neutron.docx",
    "04_QM_Measurement_Entanglement_Tunneling_DoubleSlit.docx",
    "05_Atoms_and_Cosmology.docx",
    "06_Synthesis_and_Predictions.docx",
    "07_Appendices.docx",
]

MASTER = OUT / "Packets_and_Strings_Full.docx"


def build_all() -> None:
    ps1 = ROOT / "book" / "build.ps1"
    subprocess.run(
        ["powershell", "-NoProfile", "-File", str(ps1), "all"],
        cwd=ROOT,
        check=True,
    )


def merge() -> None:
    try:
        from docx import Document
        from docxcompose.composer import Composer
    except ImportError as exc:
        raise SystemExit("pip install docxcompose python-docx") from exc

    missing = [p for p in PARTS if not (OUT / p).exists()]
    if missing:
        print("Missing parts (run build first):", ", ".join(missing))
        build_all()

    still = [p for p in PARTS if not (OUT / p).exists()]
    if still:
        raise FileNotFoundError(f"Could not build: {still}")

    master = Document(str(OUT / PARTS[0]))
    composer = Composer(master)
    for part in PARTS[1:]:
        composer.append(Document(str(OUT / part)))
        print(f"  + {part}")

    OUT.mkdir(parents=True, exist_ok=True)
    composer.save(str(MASTER))
    print(f"Wrote {MASTER}")


def main() -> None:
    if "--no-build" not in sys.argv:
        build_all()
    merge()


if __name__ == "__main__":
    main()
