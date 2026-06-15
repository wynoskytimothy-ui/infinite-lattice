#!/usr/bin/env python3
"""Export merged book .docx to PDF via Microsoft Word (docx2pdf)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "book" / "output"
SRC = OUT / "Packets_and_Strings_Full.docx"
DST = OUT / "Packets_and_Strings_Full.pdf"


def ensure_docx() -> None:
    if SRC.exists():
        return
    print(f"Missing {SRC.name}; building merged docx first...")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "merge_book_docx.py")],
        cwd=ROOT,
        check=True,
    )


def export_pdf() -> None:
    try:
        from docx2pdf import convert
    except ImportError as exc:
        raise SystemExit(
            "pip install docx2pdf\n"
            "(Windows only — uses installed Microsoft Word for layout-faithful PDF export)"
        ) from exc

    if not SRC.exists():
        if "--no-build" in sys.argv:
            raise FileNotFoundError(
                f"Missing {SRC}. Run npm run book:full first, or omit --no-build."
            )
        ensure_docx()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Converting {SRC.name} -> {DST.name} (Word export)...")
    convert(str(SRC), str(DST))
    print(f"Wrote {DST} ({DST.stat().st_size:,} bytes)")


def main() -> None:
    export_pdf()


if __name__ == "__main__":
    main()
