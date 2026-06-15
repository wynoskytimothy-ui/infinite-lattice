# Packets and Strings — book build

Generates `.docx` chapters from Node scripts using `docx`.

## Setup

Requires [Node.js](https://nodejs.org/) (includes `npm`). Installed via winget: **Node v24 LTS**.

If `npm` is not recognized, the terminal was opened **before** install. Either **restart the terminal** (or Cursor), or run:

```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')
```

Or use the helper script (refreshes PATH automatically):

```powershell
.\book\build.ps1 front
.\book\build.ps1 ch1
.\book\build.ps1 appendices
.\book\build.ps1 all
.\book\build.ps1 full
.\book\build.ps1 pdf
```

```bash
npm install
```

## Build

```bash
npm run book:front
npm run book:ch1
npm run book:ch2
npm run book:ch3
npm run book:ch4
npm run book:ch5
npm run book:ch6
npm run book:appendices
```

Physics E-check (Ch 16/17 discriminators):

```powershell
python scripts/calibrate_discriminators.py
python scripts/pattern_why_discriminators.py
python aethos_physics.py   # includes report_discriminator_calibration
```

Why those values: `derivations/book_ch17_why_calibration_patterns.md`

Output:

- `book/output/00_Front_Matter.docx`
- `book/output/01_The_Lattice_of_Reality.docx`
- `book/output/02_The_Pi_Lattice.docx`
- `book/output/03_Particles_Electron_Proton_Neutron.docx`
- `book/output/04_QM_Measurement_Entanglement_Tunneling_DoubleSlit.docx`
- `book/output/05_Atoms_and_Cosmology.docx`
- `book/output/06_Synthesis_and_Predictions.docx`
- `book/output/07_Appendices.docx` (A–G, incl. quantum formula sheet)
- `book/output/Packets_and_Strings_Full.docx` (merged master — `npm run book:full`)
- `book/output/Packets_and_Strings_Full.pdf` (Word export — `npm run book:pdf`)

### PDF export

Requires **Microsoft Word** (installed on this machine) and:

```powershell
pip install docx2pdf
npm run book:pdf
```

Builds the merged `.docx` first if it is missing. Re-export only from an existing docx:

```powershell
python scripts/export_book_pdf.py --no-build
```

## Conventions

**Ontology:** [`ONTOLOGY.md`](../ONTOLOGY.md) — **π lattice** (Ch 1–2) vs **3D complex plane** (Ch 3–7). Deprecated: prime lattice, infinity lattice.

- Narrative scripts live here; formal proofs stay in `derivations/`.
- Ch 1–2 (π lattice) → `derivations/section_01_derivations.md`, `book_ch02_pi_lattice_derivations.md`, `pi/constructive_pi.py`.
- Ch 3–7 (3D complex plane) → `derivations/book_ch03-05_3d_complex_plane.md`, `aethos_complex_plane.py`.
- Ch 8–10 (particles) → `derivations/book_ch03-05_derivations.md` (C1–C3).
- Ch 11–14 (QM) → `derivations/book_ch06-09_derivations.md` (C5 Bell).
- Ch 15–19 (atom/cosmo) → `derivations/book_ch10-14_derivations.md`.
- Ch 20–22 (synthesis) → `book_ch15-17_derivations.md`, `book_ch17_hidden_patterns_audit.md`.
- Front matter & appendices → `derivations/book_front_appendices_derivations.md`.
- Cell width = `λ_C` (full Compton); coin half-width `L = λ_C/2` (C1).
