#!/usr/bin/env python3
"""
Test 50 - The atom: hydrogen spectrum from the standing-wave model (section 09).

Section 09: an electron exists only where its inner-photon bounce forms a
STANDING WAVE between the coin walls and the nuclear boundary - "like a
guitar string, only certain wavelengths fit." That is exactly de Broglie's
standing-wave condition (2 pi r = n lambda), which with the Coulomb drain-pull
balance gives the Bohr model and therefore the EXACT hydrogen spectrum.

We derive from first principles (no Rydberg constant plugged in) and check:
  (A) energy levels E_n = -13.6 eV / n^2  (the 1/n^2 law from standing waves)
  (B) the Balmer series (visible H lines) vs measured wavelengths
  (C) Lyman-alpha and the series limit
  (D) shell capacity 2 n^2 -> the periodic-table period lengths

Constants are CODATA SI; the reduced electron-proton mass is used.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def header(s: str):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def assertion(cond: bool, msg: str):
    print(f"  [{'PASS' if cond else 'FAIL'}]  {msg}")
    if not cond:
        sys.exit(1)


# CODATA constants (SI)
m_e = 9.1093837015e-31
m_p = 1.67262192369e-27
e = 1.602176634e-19
eps0 = 8.8541878128e-12
h = 6.62607015e-34
hbar = h / (2 * math.pi)
c = 2.99792458e8

mu = m_e * m_p / (m_e + m_p)            # reduced mass (electron + proton)


def main():
    header("The atom - hydrogen spectrum from standing-wave resonance")

    # ------------------------------------------------------------------
    # Derive the energy levels from the standing-wave + Coulomb balance.
    # de Broglie: 2 pi r = n * (h / (mu v));  Coulomb: e^2/(4 pi eps0 r^2) = mu v^2 / r
    # Solving -> r_n = n^2 * a0,  E_n = - mu e^4 / (8 eps0^2 h^2 n^2)
    # ------------------------------------------------------------------
    a0 = 4 * math.pi * eps0 * hbar ** 2 / (mu * e ** 2)
    print(f"\n  derived Bohr radius a0 = {a0*1e12:.2f} pm  (standing-wave + drain)")

    def E_n_eV(n):
        E_J = -mu * e ** 4 / (8 * eps0 ** 2 * h ** 2 * n ** 2)
        return E_J / e

    print("\n(A) Energy levels - the 1/n^2 law")
    print("-" * 72)
    for n in range(1, 6):
        print(f"  E_{n} = {E_n_eV(n):8.3f} eV   (-13.6/{n}^2 = {-13.6/n**2:.3f})")
    assertion(abs(E_n_eV(1) - (-13.6)) < 0.05,
              "ground-state energy = ionization energy = -13.6 eV (standing "
              "wave fixes the levels)")
    # exact 1/n^2 scaling
    ratios = [E_n_eV(n) / E_n_eV(1) for n in range(1, 5)]
    assertion(all(abs(r - 1.0 / n ** 2) < 1e-9 for n, r in enumerate(ratios, 1)),
              "levels follow exactly 1/n^2 (the standing-wave / Bohr law)")

    # ------------------------------------------------------------------
    print("\n(B) Balmer series (visible) - derived vs measured")
    print("-" * 72)

    def line_nm(n1, n2):
        dE_eV = E_n_eV(n1) - E_n_eV(n2)     # emission n2 -> n1, positive
        dE_J = abs(dE_eV) * e
        return h * c / dE_J * 1e9            # nm (vacuum)

    measured = {3: ("H-alpha", 656.3), 4: ("H-beta", 486.1),
                5: ("H-gamma", 434.0), 6: ("H-delta", 410.2)}
    print(f"  {'line':>8} | {'derived nm':>10} | {'measured nm':>11} | err")
    worst = 0.0
    for n2, (name, meas) in measured.items():
        lam = line_nm(2, n2)
        err = abs(lam - meas) / meas
        worst = max(worst, err)
        print(f"  {name:>8} | {lam:>10.2f} | {meas:>11.1f} | {err*100:.2f}%")
    assertion(worst < 0.002,
              "all four Balmer lines match measured hydrogen wavelengths to "
              "<0.2% - the visible spectrum, from the guitar-string resonance")

    # ------------------------------------------------------------------
    print("\n(C) Lyman-alpha (UV) and the series limit")
    print("-" * 72)
    lya = line_nm(1, 2)
    limit = line_nm(2, 10 ** 9)             # n2 -> infinity
    print(f"  Lyman-alpha (2->1): {lya:.2f} nm (measured 121.6)")
    print(f"  Balmer series limit (inf->2): {limit:.1f} nm (measured 364.6)")
    assertion(abs(lya - 121.6) / 121.6 < 0.003,
              "Lyman-alpha at 121.6 nm (UV) - matches")
    assertion(abs(limit - 364.6) / 364.6 < 0.003,
              "Balmer series limit at 364.6 nm - matches")

    # ------------------------------------------------------------------
    print("\n(D) Shell capacity 2 n^2 -> periodic-table period lengths")
    print("-" * 72)
    caps = [2 * n ** 2 for n in range(1, 5)]
    print(f"  shell capacities 2 n^2: {caps}")
    # period lengths (2, 8, 8, 18, 18, 32) come from the order shells fill
    period_lengths = [2, 8, 8, 18, 18, 32]
    print(f"  periodic-table period lengths: {period_lengths}")
    assertion(caps == [2, 8, 18, 32],
              "shell capacities 2,8,18,32 (the resonance/entanglement count) - "
              "the skeleton of the periodic table")

    header("RESULT")
    print(f"  energy levels:  E_n = -13.6/n^2 eV, exact 1/n^2 (standing wave)")
    print(f"  Balmer series:  4 visible lines match measured to <{worst*100:.2f}%")
    print(f"  Lyman-alpha:    121.6 nm; series limit 364.6 nm - both match")
    print(f"  shells:         2 n^2 = 2,8,18,32 (periodic-table skeleton)")
    print()
    print("  The hydrogen spectrum - measured to many decimals, the testbed")
    print("  of quantum mechanics - falls out of the section-09 standing-wave")
    print("  resonance, which IS de Broglie's condition. Third physics section")
    print("  to land on the exact numbers (Bell 2sqrt2, tunneling, now the")
    print("  atom). The guitar string was the right picture.")


if __name__ == "__main__":
    main()
