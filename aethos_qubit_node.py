"""
aethos_qubit_node.py - any lattice node as a qubit, backed by exact statevectors.

Unifies the electron model (section_02_electron.md, section_05_measurement)
with the existing AETHOS quantum stack:

  - aethos_quantum.TwoQubitRegister   exact 2-qubit statevector + CHSH
  - aethos_physics.bell_correlation_*  the ocean-fill (Werner) contract
  - aethos_ocean_graph                 phi/coherence between sites

This module adds the pieces needed to make "any node is a qubit, entangle
any two, measure one to collapse many" first-class and reusable:

  QubitNode          a lattice prime carrying a qubit identity
  entangle_pair      two nodes -> singlet register with ocean fill phi
  werner_threshold   the exact classical/quantum crossover (phi* = 1/sqrt2)
  GHZ3               three-qubit GHZ statevector + Pauli-product measurement
                     (the Mermin all-or-nothing test of local realism)

Design note: the lattice node's z = X + iY is already a complex-amplitude
container. The same data structure hosts qubit amplitudes; only the update
rule changes (timing-collapse vs Born-projection). This module uses exact
statevectors so results are machine-precise, not Monte-Carlo.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np

from aethos_quantum import (
    TwoQubitRegister,
    bell_correlation_register,
    chsh_s_register,
)

# ---------------------------------------------------------------------------
# Single Pauli matrices (for the GHZ all-or-nothing test)
# ---------------------------------------------------------------------------

_I2 = np.array([[1, 0], [0, 1]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI = {"I": _I2, "X": _X, "Y": _Y, "Z": _Z}


# ---------------------------------------------------------------------------
# QubitNode: a lattice prime that carries a qubit
# ---------------------------------------------------------------------------

@dataclass
class QubitNode:
    """A lattice node (addressed by its prime) viewed as a qubit.

    The pure state lives in the X-Z plane of the Bloch sphere, parameterized
    by `axis` (the photon's commitment angle). This is enough for spin-style
    measurement and Bell/GHZ tests; the prime is the node's lattice address.
    """

    prime: int
    axis: float = 0.0          # Bloch angle in the X-Z plane

    def sub_quadrant(self) -> int:
        """The 4 sub-quadrant definitions = the photon's Bloch arc."""
        return int((self.axis % (2 * math.pi)) / (math.pi / 2))

    def born_up(self, measure_axis: float) -> float:
        """P(up) = cos^2((measure_axis - state)/2) - the model's cosine rule."""
        return math.cos((measure_axis - self.axis) / 2.0) ** 2


def pair_address(node_a: QubitNode, node_b: QubitNode) -> int:
    """Composite (FTA) entanglement address - any two nodes, any moment.

    Distinct unordered pairs get distinct addresses (Test 3 injectivity), so
    the 'every pair entanglable at any moment' claim is collision-free.
    """
    return node_a.prime * node_b.prime


# ---------------------------------------------------------------------------
# Entanglement via the existing exact 2-qubit register + ocean fill
# ---------------------------------------------------------------------------

def entangle_pair(phi_ab: float = 1.0, coherence: float = 1.0) -> TwoQubitRegister:
    """Singlet register dephased by ocean fill phi*C (Werner state).

    phi_ab * coherence = 1 -> ideal singlet (E = -cos(a-b), CHSH = 2sqrt2).
    Lower fill -> classical mixture; the crossover is werner_threshold().
    """
    from aethos_quantum import aethos_dephase

    reg = TwoQubitRegister.singlet()
    return aethos_dephase(reg, phi_ab, coherence)


def werner_threshold() -> float:
    """Ocean-fill value where CHSH crosses Bell's |S| = 2.

    For a Werner state with visibility v, CHSH = 2*sqrt(2)*v, so the quantum
    regime begins at v = 1/sqrt(2). The AETHOS ocean fill phi reproduces this.
    """
    return 1.0 / math.sqrt(2.0)


def chsh_at_fill(phi_ab: float) -> float:
    """Analytic CHSH at a given ocean fill (S = 2sqrt2 * phi)."""
    return 2.0 * math.sqrt(2.0) * phi_ab


# ---------------------------------------------------------------------------
# GHZ3: three-qubit all-or-nothing (Mermin) - stronger than Bell
# ---------------------------------------------------------------------------

class GHZ3:
    """Three-qubit GHZ state (|000> + |111>)/sqrt(2) with exact operators.

    The Mermin test: for the GHZ state, quantum mechanics predicts the four
    joint observables
        <X X X> = +1,  <X Y Y> = <Y X Y> = <Y Y X> = -1
    No assignment of definite +/-1 values to local X and Y on each qubit can
    satisfy all four at once (their product forces +1 = -1). Local realism
    fails in a SINGLE measurement round, not just on average.
    """

    def __init__(self):
        n = 8
        v = np.zeros(n, dtype=complex)
        v[0b000] = 1.0 / math.sqrt(2.0)
        v[0b111] = 1.0 / math.sqrt(2.0)
        self.state = v

    @staticmethod
    def _op(p0: str, p1: str, p2: str) -> np.ndarray:
        return np.kron(np.kron(_PAULI[p0], _PAULI[p1]), _PAULI[p2])

    def expectation(self, p0: str, p1: str, p2: str) -> float:
        op = self._op(p0, p1, p2)
        val = self.state.conj() @ (op @ self.state)
        return float(val.real)

    def mermin_operators(self) -> dict[str, float]:
        return {
            "XXX": self.expectation("X", "X", "X"),
            "XYY": self.expectation("X", "Y", "Y"),
            "YXY": self.expectation("Y", "X", "Y"),
            "YYX": self.expectation("Y", "Y", "X"),
        }

    def local_realism_contradiction(self) -> tuple[float, float]:
        """Return (quantum XXX, local-realism-forced XXX).

        Local hidden variables force XXX = (XYY)(YXY)(YYX) because each Y
        appears twice and squares to +1. Quantum mechanics gives the opposite
        sign - the contradiction.
        """
        ops = self.mermin_operators()
        forced = ops["XYY"] * ops["YXY"] * ops["YYX"]
        return ops["XXX"], forced


__all__ = [
    "QubitNode",
    "pair_address",
    "entangle_pair",
    "werner_threshold",
    "chsh_at_fill",
    "GHZ3",
    "bell_correlation_register",
    "chsh_s_register",
]
