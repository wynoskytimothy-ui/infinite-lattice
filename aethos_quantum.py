"""
AETHOS minimal circuit simulator (2-qubit MVP).

Not a full QC stack: small statevector, standard gates, Born sampling,
and C7 Stage-B noise from ocean fill phi_AB + coherence C.

Ocean graph supplies pair observables; this module runs Bell/CHSH on a register.
"""

from __future__ import annotations

import cmath
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from aethos_physics import (
    bell_correlation_phi_fill,
    bell_correlation_qm,
    chsh_s_quantum,
)

if TYPE_CHECKING:
    from aethos_ocean_graph import OceanGraph, ObservablePair


_SQRT2_INV = 1.0 / math.sqrt(2.0)

# Single-qubit gates (qubit index 0 = |q0 q1⟩ MSB)
_I = ((1.0, 0.0), (0.0, 1.0))
_X = ((0.0, 1.0), (1.0, 0.0))
_Z = ((1.0, 0.0), (0.0, -1.0))
_H = ((_SQRT2_INV, _SQRT2_INV), (_SQRT2_INV, -_SQRT2_INV))


def _mat2_vec(gate: Sequence[Sequence[float]], v0: complex, v1: complex) -> tuple[complex, complex]:
    return (
        gate[0][0] * v0 + gate[0][1] * v1,
        gate[1][0] * v0 + gate[1][1] * v1,
    )


def _kron2(
    a: Sequence[Sequence[float]],
    b: Sequence[Sequence[float]],
) -> tuple[tuple[complex, ...], ...]:
    """4x4 Kronecker for 2-qubit gate (q0 ⊗ q1)."""
    rows: list[tuple[complex, ...]] = []
    for i in range(2):
        for j in range(2):
            row: list[complex] = []
            for k in range(2):
                for m in range(2):
                    row.append(a[i][k] * b[j][m])
            rows.append(tuple(row))
    return tuple(rows)


def _cnot_matrix() -> tuple[tuple[complex, ...], ...]:
    # |00>,|01>,|10>,|11> — control q0, target q1
    z = 0.0
    o = 1.0
    return (
        (o, z, z, z),
        (z, o, z, z),
        (z, z, z, o),
        (z, z, o, z),
    )


_CNOT = _cnot_matrix()


@dataclass
class TwoQubitRegister:
    """Pure 2-qubit state |ψ⟩ in basis |00⟩,|01⟩,|10⟩,|11⟩."""

    amps: tuple[complex, complex, complex, complex]

    @classmethod
    def zeros(cls) -> TwoQubitRegister:
        return cls((1.0, 0.0, 0.0, 0.0))

    @classmethod
    def singlet(cls) -> TwoQubitRegister:
        """|Ψ-⟩ = (|01⟩ - |10⟩) / √2 — E(a,b) = -cos(a-b) at full fill."""
        return cls((0.0, _SQRT2_INV, -_SQRT2_INV, 0.0))

    @classmethod
    def phi_plus(cls) -> TwoQubitRegister:
        return cls((_SQRT2_INV, 0.0, 0.0, _SQRT2_INV))

    def copy(self) -> TwoQubitRegister:
        return TwoQubitRegister(self.amps)

    def norm(self) -> float:
        return math.sqrt(sum(abs(a) ** 2 for a in self.amps))

    def normalize(self) -> TwoQubitRegister:
        n = self.norm()
        if n <= 0.0:
            return self
        s = 1.0 / n
        self.amps = tuple(a * s for a in self.amps)
        return self

    def probabilities(self) -> tuple[float, float, float, float]:
        return tuple(abs(a) ** 2 for a in self.amps)

    def apply_1q(self, qubit: int, gate: Sequence[Sequence[float]]) -> TwoQubitRegister:
        if qubit not in (0, 1):
            raise ValueError("qubit must be 0 or 1")
        a00, a01, a10, a11 = self.amps
        if qubit == 0:
            a00, a10 = _mat2_vec(gate, a00, a10)
            a01, a11 = _mat2_vec(gate, a01, a11)
        else:
            a00, a01 = _mat2_vec(gate, a00, a01)
            a10, a11 = _mat2_vec(gate, a10, a11)
        self.amps = (a00, a01, a10, a11)
        return self

    def apply_2q(self, gate4: Sequence[Sequence[complex | float]]) -> TwoQubitRegister:
        out = [0j] * 4
        for i in range(4):
            for j in range(4):
                out[i] += gate4[i][j] * self.amps[j]
        self.amps = tuple(out)
        return self

    def apply_h(self, qubit: int) -> TwoQubitRegister:
        return self.apply_1q(qubit, _H)

    def apply_x(self, qubit: int) -> TwoQubitRegister:
        return self.apply_1q(qubit, _X)

    def apply_z(self, qubit: int) -> TwoQubitRegister:
        return self.apply_1q(qubit, _Z)

    def apply_cnot(self, control: int = 0, target: int = 1) -> TwoQubitRegister:
        if control == 0 and target == 1:
            return self.apply_2q(_CNOT)
        if control == 1 and target == 0:
            # swap qubits, CNOT, swap back
            reg = self.copy()
            reg.swap_qubits().apply_2q(_CNOT).swap_qubits()
            self.amps = reg.amps
            return self
        raise ValueError("MVP supports CNOT(0->1) or CNOT(1->0) via swap")

    def swap_qubits(self) -> TwoQubitRegister:
        a00, a01, a10, a11 = self.amps
        self.amps = (a00, a10, a01, a11)
        return self

    def prepare_bell_singlet_circuit(self) -> TwoQubitRegister:
        """|00⟩ → X₁ → H₀ → CNOT₀₁  → |Ψ-⟩."""
        self.amps = (1.0, 0.0, 0.0, 0.0)
        self.apply_x(1).apply_h(0).apply_cnot(0, 1)
        return self.normalize()

    def sample_bitstring(self, rng: random.Random | None = None) -> tuple[int, int]:
        r = rng or random.Random()
        probs = self.probabilities()
        x = r.random()
        acc = 0.0
        idx = 3
        for i, p in enumerate(probs):
            acc += p
            if x <= acc:
                idx = i
                break
        return (idx // 2, idx % 2)


def pauli_theta_matrix(theta_rad: float) -> tuple[tuple[float, ...], ...]:
    """σ(θ) = cos(θ) σ_z + sin(θ) σ_x."""
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return ((c, s), (s, -c))


def expectation_2q(
    reg: TwoQubitRegister,
    gate_a: Sequence[Sequence[float]],
    gate_b: Sequence[Sequence[float]],
) -> float:
    """⟨ψ| G_a ⊗ G_b |ψ⟩ for Hermitian 2×2 real Pauli combinations."""
    reg = reg.copy().normalize()
    a00, a01, a10, a11 = reg.amps

    def on_qubit(
        v00: complex,
        v01: complex,
        v10: complex,
        v11: complex,
        q: int,
        g: Sequence[Sequence[float]],
    ) -> tuple[complex, complex, complex, complex]:
        if q == 0:
            n00, n10 = _mat2_vec(g, v00, v10)
            n01, n11 = _mat2_vec(g, v01, v11)
            return n00, n01, n10, n11
        n00, n01 = _mat2_vec(g, v00, v01)
        n10, n11 = _mat2_vec(g, v10, v11)
        return n00, n01, n10, n11

    b00, b01, b10, b11 = on_qubit(a00, a01, a10, a11, 1, gate_b)
    f00, f01, f10, f11 = on_qubit(b00, b01, b10, b11, 0, gate_a)
    return float((f00 * a00.conjugate() + f01 * a01.conjugate() + f10 * a10.conjugate() + f11 * a11.conjugate()).real)


def bell_correlation_register(
    reg: TwoQubitRegister,
    angle_a_rad: float,
    angle_b_rad: float,
) -> float:
    """E = ⟨σ_a(θ_a) ⊗ σ_b(θ_b)⟩."""
    ga = pauli_theta_matrix(angle_a_rad)
    gb = pauli_theta_matrix(angle_b_rad)
    return expectation_2q(reg, ga, gb)


def chsh_s_register(
    reg: TwoQubitRegister,
    a: float = 0.0,
    a_prime: float = math.pi / 2,
    b: float = math.pi / 4,
    b_prime: float = 3 * math.pi / 4,
) -> float:
    return (
        bell_correlation_register(reg, a, b)
        - bell_correlation_register(reg, a, b_prime)
        + bell_correlation_register(reg, a_prime, b)
        + bell_correlation_register(reg, a_prime, b_prime)
    )


def aethos_dephase(
    reg: TwoQubitRegister,
    phi_ab: float,
    coherence: float = 1.0,
) -> TwoQubitRegister:
    """
    Werner-style ocean noise: keep fraction p = φ_AB · C of entangled weight.

    Uniform scaling of |01⟩,|10⟩ leaves Bell E unchanged; this leaks weight
    into |00⟩,|11⟩ so correlations drop when fill or coherence is low.
    """
    p = max(0.0, min(1.0, phi_ab * coherence))
    a00, a01, a10, a11 = reg.amps
    ent_sq = abs(a01) ** 2 + abs(a10) ** 2
    if ent_sq < 1e-30:
        return reg
    ent = math.sqrt(ent_sq)
    new_ent = p * ent
    leak = math.sqrt(max(0.0, ent_sq - new_ent**2))
    if ent_sq > 0:
        scale = new_ent / ent
        a01, a10 = a01 * scale, a10 * scale
    half = leak / math.sqrt(2.0)
    a00 = complex(a00) + half
    a11 = complex(a11) + half
    reg.amps = (a00, a01, a10, a11)
    return reg.normalize()


def entangled_weight(reg: TwoQubitRegister) -> float:
    """Fraction of probability in |01⟩ and |10⟩ subspace."""
    _, a01, a10, _ = reg.amps
    return abs(a01) ** 2 + abs(a10) ** 2


def bell_correlation_with_fill(
    reg: TwoQubitRegister,
    angle_a_rad: float,
    angle_b_rad: float,
    phi_ab: float,
    coherence: float = 1.0,
) -> float:
    """C7 contract on register: scales E by φ·C (singlet-like seed)."""
    phi_eff = max(0.0, min(1.0, phi_ab * coherence))
    return phi_eff * bell_correlation_register(reg, angle_a_rad, angle_b_rad)


def aethos_correlation_from_fill(
    angle_a_rad: float,
    angle_b_rad: float,
    phi_ab: float,
) -> float:
    """Analytic C7 contract (same as aethos_physics)."""
    return bell_correlation_phi_fill(angle_a_rad, angle_b_rad, phi_ab)


def chsh_s_aethos(phi_ab: float) -> float:
    """CHSH S using E = -φ cos(a-b) at fixed fill."""
    a, a_prime = 0.0, math.pi / 2
    b, b_prime = math.pi / 4, 3 * math.pi / 4
    return (
        aethos_correlation_from_fill(a, b, phi_ab)
        - aethos_correlation_from_fill(a, b_prime, phi_ab)
        + aethos_correlation_from_fill(a_prime, b, phi_ab)
        + aethos_correlation_from_fill(a_prime, b_prime, phi_ab)
    )


@dataclass(frozen=True)
class OceanPairSession:
    """2-qubit register seeded from an ocean observable pair."""

    node_a: int
    node_b: int
    phi_ab: float
    coherence: float
    species_a: str
    species_b: str


def register_from_ocean_pair(
    graph: OceanGraph,
    pair: ObservablePair,
    *,
    singlet: bool = True,
) -> TwoQubitRegister:
    """Build register; apply aethos_dephase from edge φ and C."""
    reg = TwoQubitRegister.singlet() if singlet else TwoQubitRegister.phi_plus()
    return aethos_dephase(reg, pair.phi, pair.coherence)


def session_from_ocean(
    graph: OceanGraph,
    node_a: int,
    node_b: int,
) -> tuple[OceanPairSession, TwoQubitRegister]:
    """Top matching edge between nodes, or φ=0 session if none."""
    edge = None
    for e in graph.edges:
        if {e.a, e.b} == {node_a, node_b}:
            edge = e
            break
    if edge is None:
        sess = OceanPairSession(node_a, node_b, 0.0, 0.0, "", "")
        return sess, aethos_dephase(TwoQubitRegister.singlet(), 0.0, 0.0)
    sa = graph.sites[edge.a]
    sb = graph.sites[edge.b]
    sess = OceanPairSession(
        node_a=edge.a,
        node_b=edge.b,
        phi_ab=edge.phi,
        coherence=edge.coherence,
        species_a=sa.chain_species,
        species_b=sb.chain_species,
    )
    reg = TwoQubitRegister.singlet()
    return sess, aethos_dephase(reg, edge.phi, edge.coherence)


def demo_bell_chsh() -> str:
    reg = TwoQubitRegister.singlet()
    a0, b0 = 0.0, math.pi / 4
    lines = [
        "AETHOS 2-qubit quantum MVP",
        "=" * 40,
        f"E singlet reg (0, pi/4)     = {bell_correlation_register(reg, a0, b0):.6f}",
        f"E QM analytic               = {bell_correlation_qm(a0, b0):.6f}",
        f"CHSH S register             = {chsh_s_register(reg):.6f}",
        f"CHSH S QM                   = {chsh_s_quantum():.6f}",
        "",
        "phi_AB fill (Stage B contract):",
        f"  phi=1.0  S_aethos         = {chsh_s_aethos(1.0):.6f}",
        f"  phi=0.5  S_aethos         = {chsh_s_aethos(0.5):.6f}",
        "",
        "Circuit prep |Ψ-⟩:",
    ]
    circ = TwoQubitRegister.zeros().prepare_bell_singlet_circuit()
    lines.append(f"  E circuit (0, pi/4)       = {bell_correlation_register(circ, a0, b0):.6f}")
    noisy = TwoQubitRegister.singlet().copy()
    aethos_dephase(noisy, 0.3, 0.8)
    lines.append(f"  E after dephase phi=0.3,C=0.8 = {bell_correlation_register(noisy, a0, b0):.6f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(demo_bell_chsh())
