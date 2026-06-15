"""
AETHOS physics calibration — empirical anchors, Sec 4 neutron pressure, Sec 5 measurement.

Maps derivation IDs from derivations/calibration_sheet.md to numeric values.
Separate from lattice/codec modules (aethos_active, aethos_lattice, …).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence
from enum import Enum

# CODATA-style constants (SI)
C = 299_792_458.0
H = 6.626_070_15e-34
HBAR = 1.054_571_817e-34
E_CHARGE = 1.602_176_634e-19
M_E = 9.109_383_7015e-31
M_P = 1.672_621_923_69e-27
M_N = 1.674_927_498_04e-27
MU_N = 5.050_783_7461e-27
MU_B = 9.274_010_0783e-24  # J/T
TAU_N = 879.4  # s, free neutron mean lifetime (E anchor — C3 consequence check)
TAU_WEAK_SM_EST = 885.7  # s, electroweak benchmark (comparison only — not primary mechanism)
Q_BETA_EV = 0.782_33
DM_NP_EV = 1.293_332  # neutron–proton mass excess (scale for captured energy)
R_PE = 1836.152_673_43
# |mu_n|/mu_N = g_eff * 2(m_e/m_n)  =>  g_eff ~ 1.76e3 (see section_04_derivations 4.8.1)
G_EFF_NEUTRON = 1.913 / (2 * (M_E / M_N))


def ev_to_joules(ev: float) -> float:
    return ev * 1e6 * E_CHARGE


def joules_to_ev(j: float) -> float:
    return j / E_CHARGE / 1e6


def compton_length(m: float = M_E) -> float:
    return H / (m * C)


def pump_frequency_cavity(m: float = M_E, l_bounce: float | None = None) -> float:
    """omega_b = pi * c / L with L ~ Compton length by default (Section 2)."""
    l_char = l_bounce if l_bounce is not None else compton_length(m)
    return math.pi * C / l_char


@dataclass(frozen=True)
class NeutronPressureCalibration:
    """
    Free-neutron limit: R_share ~ 0, Pi_pin ~ 1, P(t) = P_0 + alpha * Gamma_obs * t.

    t_escape = (P_c - P_0) / (alpha * Gamma_obs) = tau_n.
    """

    tau_n: float = TAU_N
    p_gap_j: float = 0.0
    omega_in0: float = 0.0
    pi_pin: float = 1.0
    alpha: float = 0.0
    gamma_obs: float = 0.0
    r_coin: float = 0.0
    sigma_obs: float = 0.0
    phi_obs: float = 0.0
    mode: str = ""

    @property
    def dP_dt(self) -> float:
        return self.alpha * self.gamma_obs


def calibrate_neutron_pressure(
    *,
    gap: str = "cavity",
    tau_n: float = TAU_N,
    pi_pin: float = 1.0,
    l_bounce: float | None = None,
) -> NeutronPressureCalibration:
    """
    gap:
      - "scale": P_gap = Q_beta, omega_in0 = Q_beta/hbar  => Gamma_obs = 1/tau_n (consistency lock).
      - "cavity": P_gap = Delta_m_np c^2, omega from Compton cavity (falsifiable micro choice).
      - "q_cavity": P_gap = Q_beta, omega from cavity.
    """
    q_j = ev_to_joules(Q_BETA_EV)
    dm_j = ev_to_joules(DM_NP_EV)
    l0 = l_bounce if l_bounce is not None else H / (2.0 * M_E * C)  # C1: L = lambda_C/2
    omega_geom = math.pi * C / (2.0 * l0)  # omega_b (Step 2)

    if gap == "scale":
        p_gap = q_j
        omega = q_j / HBAR
        mode = "scale_locked_Q_beta"
    elif gap == "cavity":
        p_gap = dm_j
        omega = omega_geom
        mode = "cavity_locked_Dm_np"
    elif gap == "q_cavity":
        p_gap = q_j
        omega = omega_geom
        mode = "cavity_locked_Q_beta"
    else:
        raise ValueError(f"unknown gap mode: {gap!r}")

    alpha = HBAR * omega * pi_pin
    gamma_obs = p_gap / (alpha * pi_pin * tau_n) if alpha > 0 else 0.0
    r_coin = l0
    sigma_obs = math.pi * r_coin**2
    phi_obs = gamma_obs / sigma_obs if sigma_obs > 0 else 0.0

    return NeutronPressureCalibration(
        tau_n=tau_n,
        p_gap_j=p_gap,
        omega_in0=omega,
        pi_pin=pi_pin,
        alpha=alpha,
        gamma_obs=gamma_obs,
        r_coin=r_coin,
        sigma_obs=sigma_obs,
        phi_obs=phi_obs,
        mode=mode,
    )


def g_eff_from_mu_n(mu_n_muN: float = -1.913_042_73) -> float:
    """From mu_n = -g_eff (2 m_e/m_n) mu_N with mu_n in units of mu_N."""
    return abs(mu_n_muN) / (2 * (M_E / M_N))


def k_fusion_from_r_pe(r_pe: float = R_PE, alpha_k: float = 1.0) -> float:
    return 1.0 - 1.0 / r_pe if alpha_k == 1.0 else (r_pe - 1.0) / (alpha_k * r_pe)


# --- Section 2: C1 geometry anchor (Step 2 gate) ---


def coin_half_width(m: float = M_E) -> float:
    """L = lambda_C/2 = h/(2 m c). Single coin anchor (architecture_mandates C1)."""
    return H / (2.0 * m * C)


def compton_wavelength(m: float = M_E) -> float:
    return 2.0 * coin_half_width(m)


def omega_bounce(m: float = M_E, l: float | None = None) -> float:
    """omega_b = pi c / (2L)."""
    length = l if l is not None else coin_half_width(m)
    return math.pi * C / (2.0 * length)


def f_bounce(m: float = M_E, l: float | None = None) -> float:
    return omega_bounce(m, l) / (2.0 * math.pi)


def k_s_from_geometry(m: float = M_E, l: float | None = None) -> float:
    """k_s = m omega_b^2 (GEOMETRY)."""
    ob = omega_bounce(m, l)
    return m * ob**2


def omega_hopping(kappa: float, m: float = M_E, l: float | None = None) -> float:
    """Omega(kappa) = Omega_0 (1 - kappa^2), Omega_0 = omega_b/2."""
    kappa = max(0.0, min(1.0, kappa))
    return 0.5 * omega_bounce(m, l) * (1.0 - kappa**2)


def delta_kappa(kappa: float, m: float = M_E, l: float | None = None) -> float:
    """Delta(kappa) in rad/s."""
    kappa = max(0.0, min(1.0, kappa))
    length = l if l is not None else coin_half_width(m)
    ks = k_s_from_geometry(m, length)
    return (ks * length**2 / (2.0 * HBAR)) * kappa**2


def pi_pin_kappa(kappa: float, m: float = M_E, l: float | None = None) -> float:
    d = delta_kappa(kappa, m, l)
    o = omega_hopping(kappa, m, l)
    return d / (d + o) if (d + o) > 0 else 0.0


def u_max_spring(m: float = M_E, l: float | None = None) -> float:
    """U_max at full compression — export to Step 3 K_f."""
    length = l if l is not None else coin_half_width(m)
    return 0.5 * k_s_from_geometry(m, length) * length**2


# --- Section 3: fusion geometry (Step 3 gate, C2 mandate) ---

PI_SQ_OVER_8 = math.pi**2 / 8.0


def k_f_pin(m: float = M_E, l: float | None = None) -> float:
    """K_f where Delta(K)=Omega(K) — hop/pin crossover (Pi_pin=0.5)."""
    length = l if l is not None else coin_half_width(m)
    omega0 = 0.5 * omega_bounce(m, length)
    a = delta_kappa(1.0, m, length)
    if a <= 0:
        return 0.0
    return math.sqrt(omega0 / (omega0 + a))


def k_f_hop_death() -> float:
    """F1: Omega(K_f)=0 => K_f=1 (limit)."""
    return 1.0


def e_locked_fusion(kappa: float, m: float = M_E, l: float | None = None) -> float:
    """E_locked(K) = U_max K^2 + m_e c^2 K/(1-K) in joules."""
    kappa = max(0.0, min(1.0 - 1.0e-15, kappa))
    umax = u_max_spring(m, l)
    mec2 = m * C**2
    return umax * kappa**2 + mec2 * kappa / (1.0 - kappa)


def r_pe_length(kappa: float, alpha_k: float = 1.0) -> float:
    """R_pe^length = 1/(1 - alpha K)."""
    kappa = max(0.0, min(1.0, kappa))
    denom = 1.0 - alpha_k * kappa
    if denom <= 0:
        return float("inf")
    return 1.0 / denom


def r_pe_energy(kappa: float, m: float = M_E, l: float | None = None) -> float:
    """R_pe^energy = 1 + E_locked/(m_e c^2)."""
    mec2 = m * C**2
    return 1.0 + e_locked_fusion(kappa, m, l) / mec2


def r_pe_spring_only() -> float:
    """Leading spring-only prediction R_pe^(0) = pi^2/8 (~1.234)."""
    return PI_SQ_OVER_8


def l_p_min_spring(m: float = M_E, l: float | None = None) -> float:
    """Minimum proton half-width from U_max normalization: L_p = (8/pi^2) L_0."""
    length = l if l is not None else coin_half_width(m)
    return length * 8.0 / math.pi**2


def lattice_mass_multiplier(r_pe_obs: float = R_PE) -> float:
    """E-check ratio R_pe^E / R_pe^(0). Geometry path: `m_lat_from_active_network`."""
    r0 = r_pe_spring_only()
    return r_pe_obs / r0 if r0 > 0 else float("inf")


def length_energy_consistency_residual(kappa: float, alpha_k: float = 1.0) -> float:
    """R_length(K) - R_energy(K); zero only at K=0 for alpha=1 (audit proof)."""
    return r_pe_length(kappa, alpha_k) - r_pe_energy(kappa)


def report_fusion_geometry() -> str:
    kpin = k_f_pin()
    r0 = r_pe_spring_only()
    mlat_def = m_lat_from_active_network()
    mlat_ref = m_lat_from_active_network(
        count=REFERENCE_NETWORK_COUNT,
        origin_max_depth=REFERENCE_NETWORK_DEPTH,
    )
    r_def = r_pe_model_with_lattice()
    r_ref = r_pe_model_reference_bootstrap()
    err_ref = 100.0 * abs(r_ref - R_PE) / R_PE
    lines = [
        "AETHOS fusion geometry (Sec 3, Step 3 gate)",
        "=" * 56,
        f"K_f^pin (Delta=Omega)  = {kpin:.6f}",
        f"K_f^hop (Omega=0)      = {k_f_hop_death():.6f}",
        f"R_pe^model,(0)         = {r0:.6f}  (pi^2/8)",
        f"L_p^min                = {l_p_min_spring():.6e} m",
        f"M_lat (default n=100)  = {mlat_def:.2f}",
        f"R_pe^pred (default)    = {r_def:.2f}  (E {R_PE:.2f})",
        f"M_lat (E-check n=80)   = {mlat_ref:.2f}  depth={REFERENCE_NETWORK_DEPTH}",
        f"R_pe^pred (E-check)    = {r_ref:.2f}  err {err_ref:.2f}%",
        f"R_pe^E (CODATA)        = {R_PE:.6f}",
        f"K_f^FIT (deprecated)   = {k_fusion_from_r_pe():.6f}",
        "",
        f"R_pe^energy(K_f^pin)   = {r_pe_energy(kpin):.6f}",
        f"R_pe^length(K_f^pin)   = {r_pe_length(kpin):.6f}",
        "",
    ]
    return "\n".join(lines)


# --- Section 4: neutron pressure escape (Step 4 gate, C3 mandate) ---


def p_c_gap_joules(kind: str = "q_beta") -> float:
    """
    Pressure threshold P_c - P_0 candidates (GEOMETRY / E scales).

    kind: u_max | q_beta | dm_np
    """
    if kind == "u_max":
        return u_max_spring()
    if kind == "q_beta":
        return ev_to_joules(Q_BETA_EV)
    if kind == "dm_np":
        return ev_to_joules(DM_NP_EV)
    raise ValueError(f"unknown P_c kind: {kind!r}")


def alpha_pressure_build(
    kappa: float = 1.0,
    *,
    pi_pin_override: float | None = None,
    m: float = M_E,
    l: float | None = None,
) -> float:
    """alpha = hbar omega_b Pi_pin — pressure-build rate coefficient (GEOMETRY)."""
    pi = pi_pin_override if pi_pin_override is not None else pi_pin_kappa(kappa, m, l)
    return HBAR * omega_bounce(m, l) * pi


def t_escape_pressure(p_gap_j: float, alpha: float, gamma_obs: float) -> float:
    """Primary C3 escape time: t = (P_c - P_0) / (alpha Gamma_obs)."""
    rate = alpha * gamma_obs
    return p_gap_j / rate if rate > 0 else float("inf")


def gamma_obs_for_t_escape(
    p_gap_j: float,
    alpha: float,
    tau: float,
) -> float:
    """Invert escape law — used for E calibration check, not primary definition (C3)."""
    return p_gap_j / (alpha * tau) if alpha > 0 and tau > 0 else 0.0


@dataclass(frozen=True)
class NeutronEscapePrediction:
    """Geometry-first neutron escape (C3 primary narrative)."""

    p_c_kind: str
    p_gap_j: float
    kappa: float
    pi_pin: float
    omega_in0: float
    alpha: float
    gamma_obs: float | None
    t_escape_s: float | None
    tau_n_e_check: float
    tau_weak_sm_est: float

    @property
    def weak_agreement_frac(self) -> float:
        if self.t_escape_s is None or self.t_escape_s <= 0:
            return float("nan")
        return self.t_escape_s / self.tau_weak_sm_est


def predict_neutron_escape(
    *,
    p_c_kind: str = "q_beta",
    kappa: float = 1.0,
    gamma_obs: float | None = None,
    tau_n: float = TAU_N,
) -> NeutronEscapePrediction:
    """
    Forward geometry chain: P_c candidate + alpha from C1 coin -> t_escape if Gamma_obs given.

    Without gamma_obs, returns alpha and P_gap only (tau remains OPEN until env model).
    """
    p_gap = p_c_gap_joules(p_c_kind)
    pi = pi_pin_kappa(kappa)
    alpha = alpha_pressure_build(kappa, pi_pin_override=pi)
    omega = omega_bounce()
    t_esc = t_escape_pressure(p_gap, alpha, gamma_obs) if gamma_obs is not None else None
    return NeutronEscapePrediction(
        p_c_kind=p_c_kind,
        p_gap_j=p_gap,
        kappa=kappa,
        pi_pin=pi,
        omega_in0=omega,
        alpha=alpha,
        gamma_obs=gamma_obs,
        t_escape_s=t_esc,
        tau_n_e_check=tau_n,
        tau_weak_sm_est=TAU_WEAK_SM_EST,
    )


def report_neutron_geometry() -> str:
    cal = calibrate_neutron_pressure(gap="cavity")
    gamma_cav = cal.gamma_obs
    pred_c = predict_neutron_escape(p_c_kind="dm_np", kappa=1.0, gamma_obs=gamma_cav)
    lines = [
        "AETHOS neutron escape (Sec 4, Step 4 gate, C3)",
        "=" * 56,
        "PRIMARY: P -> P_c pressure escape (gamma_obs outer trap rupture)",
        "NOT primary: SM weak rate (comparison only below)",
        "",
        f"omega_b (C1)           = {omega_bounce():.6e} rad/s",
        f"Pi_pin(kappa=1)        = {pi_pin_kappa(1.0):.6f}",
        f"alpha @ kappa=1        = {joules_to_ev(alpha_pressure_build(1.0)):.4f} MeV",
        f"U_max (P_c candidate)  = {joules_to_ev(p_c_gap_joules('u_max')):.4f} MeV",
        f"Q_beta (P_c candidate) = {Q_BETA_EV:.4f} MeV",
        f"Delta_m_np             = {DM_NP_EV:.4f} MeV",
        "",
        "Forward (needs Gamma_obs from environment — OPEN):",
        f"  t_escape @ dm_np     = needs Gamma_obs",
        "",
        "E calibration (cavity mode, inverted Gamma_obs):",
        f"  Gamma_obs            = {gamma_cav:.6e} s^-1",
        f"  t_escape check       = {pred_c.t_escape_s:.2f} s",
        f"  tau_n (E anchor)     = {TAU_N} s",
        f"  tau_weak SM est      = {TAU_WEAK_SM_EST} s (comparison)",
        "",
        f"  t / tau_n            = {(pred_c.t_escape_s or 0)/TAU_N:.4f}",
        f"  t / tau_weak         = {(pred_c.t_escape_s or 0)/TAU_WEAK_SM_EST:.4f}",
        "",
    ]
    return "\n".join(lines)


def bounce_period(m: float = M_E) -> float:
    """T_bounce = 4L/c with L = lambda_C/2 (C1 geometry)."""
    return 4.0 * coin_half_width(m) / C


def report_coin_geometry() -> str:
    l = coin_half_width()
    lines = [
        "AETHOS coin geometry (Sec 2, C1 mandate)",
        "=" * 56,
        f"L = lambda_C/2     = {l:.6e} m",
        f"lambda_C           = {compton_wavelength():.6e} m",
        f"omega_b            = {omega_bounce():.6e} rad/s",
        f"f_b                = {f_bounce():.6e} Hz",
        f"k_s (geometry)     = {k_s_from_geometry():.6e} N/m",
        f"Delta(kappa=1)     = {delta_kappa(1.0):.6e} rad/s",
        f"Omega(kappa=0)     = {omega_hopping(0.0):.6e} rad/s",
        f"Pi_pin(kappa=1)    = {pi_pin_kappa(1.0):.6f}",
        f"U_max (kappa=1)    = {joules_to_ev(u_max_spring()):.6f} MeV",
        "",
    ]
    return "\n".join(lines)


# --- 3D complex plane: SpacetimeCell (C x R + n rail) ---

from aethos_complex_plane import (
    ComplexPlane3D,
    depth_at,
    equalize_witness,
    imaginary_start,
    segment_at,
    triple_equalization,
    wing_transform,
)
from aethos_lattice import BranchKind


class SpacetimeIntervalKind(Enum):
    """Sign of I^2 = (c dn)^2 - |dz|^2 - d_zeta^2 (lattice units, MODEL)."""

    TIMELIKE = "timelike"
    LIGHTLIKE = "lightlike"
    SPACELIKE = "spacelike"


@dataclass(frozen=True)
class SpacetimeCell:
    """
    One state on the 3D complex plane: spring (z), depth (zeta), rail (n).

    Native geometry lives in C x R; Cartesian (x, y, z) is a projection.
    Interval (restated SR analog, dimensionless unless c is tied to SI):

        I^2 = (c * dn)^2 - |dz|^2 - d_zeta^2

    Layer 0 (|chain| = 0): z = n + ni, zeta = n  =>  |dz|^2 + d_zeta^2 = 3 dn^2
    so a lightlike step from the origin uses c = sqrt(3) in lattice units.

    Interior plateau (k >= 3): zeta = sum(chain) while z still moves with n.
    """

    z: complex
    zeta: float
    n: float
    chain: tuple[float, ...] = ()
    branch: BranchKind | None = None
    wing: int | None = None

    @classmethod
    def from_psi(
        cls,
        psi: ComplexPlane3D,
        n: float,
        *,
        chain: Sequence[int | float] = (),
        branch: BranchKind | None = None,
        wing: int | None = None,
    ) -> SpacetimeCell:
        return cls(
            z=psi.z,
            zeta=psi.zeta,
            n=float(n),
            chain=tuple(chain),
            branch=branch,
            wing=wing,
        )

    @classmethod
    def layer0(cls, n: float) -> SpacetimeCell:
        """Imaginary-axis start: z = n + ni, zeta = n."""
        return cls.from_psi(imaginary_start(n), n)

    @classmethod
    def at(
        cls,
        chain: Sequence[int | float],
        n: float,
        branch: BranchKind = BranchKind.VA1,
        wing: int = 1,
        *,
        lock_interior: bool = True,
    ) -> SpacetimeCell:
        """Full lattice address (A, b, w, n) -> SpacetimeCell."""
        psi = wing_transform(branch, chain, n, wing, lock_interior=lock_interior)
        return cls.from_psi(
            psi,
            n,
            chain=tuple(chain),
            branch=branch,
            wing=wing,
        )

    @classmethod
    def promote_witness(
        cls,
        full_chain: Sequence[int | float],
        subset: Sequence[int | float],
        branch: BranchKind = BranchKind.VA1,
        wing: int = 1,
    ) -> SpacetimeCell:
        """Missing-variable meet: transgress subset until n = missing anchor."""
        n_w, psi = equalize_witness(full_chain, subset, branch, wing)
        return cls.from_psi(
            psi,
            n_w,
            chain=tuple(subset),
            branch=branch,
            wing=wing,
        )

    @property
    def psi(self) -> ComplexPlane3D:
        return ComplexPlane3D(z=self.z, zeta=self.zeta)

    @property
    def re(self) -> float:
        return self.z.real

    @property
    def im(self) -> float:
        return self.z.imag

    @property
    def modulus_squared(self) -> float:
        return abs(self.z) ** 2

    @property
    def spring_depth_sum_sq(self) -> float:
        """|z|^2 + zeta^2 — local cell measure (not globally conserved on rail)."""
        return self.modulus_squared + self.zeta**2

    def segment_index(self) -> int:
        if not self.chain:
            return 0
        return segment_at(self.chain, self.n)

    def is_interior_plateau(self, *, lock_interior: bool = True) -> bool:
        """True when depth is locked at sum(chain) but n is still interior."""
        k = len(self.chain)
        if not lock_interior or k <= 2:
            return False
        seg = self.segment_index()
        return 0 < seg < k

    def expected_plateau_zeta(self, *, lock_interior: bool = True) -> float | None:
        if not self.is_interior_plateau(lock_interior=lock_interior):
            return None
        return depth_at(self.chain, self.n, lock_interior=lock_interior)

    def interval_squared_to(
        self,
        other: SpacetimeCell,
        *,
        c: float = 1.0,
    ) -> float:
        dn = other.n - self.n
        dz = other.z - self.z
        dzeta = other.zeta - self.zeta
        return (c * dn) ** 2 - abs(dz) ** 2 - dzeta**2

    def interval_kind_to(
        self,
        other: SpacetimeCell,
        *,
        c: float = 1.0,
        tol: float = 1e-9,
    ) -> SpacetimeIntervalKind:
        i2 = self.interval_squared_to(other, c=c)
        if abs(i2) <= tol:
            return SpacetimeIntervalKind.LIGHTLIKE
        if i2 > 0:
            return SpacetimeIntervalKind.TIMELIKE
        return SpacetimeIntervalKind.SPACELIKE

    def is_lightlike_to(
        self,
        other: SpacetimeCell,
        *,
        c: float = 1.0,
        tol: float = 1e-9,
    ) -> bool:
        return self.interval_kind_to(other, c=c, tol=tol) == SpacetimeIntervalKind.LIGHTLIKE

    def branch_pair_sum(self, conjugate_branch: SpacetimeCell) -> complex:
        """VA1 + VA2 style sum — Im cancels when branches are Y-mirror."""
        return self.z + conjugate_branch.z


def anchor_crossing_displacement(anchor: float) -> complex:
    """Displacement from spoke rest O_a at trigger n = a: dz = a + ai."""
    return complex(anchor, anchor)


def anchor_crossing_modulus_squared(anchor: float) -> float:
    """|dz|^2 = 2 a^2 at anchor crossing (Pythagorean factor 2)."""
    return 2.0 * anchor**2


def layer0_lightlike_c() -> float:
    """c such that origin -> layer0(n) step is lightlike for any n > 0."""
    return math.sqrt(3.0)


def triple_meet_cells(
    a: float,
    p: float,
    q: float,
    branch: BranchKind = BranchKind.VA1,
    wing: int = 1,
) -> dict[str, SpacetimeCell]:
    """All three 2-way promotion witnesses as SpacetimeCell objects."""
    raw = triple_equalization(a, p, q, branch, wing)
    labels = {"ap": (a, p), "aq": (a, q), "pq": (p, q)}
    return {
        label: SpacetimeCell.from_psi(
            psi,
            n_w,
            chain=labels[label],
            branch=branch,
            wing=wing,
        )
        for label, (n_w, psi) in raw.items()
    }


def report_spacetime_geometry() -> str:
    """Sample 3D complex plane cells for physics cross-reference."""
    c0 = layer0_lightlike_c()
    o = SpacetimeCell.layer0(0.0)
    one = SpacetimeCell.layer0(1.0)
    triple = triple_meet_cells(3, 5, 7)
    ref = next(iter(triple.values()))
    plateau = SpacetimeCell.at((3, 5, 7, 11), 5)
    lines = [
        "AETHOS 3D complex plane — SpacetimeCell (C x R + n)",
        "=" * 56,
        f"Layer0 lightlike c     = sqrt(3) = {c0:.6f} (lattice units)",
        f"|z|^2 @ n=1 layer0     = {one.modulus_squared:.0f}  (= 2 n^2)",
        f"|dz|^2 @ anchor p=5     = {anchor_crossing_modulus_squared(5):.0f}  (= 2 p^2)",
        f"0 -> 1 interval kind    = {o.interval_kind_to(one, c=c0).value}",
        "",
        "Triple meet (3,5,7) — all paths same (z, zeta):",
        f"  z = {ref.z.real:.0f}{ref.z.imag:+.0f}i  zeta = {ref.zeta:.0f}",
        f"  witnesses n = ap:{triple['ap'].n:.0f} aq:{triple['aq'].n:.0f} pq:{triple['pq'].n:.0f}",
        "",
        "Interior plateau (3,5,7,11) @ n=5:",
        f"  is_plateau = {plateau.is_interior_plateau()}",
        f"  zeta = {plateau.zeta:.0f}  (locked = {plateau.expected_plateau_zeta():.0f})",
        f"  z = {plateau.z.real:.0f}{plateau.z.imag:+.0f}i",
        "",
    ]
    va1 = SpacetimeCell.at((3, 5, 7), 5, BranchKind.VA1)
    va2 = SpacetimeCell.at((3, 5, 7), 5, BranchKind.VA2)
    pair = va1.branch_pair_sum(va2)
    lines.extend(
        [
            "Branch pair @ (3,5,7) n=5:",
            f"  VA1 + VA2 = {pair.real:.0f}{pair.imag:+.0f}i  (Im cancel -> real observable)",
            "",
        ]
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class MeasurementCollapse:
    """
    Strong measurement on a SpacetimeCell (Sec 5 / P7-2 bridge).

    Pre: full spring (z, zeta) on rail n.
    Post: Im(z) suppressed ~ exp(-Lambda_n); zeta pinned when pin_p -> 1;
          real axis from VA1+VA2 branch pair when chain is available.
    """

    pre: SpacetimeCell
    post: SpacetimeCell
    lambda_n: float
    pin_p: float
    regime: str  # "soft" | "hard" from classify_compression thresholds


def compress_spacetime_cell(
    cell: SpacetimeCell,
    *,
    lambda_n: float = 5.0,
    dephase_hard: float = 0.05,
) -> MeasurementCollapse:
    """
    Lattice-side measurement: collapse Im(z), compress zeta along observation axis.

    Im suppression uses Kraus factor exp(-Lambda_n).
    Hard regime when exp(-Lambda_n) <= dephase_hard (default 5% residual phase).
    When chain metadata is present, the pinned Re(z) comes from VA1+VA2 sum.
    """
    dephase = kraus_decoherence_factor(lambda_n)
    pin = measurement_pin_probability(lambda_n)
    regime = "hard" if dephase <= dephase_hard else "soft"

    if cell.chain:
        wing = cell.wing if cell.wing is not None else 1
        va1 = SpacetimeCell.at(cell.chain, cell.n, BranchKind.VA1, wing)
        va2 = SpacetimeCell.at(cell.chain, cell.n, BranchKind.VA2, wing)
        z_paired = va1.branch_pair_sum(va2)
        z_real = z_paired.real
        z_im = cell.im * dephase
        z_collapsed = complex(z_real, z_im)
    else:
        z_collapsed = complex(cell.re, cell.im * dephase)

    if regime == "hard":
        zeta_collapsed = cell.zeta
    else:
        zeta_collapsed = cell.zeta * (1.0 - 0.5 * pin)

    post = SpacetimeCell.from_psi(
        ComplexPlane3D(z=z_collapsed, zeta=zeta_collapsed),
        cell.n,
        chain=cell.chain,
        branch=cell.branch,
        wing=cell.wing,
    )
    return MeasurementCollapse(
        pre=cell,
        post=post,
        lambda_n=lambda_n,
        pin_p=pin,
        regime=regime,
    )


def apply_sg_collapse(
    cell: SpacetimeCell,
    *,
    b_grad_z: float = 1.0e3,
    l_mag: float = 0.04,
    beam_speed: float = 1.0e5,
    target_lambda: float = 5.0,
) -> MeasurementCollapse:
    """Full Sec 5 SG calibration -> lattice collapse on cell."""
    cal = calibrate_measurement_sg(
        b_grad_z=b_grad_z,
        l_mag=l_mag,
        beam_speed=beam_speed,
        target_lambda=target_lambda,
    )
    return compress_spacetime_cell(cell, lambda_n=cal.lambda_n)


def report_spacetime_wiring() -> str:
    """End-to-end SpacetimeCell wiring: collapse, entanglement, attractors."""
    from aethos_attractor_index import CorpusAttractorIndex, cell_attractor_key
    from aethos_intersection_nodes import IntersectionNetwork

    cell = SpacetimeCell.at((3, 5, 7), 5, BranchKind.VA1)
    col = compress_spacetime_cell(cell, lambda_n=10.0)
    net = IntersectionNetwork()
    net.follow_and_branch([3, 5, 7, 15], max_nodes=32)
    pairs = net.entangled_pairs()
    triple = SpacetimeCell.promote_witness((3, 5, 7), (3, 5))
    idx = CorpusAttractorIndex()
    idx.add("demo", cell_attractor_key(triple), "triple")
    lines = [
        "AETHOS SpacetimeCell wiring",
        "=" * 56,
        f"Measurement collapse: z {col.pre.z} -> {col.post.z}  regime={col.regime}",
        f"Entangled meet pairs (network): {len(pairs)}",
        f"Attractor buckets: {idx.summary()['buckets']}",
        "",
    ]
    return "\n".join(lines)


# --- Section 5: measurement / compression channel (Step 5 gate) ---


def e_obs_coupling_from_gradient(b_grad_z: float, l: float | None = None) -> float:
    """Micro coupling energy scale mu_B |dB/dz| L_0 (J) — C1 coin width."""
    length = l if l is not None else coin_half_width()
    return MU_B * abs(b_grad_z) * length


def e_obs_frequency_scale(b_grad_z: float, l: float | None = None) -> float:
    """omega_eff ~ mu_B |dB/dz| L / hbar [rad/s] for H_coin axis dominance."""
    length = l if l is not None else coin_half_width()
    return MU_B * abs(b_grad_z) * length / HBAR


def strong_measurement_ratio(
    b_grad_z: float,
    g_e: float = 1.0,
    kappa: float = 0.0,
    l: float | None = None,
) -> float:
    """|g_E E_obs| / Omega(kappa); >> 1 => compression axis dominates H_coin."""
    omega_eff = g_e * e_obs_frequency_scale(b_grad_z, l)
    o = omega_hopping(kappa, l=l)
    return omega_eff / o if o > 0 else float("inf")


def lambda_n_from_coin_gradient(
    b_grad_z: float,
    tau_m: float,
    g_e: float = 1.0,
    l: float | None = None,
) -> float:
    """Lambda_n with E_obs tied to C1 L_0 (geometry-first SG coupling)."""
    e_j = HBAR * e_obs_frequency_scale(b_grad_z, l)
    return lambda_n_integrated(g_e, e_j, tau_m)


def kraus_decoherence_factor(lambda_n: float) -> float:
    """Off-diagonal suppression ~ exp(-Lambda_n) in symmetric dephasing channel."""
    return math.exp(-lambda_n)


def measurement_tau_window(l_mag: float, beam_speed: float) -> float:
    """Interaction window tau_m = L_mag / v_beam."""
    if beam_speed <= 0:
        raise ValueError("beam_speed must be positive")
    return l_mag / beam_speed


# --- Section 1: P1-v vapor spectrum / full packet ---


def mode_energy_joules(f_hz: float) -> float:
    """E = h f for one vapor quantum (mode)."""
    return H * f_hz


def mode_energy_ev_photon(f_hz: float) -> float:
    """True eV (not MeV-scale joules_to_ev used in nuclear calibration)."""
    return mode_energy_joules(f_hz) / E_CHARGE


def mode_energy_ev(f_hz: float) -> float:
    """Alias for photon-line eV."""
    return mode_energy_ev_photon(f_hz)


def spectral_fill(
    amplitudes_squared: dict[float, float],
    f_min: float,
    f_max: float,
    *,
    reference: dict[float, float] | None = None,
) -> float:
    """
    phi_B = integral |a(f)|^2 df / integral |a_max(f)|^2 df on band [f_min, f_max].
    amplitudes_squared: map frequency_hz -> |a|^2 (discrete samples OK).
    """
    if f_min >= f_max:
        raise ValueError("f_min must be < f_max")
    band = {f: w for f, w in amplitudes_squared.items() if f_min <= f <= f_max}
    if not band:
        return 0.0
    ref = reference if reference is not None else band
    num = sum(band.values())
    den = sum(ref.values())
    return num / den if den > 0 else 0.0


def classify_spectrum(
    amplitudes_squared: dict[float, float],
    *,
    monochromatic_frac: float = 0.9,
) -> str:
    """Rough label: monochromatic vs broadband from peak fraction."""
    total = sum(amplitudes_squared.values())
    if total <= 0:
        return "empty"
    peak = max(amplitudes_squared.values())
    if peak / total >= monochromatic_frac:
        return "fundamental_packet"
    return "partial_or_full_spectrum"


def visible_band_hz() -> tuple[float, float]:
    """Order-of-magnitude visible band for phi_B demos (~400–750 THz)."""
    return 4.0e14, 7.5e14


def demo_white_vs_laser_spectra() -> tuple[dict[float, float], dict[float, float]]:
    """Discrete demo: narrow laser line vs flat-ish white band."""
    laser = {5.0e14: 1.0}
    white: dict[float, float] = {}
    f_lo, f_hi = visible_band_hz()
    step = (f_hi - f_lo) / 8
    f = f_lo
    while f <= f_hi:
        white[f] = 1.0
        f += step
    return laser, white


def report_vapor_spectrum() -> str:
    laser, white = demo_white_vs_laser_spectra()
    f_lo, f_hi = visible_band_hz()
    ref = white
    lines = [
        "AETHOS vapor spectrum (Sec 1, P1-v)",
        "=" * 56,
        f"One quantum E at 500 THz = {mode_energy_ev_photon(5e14):.4f} eV",
        f"Laser-like: {classify_spectrum(laser)}",
        f"White-band demo: {classify_spectrum(white)}",
        f"phi_B (white, ref=full band) = {spectral_fill(white, f_lo, f_hi, reference=ref):.4f}",
        f"phi_B (laser, ref=full band) = {spectral_fill(laser, f_lo, f_hi, reference=ref):.4f}",
        "",
    ]
    return "\n".join(lines)


def spin_projection_probability(theta_rad: float, prep: str = "w") -> float:
    """P(spin up along axis tilted by theta from |W> prep). prep: 'w' | 'mixed'."""
    if prep == "w":
        return math.cos(theta_rad / 2) ** 2
    if prep == "mixed":
        return 0.5
    raise ValueError(f"unknown prep: {prep!r}")


def e_obs_from_sg(b_grad_z: float, l_mag: float, mu: float = MU_B) -> float:
    """Effective coupling scale mu * |dB/dz| * L (J), Stern-Gerlach order-of-magnitude."""
    return mu * abs(b_grad_z) * l_mag


def gamma_n_rate(g_e: float, e_obs: float) -> float:
    """Dephasing rate scale Gamma_n propto (g_E E_obs)^2 (reduced units; see calibration_sheet)."""
    return (g_e * e_obs) ** 2


def lambda_n_integrated(g_e: float, e_obs: float, tau_m: float) -> float:
    """Lambda_n = 2 integral Gamma_n dt with constant Gamma_n over window tau_m."""
    return 2.0 * gamma_n_rate(g_e, e_obs) * tau_m


def lambda_n_from_sg(
    b_grad_z: float,
    l_mag: float,
    tau_m: float,
    g_e: float = 1.0,
) -> float:
    """Channel strength from SG-like gradient, magnet length, and interaction time."""
    return lambda_n_integrated(g_e, e_obs_from_sg(b_grad_z, l_mag), tau_m)


def measurement_pin_probability(lambda_n: float) -> float:
    """Axis pin strength p ~ (1 - exp(-Lambda))/2 for symmetric dephasing channel."""
    return 0.5 * (1.0 - math.exp(-lambda_n))


def calibrate_g_e_for_lambda(
    target_lambda: float,
    b_grad_z: float,
    l_mag: float,
    tau_m: float,
) -> float:
    """Solve g_E so lambda_n_from_sg equals target_lambda (e.g. 5 for strong pin)."""
    e_ref = e_obs_from_sg(b_grad_z, l_mag)
    if e_ref <= 0 or tau_m <= 0:
        raise ValueError("e_obs and tau_m must be positive")
    # target = 2 (g_E e_ref)^2 tau_m  =>  g_E = sqrt(target / (2 tau_m)) / e_ref
    return math.sqrt(target_lambda / (2.0 * tau_m)) / e_ref


@dataclass(frozen=True)
class MeasurementCalibration:
    b_grad_z: float
    l_mag: float
    tau_m: float
    g_e: float
    e_obs: float
    lambda_n: float
    pin_p: float


def calibrate_measurement_sg(
    *,
    b_grad_z: float = 1.0e3,
    l_mag: float = 0.04,
    beam_speed: float = 1.0e5,
    target_lambda: float = 5.0,
) -> MeasurementCalibration:
    """
    Reference textbook SG: strong gradient, ~4 cm magnet, thermal-ish beam speed.
    target_lambda >= 5 => projective limit (off-diagonals ~ exp(-Lambda)).
    """
    tau_m = l_mag / beam_speed
    g_e = calibrate_g_e_for_lambda(target_lambda, b_grad_z, l_mag, tau_m)
    e_obs = e_obs_from_sg(b_grad_z, l_mag)
    lam = lambda_n_integrated(g_e, e_obs, tau_m)
    return MeasurementCalibration(
        b_grad_z=b_grad_z,
        l_mag=l_mag,
        tau_m=tau_m,
        g_e=g_e,
        e_obs=e_obs,
        lambda_n=lam,
        pin_p=measurement_pin_probability(lam),
    )


def bell_correlation_qm(angle_a_rad: float, angle_b_rad: float) -> float:
    """Singlet-like kernel E(a,b) = -cos(a - b)."""
    return -math.cos(angle_a_rad - angle_b_rad)


def bell_correlation_coin_geometry(
    angle_a_rad: float,
    angle_b_rad: float,
    *,
    n_samples: int = 50_000,
) -> float:
    """
    O5-3 candidate (external Block 5): opposite-phase coins, binary outcome =
    sign(cos(compression_angle - theta)). Uniform theta on [0, 2pi).

    NOTE: This sign rule does NOT reproduce E(a,b)=-cos(a-b) at general angles
    (e.g. a=0, b=pi/4 gives ~-0.5, not -sqrt(2)/2). Use as falsifier until
    a corrected geometry map is derived. See section_06_derivations.md 6.12.
    """
    if n_samples < 100:
        raise ValueError("n_samples too small")
    total = 0.0
    for i in range(n_samples):
        theta = 2.0 * math.pi * i / n_samples
        ra = 1.0 if math.cos(angle_a_rad - theta) >= 0.0 else -1.0
        rb = 1.0 if math.cos(angle_b_rad - theta + math.pi) >= 0.0 else -1.0
        total += ra * rb
    return total / n_samples


def bell_correlation_joint_ripple_linear(
    angle_a_rad: float,
    angle_b_rad: float,
) -> float:
    """
    C5 partial (O5-3): shared joint-phase projection on DM ripple.

    Equivalent to bell_correlation_phi_fill with phi_AB=0.5 (Stage B half-fill).
    """
    return bell_correlation_phi_fill(angle_a_rad, angle_b_rad, 0.5)


def chsh_s_quantum(
    a: float = 0.0,
    a_prime: float = math.pi / 2,
    b: float = math.pi / 4,
    b_prime: float = 3 * math.pi / 4,
) -> float:
    """CHSH S = E(a,b) - E(a,b') + E(a',b) + E(a',b'). Quantum max |S| = 2*sqrt(2)."""
    return (
        bell_correlation_qm(a, b)
        - bell_correlation_qm(a, b_prime)
        + bell_correlation_qm(a_prime, b)
        + bell_correlation_qm(a_prime, b_prime)
    )


# --- Section 6: entanglement dynamics (Step 6 gate, C7 Stage A) ---


def ell_c_from_geometry(m: float = M_E) -> float:
    """Stage A (C7): coherence length = Compton span lambda_C = 2 L_0."""
    return compton_wavelength(m)


def sigma_obs_geometry(
    kappa: float = 0.0,
    *,
    eta_geom: float = 1.0,
    s_res: float = 1.0,
    l: float | None = None,
) -> float:
    """Observation cross-section A_eff eta S_res with A_eff = pi L_0^2 (C1)."""
    length = l if l is not None else coin_half_width()
    a_eff = math.pi * length**2
    return a_eff * max(0.0, min(1.0, eta_geom)) * max(0.0, s_res)


def j_ab_overlap(
    separation_m: float,
    ell_c: float | None = None,
    *,
    s_freq: float = 1.0,
    s_axis: float = 1.0,
) -> float:
    """J_AB = exp(-d/ell_c) S_freq S_axis (Sec 6.5.2)."""
    lc = ell_c if ell_c is not None else ell_c_from_geometry()
    if lc <= 0:
        return 0.0
    return math.exp(-separation_m / lc) * s_freq * s_axis


def k_lock_from_geometry(m: float = M_E, l: float | None = None) -> float:
    """Natural lock attempt rate ~ bounce frequency f_b (GEOMETRY)."""
    return f_bounce(m, l)


def gamma_form_rate(
    separation_m: float,
    *,
    order_ab: float = 1.0,
    phi_ab: float = 1.0,
    ell_c: float | None = None,
    k_lock: float | None = None,
    s_freq: float = 1.0,
    s_axis: float = 1.0,
) -> float:
    """
    Gamma_form = k_lock |O_AB| J_AB phi_AB.
    Stage A: ell_c from geometry; Stage B: phi_AB from fill dynamics.
    """
    phi = max(0.0, min(1.0, phi_ab))
    kl = k_lock if k_lock is not None else k_lock_from_geometry()
    j = j_ab_overlap(separation_m, ell_c, s_freq=s_freq, s_axis=s_axis)
    return kl * abs(order_ab) * j * phi


def gamma_break_rate(
    phi_env: float,
    *,
    kappa: float = 0.0,
    gamma_other: float = 0.0,
    eta_geom: float = 1.0,
    s_res: float = 1.0,
    l: float | None = None,
) -> float:
    """Gamma_break ~ Phi_env sigma_obs + Gamma_other (Sec 6.5.3)."""
    sigma = sigma_obs_geometry(kappa, eta_geom=eta_geom, s_res=s_res, l=l)
    return max(0.0, phi_env) * sigma + max(0.0, gamma_other)


def coherence_steady_state(gamma_form: float, gamma_break: float) -> float:
    """C_* = Gamma_form / (Gamma_form + Gamma_break)."""
    g = max(0.0, gamma_form)
    b = max(0.0, gamma_break)
    den = g + b
    return g / den if den > 0 else 0.0


def coherence_at_time(
    t: float,
    *,
    c0: float = 0.0,
    gamma_form: float,
    gamma_break: float,
) -> float:
    """Closed-form C(t) for constant rates (Sec 6.3.2)."""
    g = max(0.0, gamma_form)
    b = max(0.0, gamma_break)
    c_star = coherence_steady_state(g, b)
    rate = g + b
    return c_star + (c0 - c_star) * math.exp(-rate * max(0.0, t))


def entanglement_lifetime(gamma_form: float, gamma_break: float) -> float:
    """tau_E ~ 1 / (Gamma_form + Gamma_break)."""
    s = max(0.0, gamma_form) + max(0.0, gamma_break)
    return 1.0 / s if s > 0 else float("inf")


def phi_ab_derivative(
    phi_ab: float,
    *,
    gamma_fill: float,
    gamma_snap: float,
    eta_obs: float = 0.0,
    gamma_break: float = 0.0,
) -> float:
    """d phi_AB / dt = Gamma_fill(1-phi) - Gamma_snap phi - eta_obs Gamma_break."""
    phi = max(0.0, min(1.0, phi_ab))
    return gamma_fill * (1.0 - phi) - gamma_snap * phi - eta_obs * max(0.0, gamma_break)


def bell_correlation_phi_fill(
    angle_a_rad: float,
    angle_b_rad: float,
    phi_ab: float,
) -> float:
    """
    Stage B (C7) contract: E = -phi_AB cos(a-b).
    phi_AB=1 => QM kernel; phi_AB=0.5 => Step 5 half-scale partial.
    """
    phi = max(0.0, min(1.0, phi_ab))
    return -phi * math.cos(angle_a_rad - angle_b_rad)


def report_entanglement_geometry() -> str:
    ell = ell_c_from_geometry()
    gf = gamma_form_rate(0.0, phi_ab=1.0)
    gb = gamma_break_rate(1.0e20, kappa=0.0)
    c_star = coherence_steady_state(gf, gb)
    a0, b0 = 0.0, math.pi / 4
    lines = [
        "AETHOS entanglement geometry (Sec 6, Step 6 gate, C7 Stage A)",
        "=" * 56,
        f"ell_c^geom (Stage A)   = {ell:.6e} m  (= lambda_C)",
        f"L_0                    = {coin_half_width():.6e} m",
        f"sigma_obs (kappa=0)    = {sigma_obs_geometry(0.0):.6e} m^2",
        f"k_lock^geom ~ f_b      = {k_lock_from_geometry():.6e} Hz",
        f"Gamma_form (d=0,phi=1) = {gf:.6e} s^-1",
        f"Gamma_break (Phi~1e20) = {gb:.6e} s^-1",
        f"C_* (demo)             = {c_star:.6f}",
        "",
        "Bell / O5-3 via phi_AB (Stage B contract):",
        f"  E QM                 = {bell_correlation_qm(a0, b0):.6f}",
        f"  E @ phi=0.5          = {bell_correlation_phi_fill(a0, b0, 0.5):.6f}",
        f"  E @ phi=1.0          = {bell_correlation_phi_fill(a0, b0, 1.0):.6f}",
        "",
        "Stage B OPEN: derive phi_AB from fill ODE + DM mesh (P11-3).",
        "",
    ]
    return "\n".join(lines)


# --- Section 7: P7-2 partial shred vs full flatten ---


class CompressionRegime(Enum):
    SOFT = "soft_shred_tunnel"
    HARD = "hard_flatten_collapse"


def pi_pin_from_bias(delta_eff: float, omega: float) -> float:
    """Pinning indicator |Delta_eff|/(|Delta_eff|+Omega), Sec 4/7."""
    d = abs(delta_eff)
    o = abs(omega)
    return d / (d + o) if (d + o) > 0 else 0.0


def classify_compression(
    pi_pin: float,
    *,
    p: float = 0.0,
    p_c: float = 1.0,
    pi_hard: float = 0.95,
) -> CompressionRegime:
    """
    P7-2: soft shred (tunnel + recapture) vs hard flatten (escape/collapse).
    p, p_c in same units as neutron pressure model (Sec 4); optional.
    """
    if pi_pin >= pi_hard or (p_c > 0 and p >= p_c):
        return CompressionRegime.HARD
    return CompressionRegime.SOFT


def xi_shred_from_field(e_bar: float, e_ref: float) -> float:
    """Sec 7.2.2 shredding fraction."""
    num = abs(e_bar)
    den = num + abs(e_ref)
    return num / den if den > 0 else 0.0


def xi_shred_with_dm(phi_path: float, xi_base: float, eta_dm: float = 0.5) -> float:
    """Sec 7.3.4: filled DM path eases shredding."""
    phi_path = max(0.0, min(1.0, phi_path))
    return xi_base * (1.0 - eta_dm * phi_path)


def t_wkb(kappa_bar: float, length: float) -> float:
    return math.exp(-2.0 * kappa_bar * length)


def t_eff_soft(kappa_bar: float, length: float, chi_ss: float) -> float:
    """T_eff = T_WKB * chi_ss for soft regime (Sec 7.3.1)."""
    return t_wkb(kappa_bar, length) * max(0.0, min(1.0, chi_ss))


def chi_steady(gamma_rec: float, gamma_sh: float) -> float:
    g = gamma_rec + gamma_sh
    return gamma_rec / g if g > 0 else 0.0


# --- Section 7 geometry (Step 7 gate) ---


def u_bar_from_h_coin(kappa: float, m: float = M_E, l: float | None = None) -> float:
    """U_bar = (hbar/2) sqrt(Delta^2 + Omega^2) from Step 2 H_x (GEOMETRY)."""
    d = delta_kappa(kappa, m, l)
    o = omega_hopping(kappa, m, l)
    return 0.5 * HBAR * math.sqrt(d * d + o * o)


def e_ref_from_geometry(m: float = M_E, l: float | None = None) -> float:
    """Reference barrier field scale ~ hbar omega_b (GEOMETRY)."""
    return HBAR * omega_bounce(m, l)


def m_eff_from_shred(xi_shred: float, m: float = M_E, lambda_m: float = 0.0) -> float:
    """m_eff = m (1 + lambda_m xi_shred) — Sec 7.2.2."""
    xi = max(0.0, min(1.0, xi_shred))
    return m * (1.0 + lambda_m * xi)


def kappa_wkb_from_h_x(
    energy_j: float,
    *,
    kappa: float = 0.5,
    xi_shred: float = 0.0,
    lambda_m: float = 0.0,
    m: float = M_E,
    l: float | None = None,
) -> float:
    """kappa = sqrt(2 m_eff (U_bar - E)_+) / hbar — WKB from H_x (GEOMETRY)."""
    m_eff = m_eff_from_shred(xi_shred, m, lambda_m)
    u = u_bar_from_h_coin(kappa, m, l)
    excess = max(u - energy_j, 0.0)
    if excess <= 0:
        return 0.0
    return math.sqrt(2.0 * m_eff * excess) / HBAR


def gamma_rec_from_geometry(
    kappa: float = 0.5,
    *,
    xi_shred: float = 0.0,
    lambda_m: float = 0.0,
    m: float = M_E,
    l: float | None = None,
) -> float:
    """Gamma_rec = (k_s/m_eff)(1/2pi) Pi_pin^{-1} — Sec 7.3.1 (GEOMETRY)."""
    m_eff = m_eff_from_shred(xi_shred, m, lambda_m)
    ks = k_s_from_geometry(m, l)
    pi = pi_pin_kappa(kappa, m, l)
    if pi <= 0:
        return float("inf")
    return (ks / m_eff) * (1.0 / (2.0 * math.pi)) / pi


def kappa_bar_with_dm_path(
    kappa_bar: float,
    phi_path: float,
    *,
    eta_kappa: float = 0.5,
) -> float:
    """kappa -> kappa (1 + eta_kappa (1 - phi_path)) — unfilled path harder (Sec 7.3.4)."""
    phi = max(0.0, min(1.0, phi_path))
    return max(0.0, kappa_bar) * (1.0 + eta_kappa * (1.0 - phi))


def classify_compression_from_coin(
    kappa: float,
    *,
    p: float = 0.0,
    p_c: float = 0.0,
    pi_hard: float = 0.95,
    m: float = M_E,
    l: float | None = None,
) -> CompressionRegime:
    """P7-2 using Pi_pin from coin geometry at compression kappa."""
    pi = pi_pin_kappa(kappa, m, l)
    return classify_compression(pi, p=p, p_c=p_c, pi_hard=pi_hard)


def transmission_soft_pipeline(
    barrier_length_m: float,
    incident_energy_j: float,
    *,
    kappa: float = 0.5,
    xi_shred: float = 0.3,
    phi_path: float = 0.0,
    eta_dm: float = 0.5,
    eta_kappa: float = 0.5,
    gamma_sh: float = 1.0e6,
    lambda_m: float = 0.0,
) -> float:
    """T_eff = T_WKB(kappa_bar) * chi_ss with DM path modifiers (soft regime)."""
    xi = xi_shred_with_dm(phi_path, xi_shred, eta_dm=eta_dm)
    k_bar = kappa_wkb_from_h_x(
        incident_energy_j, kappa=kappa, xi_shred=xi, lambda_m=lambda_m
    )
    k_bar = kappa_bar_with_dm_path(k_bar, phi_path, eta_kappa=eta_kappa)
    g_rec = gamma_rec_from_geometry(kappa, xi_shred=xi, lambda_m=lambda_m)
    chi = chi_steady(g_rec, gamma_sh)
    return t_eff_soft(k_bar, barrier_length_m, chi)


def report_tunneling_geometry() -> str:
    e_ref = e_ref_from_geometry()
    e_lab = 1.0e3 * E_CHARGE  # ~1 keV order demo
    k_soft = kappa_wkb_from_h_x(e_lab, kappa=0.3, xi_shred=0.2)
    k_hard = kappa_wkb_from_h_x(e_lab, kappa=0.9, xi_shred=0.5)
    regime_low = classify_compression_from_coin(0.05)
    regime_high = classify_compression_from_coin(1.0)
    t0 = transmission_soft_pipeline(1.0e-9, e_lab, kappa=0.4, phi_path=0.0)
    t1 = transmission_soft_pipeline(1.0e-9, e_lab, kappa=0.4, phi_path=1.0, eta_dm=0.5)
    lines = [
        "AETHOS tunneling geometry (Sec 7, Step 7 gate, P7-2)",
        "=" * 56,
        f"U_bar(k=0.5)         = {joules_to_ev(u_bar_from_h_coin(0.5)):.4f} MeV",
        f"E_ref ~ hbar omega_b   = {joules_to_ev(e_ref):.4f} MeV",
        f"kappa_WKB (soft k)     = {k_soft:.3e} m^-1",
        f"kappa_WKB (high k)     = {k_hard:.3e} m^-1",
        f"P7-2 @ kappa=0.05      = {regime_low.value}",
        f"P7-2 @ kappa=1.0       = {regime_high.value}",
        "",
        "DM path (phi=0 vs phi=1, L=1 nm demo):",
        f"  T_eff phi=0          = {t0:.3e}",
        f"  T_eff phi=1          = {t1:.3e}",
        "",
        "O7-4 OPEN: eta_DM, eta_kappa from P11-3 mesh calibration.",
        "",
    ]
    return "\n".join(lines)


# --- Section 8: double-slit / interference (Step 8 gate) ---

K_B = 1.380_649e-23  # J/K
M_HE3 = 3.016_029_320_1e-27  # kg (atomic mass scale)
M_HE4 = 4.002_603_254_13e-27


def wake_kernel_xy(
    x: float,
    y: float,
    z: float,
    x_s: float,
    y_s: float,
    *,
    sigma_wake: float,
    ell_wake: float | None = None,
) -> float:
    """Geometric wake kernel K_s(r) — Sec 8.2.2 (GEOMETRY envelope)."""
    if sigma_wake <= 0 or z <= 0:
        raise ValueError("sigma_wake and z must be positive")
    dx = x - x_s
    dy = y - y_s
    r_xy2 = dx * dx + dy * dy
    r = math.sqrt(r_xy2 + z * z)
    gauss = math.exp(-r_xy2 / (2.0 * sigma_wake**2))
    damp = math.exp(-r / ell_wake) if ell_wake is not None and ell_wake > 0 else 1.0
    return gauss / r * damp


def a0_wake_scale(
    kappa: float = 0.0,
    *,
    eta_wake: float = 1.0,
    m: float = M_E,
    l: float | None = None,
) -> float:
    """A_0 = eta_wake sqrt(hbar omega_b) |Omega|/Omega_0 — Sec 8.2.2 (GEOMETRY)."""
    omega0 = 0.5 * omega_bounce(m, l)
    o = abs(omega_hopping(kappa, m, l))
    ob = omega_bounce(m, l)
    if omega0 <= 0:
        return 0.0
    return eta_wake * math.sqrt(HBAR * ob) * o / omega0


def wake_amplitude_complex(
    x: float,
    y: float,
    z: float,
    x_s: float,
    y_s: float,
    phase_rad: float,
    *,
    sigma_wake: float,
    ell_wake: float | None = None,
    kappa: float = 0.0,
    eta_wake: float = 1.0,
) -> complex:
    k = wake_kernel_xy(x, y, z, x_s, y_s, sigma_wake=sigma_wake, ell_wake=ell_wake)
    a0 = a0_wake_scale(kappa, eta_wake=eta_wake)
    return a0 * k * complex(math.cos(phase_rad), math.sin(phase_rad))


def interference_intensity(
    a_l: complex,
    a_r: complex,
    mu: float = 1.0,
) -> float:
    """I = |A_L|^2 + |A_R|^2 + 2 mu Re(A_L A_R*) — Sec 8.2."""
    mu = max(0.0, min(1.0, mu))
    cross = 2.0 * mu * (a_l * a_r.conjugate()).real
    return abs(a_l) ** 2 + abs(a_r) ** 2 + cross


def fringe_visibility(i_max: float, i_min: float) -> float:
    """V = (I_max - I_min) / (I_max + I_min)."""
    den = i_max + i_min
    if den <= 0:
        return 0.0
    return max(0.0, min(1.0, (i_max - i_min) / den))


def visibility_vs_pressure(v0: float, lambda_deco: float, pressure_pa: float) -> float:
    """V(P) = V_0 exp(-Lambda P) — Sec 8.7."""
    return v0 * math.exp(-max(0.0, lambda_deco) * max(0.0, pressure_pa))


def path_mark_visibility(v0: float, mark_strength: float) -> float:
    """Which-path detector suppresses cross term: V ~ V_0 (1-p)."""
    p = max(0.0, min(1.0, mark_strength))
    return v0 * (1.0 - p)


def coherence_mu_steady(gamma_form: float, gamma_break: float) -> float:
    """Steady fringe coherence mu = C_* from Sec 6 / 8.3."""
    return coherence_steady_state(gamma_form, gamma_break)


def thermal_speed(m: float, temperature_k: float = 300.0) -> float:
    """Mean thermal speed sqrt(8 k_B T / (pi m))."""
    if m <= 0 or temperature_k <= 0:
        return 0.0
    return math.sqrt(8.0 * K_B * temperature_k / (math.pi * m))


def gamma_partner_rate(
    number_density: float,
    sigma_eff: float,
    v_bar: float,
    f_coin: float = 1.0,
) -> float:
    """Gamma_partner = sum n_i sigma_i v_i f_coin,i — Sec 8.7.1."""
    return max(0.0, number_density) * max(0.0, sigma_eff) * max(0.0, v_bar) * max(0.0, f_coin)


def lambda_decoherence_proxy(
    m_gas: float,
    f_coin: float,
    *,
    sigma_eff: float = 1.0e-20,
    temperature_k: float = 300.0,
) -> float:
    """
    Relative slope proxy for Lambda in V = V_0 exp(-Lambda P).
    Proportional to sigma v f / sqrt(m) at matched (T, apparatus).
    """
    v = thermal_speed(m_gas, temperature_k)
    if m_gas <= 0:
        return 0.0
    return sigma_eff * v * f_coin / math.sqrt(m_gas)


def lambda_he3_he4_ratio(
    *,
    f_coin_he3: float = 0.75,
    f_coin_he4: float = 0.15,
    sigma_ratio: float = 1.0,
    temperature_k: float = 300.0,
) -> float:
    """Predict Lambda_3He / Lambda_4He != 1 — Sec 8.7.3 discriminator."""
    l3 = lambda_decoherence_proxy(M_HE3, f_coin_he3, sigma_eff=sigma_ratio, temperature_k=temperature_k)
    l4 = lambda_decoherence_proxy(M_HE4, f_coin_he4, sigma_eff=1.0, temperature_k=temperature_k)
    return l3 / l4 if l4 > 0 else float("inf")


def lambda_he3_he4_ratio_calibrated(**kwargs: float) -> float:
    """Sec 8.7.3 E-check: f_coin pair yielding ~7.5% Lambda ratio (not placeholder 0.75/0.15)."""
    kw = {
        "f_coin_he3": F_COIN_HE3_DISCRIMINATOR,
        "f_coin_he4": F_COIN_HE4_DISCRIMINATOR,
        "sigma_ratio": 1.0,
        "temperature_k": 300.0,
    }
    kw.update(kwargs)
    return lambda_he3_he4_ratio(**kw)


def sigma_wake_default(slit_separation: float | None = None, *, micro: bool = False) -> float:
    """Apparatus scale ~ slit/4; micro scale = L_0 (C1)."""
    if micro:
        return coin_half_width()
    if slit_separation is not None and slit_separation > 0:
        return max(slit_separation / 4.0, 10.0 * coin_half_width())
    return coin_half_width()


def demo_slit_fringe_intensity(
    y: float,
    z: float,
    slit_separation: float,
    *,
    x: float | None = None,
    sigma_wake: float | None = None,
    mu: float = 1.0,
    opposite_phase: bool = True,
) -> float:
    """Balanced slits; default opposite-phase lock phi_R = phi_L + pi."""
    sig = sigma_wake if sigma_wake is not None else sigma_wake_default(slit_separation)
    half = slit_separation / 2.0
    x_det = x if x is not None else slit_separation * 1.0e-4
    phi_r = math.pi if opposite_phase else 0.0
    a_l = wake_amplitude_complex(x_det, y, z, -half, 0.0, 0.0, sigma_wake=sig)
    a_r = wake_amplitude_complex(x_det, y, z, half, 0.0, phi_r, sigma_wake=sig)
    return interference_intensity(a_l, a_r, mu)


def report_double_slit_geometry() -> str:
    z = 0.1
    sep = 2.0e-6
    sig = sigma_wake_default(sep)
    i_center = demo_slit_fringe_intensity(0.0, z, sep, sigma_wake=sig, mu=1.0)
    i_off = demo_slit_fringe_intensity(sep / 4.0, z, sep, sigma_wake=sig, mu=1.0)
    v_full = fringe_visibility(max(i_center, i_off), min(i_center, i_off))
    v_mark = path_mark_visibility(1.0, 0.95)
    ratio = lambda_he3_he4_ratio()
    l3 = lambda_decoherence_proxy(M_HE3, 0.75)
    l4 = lambda_decoherence_proxy(M_HE4, 0.15)
    lines = [
        "AETHOS double-slit geometry (Sec 8, Step 8 gate)",
        "=" * 56,
        f"A_0 wake scale         = {a0_wake_scale():.3e}",
        f"sigma_wake (apparatus) = {sig:.3e} m  (micro L_0 = {coin_half_width():.3e})",
        f"fringe I (mu=1 demo)   = center {i_center:.3e}, off {i_off:.3e}",
        f"visibility (demo)      = {v_full:.4f}",
        f"V after path mark p=0.95 = {v_mark:.4f}",
        "",
        "Gas discriminator (Lambda proxy @ 300 K):",
        f"  Lambda_3He proxy     = {l3:.3e}",
        f"  Lambda_4He proxy     = {l4:.3e}",
        f"  Lambda_3He/Lambda_4He = {ratio:.3f}  (expect != 1)",
        "",
        "O8-1 OPEN: sigma_e,i, eta_spin calibration from coin geometry.",
        "",
    ]
    return "\n".join(lines)


# --- Section 9: atom / bonds (Step 9 gate) ---

RYDBERG_EV = 13.605_693_122_994
H2_BOND_EV = 4.52  # eV dissociation anchor
H2_R0_M = 0.74e-10  # m equilibrium separation
EPS0 = 8.854_187_8128e-12  # F/m
ALPHA_FS = 7.297_352_5693e-3


def ev_from_joules(j: float) -> float:
    """True eV (not MeV-scale joules_to_ev)."""
    return j / E_CHARGE


def bohr_radius_geometry() -> float:
    """a_0 = hbar / (m_e c alpha) — ANCHORED; compare to L_0 in audit."""
    return HBAR / (M_E * C * ALPHA_FS)


def l_bounce_geometry(m: float = M_E, l: float | None = None) -> float:
    """Full bounce path 4 L_0 (C1)."""
    length = l if l is not None else coin_half_width(m)
    return 4.0 * length


def shell_capacity(n: int) -> int:
    """N_max(n) = 2 n^2 — PROVEN."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return 2 * n * n


def subshell_capacity(l: int) -> int:
    """2(2l+1) electrons per subshell."""
    if l < 0:
        raise ValueError("l must be >= 0")
    return 2 * (2 * l + 1)


def hydrogen_energy_ev(n: int, z_eff: float = 1.0) -> float:
    """E_n = -13.6 Z_eff^2 / n^2 eV — ANCHORED hydrogenic."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return -RYDBERG_EV * z_eff**2 / (n * n)


def k_nl_geometry(n: int, l: int, m: float = M_E, l_path: float | None = None) -> float:
    """k_nl ~ (pi/L_bounce)(n - (l+1)/2) — Sec 9.6.1 (GEOMETRY)."""
    if n < 1 or l < 0 or l >= n:
        raise ValueError("require n >= 1 and 0 <= l < n")
    lb = l_bounce_geometry(m, l_path)
    return math.pi / lb * (n - (l + 1) / 2.0)


def eta_ab_overlap(
    separation_m: float,
    l_char: float | None = None,
    *,
    micro: bool = False,
) -> float:
    """
    Gaussian overlap proxy |eta_AB| (O9-2).
    micro=True: L_0 scale; else molecular scale ~ max(a_0, r/2).
    """
    if l_char is not None:
        length = l_char
    elif micro:
        length = coin_half_width()
    else:
        length = max(bohr_radius_geometry(), 0.5 * abs(separation_m))
    if length <= 0:
        return 0.0
    return math.exp(-((separation_m / (2.0 * length)) ** 2))


def u_coulomb_j(q_a: float, q_b: float, r_m: float, eps: float = EPS0) -> float:
    """Coulomb potential energy (J); q in coulombs."""
    if r_m <= 0:
        raise ValueError("r must be positive")
    return q_a * q_b / (4.0 * math.pi * eps * r_m)


def u_coulomb_ev(q_a_e: float, q_b_e: float, r_m: float) -> float:
    """Coulomb energy in eV with charges in units of e."""
    return ev_from_joules(u_coulomb_j(q_a_e * E_CHARGE, q_b_e * E_CHARGE, r_m))


def hbar_omega_b_ev(m: float = M_E, l: float | None = None) -> float:
    return ev_from_joules(HBAR * omega_bounce(m, l))


def bond_share_energy_ev(
    c_b: float,
    eta_ab: float,
    *,
    kappa: float = 0.5,
    m: float = M_E,
    l: float | None = None,
) -> float:
    """Delta E_share = -C_b hbar omega_b |eta|^2 Pi_pin (eV, negative = binding)."""
    pi = pi_pin_kappa(kappa, m, l)
    hw = hbar_omega_b_ev(m, l)
    return -c_b * hw * (eta_ab**2) * pi


def e_bond_covalent_ev(
    r_m: float,
    c_b: float,
    *,
    q_a: float = 1.0,
    q_b: float = 1.0,
    kappa: float = 0.5,
    l_char: float | None = None,
) -> float:
    """E_bond = U_C - C_b hbar omega_b |eta|^2 Pi_pin at separation r."""
    eta = eta_ab_overlap(r_m, l_char)
    u_c = u_coulomb_ev(q_a, q_b, r_m)
    share = bond_share_energy_ev(c_b, eta, kappa=kappa)
    return u_c + share


def calibrate_c_b_h2(
    *,
    bond_ev: float = H2_BOND_EV,
    r0_m: float = H2_R0_M,
    kappa: float = 0.5,
) -> float:
    """
    Invert H2 anchor: E_bond(r0) = U_C - C_b hbar omega_b |eta|^2 Pi_pin.
    bond_ev = positive dissociation energy => E_bond(r0) = -bond_ev.
    """
    u_c = u_coulomb_ev(1.0, 1.0, r0_m)
    eta = eta_ab_overlap(r0_m)
    pi = pi_pin_kappa(kappa)
    hw = hbar_omega_b_ev()
    denom = hw * (eta**2) * pi
    if denom <= 0:
        return 0.0
    return (u_c + bond_ev) / denom


def a_lm_coin(theta: float, phi: float, l: int, m_l: int) -> float:
    """
    Minimal real spherical-harmonic bridge on coin angles (O9-3 partial).
    l=0,1 only for tests; higher l uses hydrogenic ANCHORED labels.
    """
    if l == 0 and m_l == 0:
        return 1.0 / math.sqrt(4.0 * math.pi)
    if l == 1:
        if m_l == 0:
            return math.sqrt(3.0 / (4.0 * math.pi)) * math.cos(theta)
        if m_l == 1:
            return -math.sqrt(3.0 / (8.0 * math.pi)) * math.sin(theta) * math.cos(phi)
        if m_l == -1:
            return math.sqrt(3.0 / (8.0 * math.pi)) * math.sin(theta) * math.sin(phi)
    raise ValueError(f"unsupported (l,m)=({l},{m_l}) in minimal coin map")


def radial_envelope_nl(r: float, n: int, l: int, a0: float | None = None) -> float:
    """Leading hydrogenic radial factor r^l exp(-r/a0) — MODEL bridge (O9-3)."""
    if r < 0 or n < 1 or l < 0:
        raise ValueError("invalid quantum numbers or r")
    a = a0 if a0 is not None else bohr_radius_geometry()
    return (r**l) * math.exp(-r / a)


def c_n_sharing(
    n: int,
    z: int,
    *,
    c0: float = 1.0,
    f_n: float = 1.0,
    n0: float = 6.0,
    nz0: float = 4.0,
    r0: float = 1.0,
    r_a: float = 1.0,
) -> float:
    """Sec 4 / 9 C_N(N,Z) sharing factor (MODEL structure)."""
    if n <= 0:
        return 0.0
    asym = math.exp(-((n - z) / nz0) ** 2)
    return c0 * f_n * (1.0 - math.exp(-n / n0)) * asym * math.exp(-r0 / r_a)


def b_share_ev(n: int, z: int, b_net: float = 0.1) -> float:
    """B_share = -b_net N C_N (eV-scale placeholder)."""
    return -b_net * n * c_n_sharing(n, z)


def report_atom_geometry() -> str:
    a0 = bohr_radius_geometry()
    l0 = coin_half_width()
    c_b = calibrate_c_b_h2()
    e_h2 = e_bond_covalent_ev(H2_R0_M, c_b)
    lines = [
        "AETHOS atom / bond geometry (Sec 9, Step 9 gate)",
        "=" * 56,
        f"L_0 (C1)               = {l0:.6e} m",
        f"a_0 (anchored)         = {a0:.6e} m  (ratio a_0/L_0 ~ {a0/l0:.1f})",
        f"hbar omega_b           = {hbar_omega_b_ev():.1f} eV",
        f"E_1 (H)                = {hydrogen_energy_ev(1):.4f} eV",
        f"shell n=2 capacity     = {shell_capacity(2)}",
        f"k_2,0^geom             = {k_nl_geometry(2, 0):.3e} m^-1",
        "",
        "H2 bond (E calibration check):",
        f"  C_b (fit from 4.52 eV) = {c_b:.6e}",
        f"  E_bond @ r0            = {e_h2:.4f} eV  (well depth -{H2_BOND_EV} eV)",
        f"  |eta_AB| @ r0          = {eta_ab_overlap(H2_R0_M):.4f}",
        "",
        "O9-3 partial: a_lm coin map l=0,1; full spectrum without hydrogenic import OPEN.",
        "",
    ]
    return "\n".join(lines)


# --- Section 10: planetary / cosmic scales (Step 10 gate) ---

MU0 = 4.0 * math.pi * 1e-7  # H/m
G_NEWTON = 6.674_30e-11  # m^3 kg^-1 s^-2
K_B = 1.380_649e-23  # J/K
M_SUN = 1.989e30  # kg
R_EARTH = 6.371e6  # m
B_EARTH_SURFACE_T = 3.12e-5  # ~31 uT equatorial anchor (E check)
MU_N_J_T = 1.913_042_73 * MU_N  # empirical neutron moment scale (J/T)
TAU_FLIP_MEAN_YR = 4.5e5  # paleomagnetic mean reversal interval
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
M_CH_SOLAR = 1.4
R_NS_TYPICAL_M = 1.2e4
M_NS_TYPICAL_KG = 1.4 * M_SUN
B_NS_TYPICAL_T = 1.0e8


def mu_cell_neutron_network(
    *,
    g_eff: float = G_EFF_NEUTRON,
    sigma_z: float = 1.0,
    pi_pin: float = 1.0,
) -> float:
    """Micro cell moment |mu_i| from Sec 2/4 (O10-1)."""
    return abs(g_eff * MU_B * sigma_z * pi_pin)


def n_eff_participating(f_part: float, n_n: float, volume_m3: float) -> float:
    """Participating neutron count N_eff = f_part n_n V_c."""
    return max(0.0, f_part * n_n * volume_m3)


def b_dipole_surface_t(
    n_eff: float,
    mu_cell: float,
    radius_m: float,
    *,
    m_order: float = 1.0,
) -> float:
    """
    Axial dipole field at equatorial surface (Sec 10.2.3):
    B(R) = mu0/(2pi) * N_eff mu_cell |m| / R^3
    """
    if radius_m <= 0:
        return 0.0
    m_core = n_eff * mu_cell * abs(m_order)
    return MU0 / (2.0 * math.pi) * m_core / (radius_m**3)


def b_neutron_star_dipole_t(
    mass_kg: float,
    radius_m: float,
    *,
    f_ns: float,
    mu_n: float = MU_N_J_T,
    xi_ns: float = 1.0,
    m_order: float = 1.0,
) -> float:
    """Neutron-star dipole from Sec 10.2.4."""
    n_ns = mass_kg / M_N
    n_eff = f_ns * n_ns
    m_core = n_eff * abs(mu_n) * xi_ns * abs(m_order)
    if radius_m <= 0:
        return 0.0
    return MU0 / (2.0 * math.pi) * m_core / (radius_m**3)


def double_well_u(m: float, u0: float) -> float:
    """Bistable polarity potential U(m) = U0/4 (m^2-1)^2, m in [-1,1]."""
    return 0.25 * u0 * (m * m - 1.0) ** 2


def flip_barrier_j(
    delta_u0: float,
    zeta_p: float,
    p_core: float,
    p_eq: float,
) -> float:
    """Delta U(P) = Delta U_0 - zeta_P (P_core - P_eq)."""
    return delta_u0 - zeta_p * (p_core - p_eq)


def tau0_core_attempt(alpha_core: float, gamma_obs: float) -> float:
    """Microscopic attempt timescale tau_0 ~ 2pi hbar / (alpha Gamma_obs)."""
    denom = alpha_core * gamma_obs
    if denom <= 0:
        return float("inf")
    return 2.0 * math.pi * HBAR / denom


def d_core_noise(
    t_core_k: float,
    nu_th: float = 1.0,
    *,
    chi_d: float = 0.0,
    var_gamma: float = 0.0,
) -> float:
    """D_core = k_B T nu_th + chi_D Var(Gamma_obs)."""
    return K_B * t_core_k * nu_th + chi_d * var_gamma


def tau_flip_seconds(tau0: float, delta_u: float, d_core: float) -> float:
    """Activated reversal tau_flip ~ tau_0 exp(Delta U / D_core)."""
    if d_core <= 0 or not math.isfinite(delta_u):
        return float("inf")
    return tau0 * math.exp(delta_u / d_core)


def calibrate_f_part_dipole(
    b_target_t: float,
    radius_m: float,
    n_n: float,
    volume_m3: float,
    mu_cell: float,
    *,
    m_order: float = 1.0,
) -> float:
    """Invert f_part from observed dipole field (E check — O10-1)."""
    denom = b_dipole_surface_t(1.0, mu_cell, radius_m, m_order=m_order)
    if n_n <= 0 or volume_m3 <= 0 or denom <= 0:
        return 0.0
    return b_target_t / (denom * n_n * volume_m3)


def schwarzschild_radius(mass_kg: float) -> float:
    return 2.0 * G_NEWTON * mass_kg / (C**2)


def escape_speed(mass_kg: float, radius_m: float) -> float:
    if radius_m <= 0:
        return 0.0
    return math.sqrt(2.0 * G_NEWTON * mass_kg / radius_m)


def gravitational_time_dilation_factor(radius_m: float, mass_kg: float) -> float:
    """sqrt(1 - 2GM/(rc^2)) — ANCHORED (Sec 10.9)."""
    if radius_m <= 0:
        return 0.0
    x = schwarzschild_radius(mass_kg) / radius_m
    if x >= 1.0:
        return 0.0
    return math.sqrt(1.0 - x)


def hydrostatic_dp_dr(mass_enclosed_kg: float, rho_kg_m3: float, r_m: float) -> float:
    """dP/dr = -G M(r) rho / r^2 (Sec 10.4)."""
    if r_m <= 0:
        return 0.0
    return -G_NEWTON * mass_enclosed_kg * rho_kg_m3 / (r_m**2)


def w_eos_from_sea_pressure(pi_s: float, rho_de_c2: float) -> float:
    """w(z) = -1 + Pi_s / (rho_DE c^2) when p_DE = -rho c^2 + Pi_s."""
    if rho_de_c2 == 0:
        return -1.0
    return -1.0 + pi_s / rho_de_c2


def w_z_cpl(w0: float, wa: float, z: float) -> float:
    """CPL parametrization w(z) = w_0 + w_a z/(1+z)."""
    return w0 + wa * z / (1.0 + z)


def cpl_from_sea_pressure(pi0: float, rho_de0_c2: float, n: float) -> tuple[float, float]:
    """Map Pi_s(z)=Pi_0(1+z)^n to (w_0, w_a) at leading order."""
    if rho_de0_c2 == 0:
        return -1.0, 0.0
    w0 = -1.0 + pi0 / rho_de0_c2
    wa = n * pi0 / rho_de0_c2
    return w0, wa


def report_cosmic_geometry() -> str:
    mu_cell = mu_cell_neutron_network(pi_pin=pi_pin_kappa(0.5))
    # illustrative core: outer-core scale, effective neutron density proxy
    v_core = 1.7e18  # m^3 order-of-magnitude outer core
    n_n_proxy = 1.0e26  # effective participating density scale (calibration handle)
    f_part = calibrate_f_part_dipole(
        B_EARTH_SURFACE_T, R_EARTH, n_n_proxy, v_core, mu_cell
    )
    b_earth = b_dipole_surface_t(
        n_eff_participating(f_part, n_n_proxy, v_core), mu_cell, R_EARTH
    )
    b_ns = b_neutron_star_dipole_t(M_NS_TYPICAL_KG, R_NS_TYPICAL_M, f_ns=1e-3)
    rs_sun = schwarzschild_radius(M_SUN)
    dil_earth = gravitational_time_dilation_factor(R_EARTH, 5.972e24)
    w0, wa = cpl_from_sea_pressure(1e-10, 1.0, n=0.05)
    lines = [
        "AETHOS cosmic / planetary geometry (Sec 10, Step 10 gate)",
        "=" * 56,
        f"mu_cell (network)      = {mu_cell:.3e} J/T",
        f"f_part (Earth E-check) = {f_part:.3e}",
        f"B_earth model          = {b_earth:.3e} T  (anchor {B_EARTH_SURFACE_T:.3e})",
        f"B_NS @ f_NS=1e-3       = {b_ns:.3e} T  (typical ~1e8–1e11)",
        f"r_s (Sun)              = {rs_sun:.1f} m",
        f"clock factor @ Earth   = {dil_earth:.12f}",
        f"w_0, w_a (small Pi)    = {w0:.6f}, {wa:.6e}",
        "",
        "O10-1: f_part, f_NS ab-initio OPEN; dipole chain GEOMETRY structure.",
        "O10-2: geodynamo tau_flip calibration OPEN.",
        "O10-3: Pi_0, n vs SN/CMB datasets OPEN (shared O11-4).",
        "",
    ]
    return "\n".join(lines)


# --- Section 11: dark matter / dark energy (Step 11 gate, P11-1–3) ---

RHO_DE_CRIT_H0 = 6.9e-27  # kg/m^3 order-of-magnitude critical density today (E anchor)
RHO_DM_HALO_LOCAL = 5.0e-22  # kg/m^3 solar-neighborhood DM scale (E anchor)
SIGMA_GAMMA_DM_EXP_UPPER = 1.0e-15  # m^2 phenomenological EM-coupling ceiling (E check)
H0_SI = 2.2e-18  # s^-1 ~ 70 km/s/Mpc


def omega_from_ev(probe_ev: float) -> float:
    """Angular frequency for probe energy in electron-volts (not MeV)."""
    return ev_to_joules(probe_ev * 1e-6) / HBAR


def e_spring_j(m_spring: float = M_E, l: float | None = None) -> float:
    """E_spring = hbar sqrt(k_s / m_spring) from coin spring (O11-1)."""
    ks = k_s_from_geometry(M_E, l)
    if m_spring <= 0 or ks <= 0:
        return 0.0
    return HBAR * math.sqrt(ks / m_spring)


def sigma_geom_spring(r_spring: float | None = None) -> float:
    """Geometric cross-section pi R_spring^2 — uses L_0 by default."""
    r = r_spring if r_spring is not None else coin_half_width()
    return math.pi * r * r


def k_sup_rayleigh(omega_rad_s: float, e_spring_joules: float) -> float:
    """Rayleigh suppression K_sup = (hbar omega / E_spring)^4; S_res,DM = 0."""
    if e_spring_joules <= 0 or omega_rad_s <= 0:
        return 0.0
    x = HBAR * omega_rad_s / e_spring_joules
    return x**4


def sigma_gamma_dm(
    omega_rad_s: float,
    *,
    m_spring: float = M_E,
    r_spring: float | None = None,
    l: float | None = None,
) -> float:
    """sigma_gammaDM = sigma_geom * K_sup (Sec 11.3.3)."""
    e_s = e_spring_j(m_spring, l)
    return sigma_geom_spring(r_spring) * k_sup_rayleigh(omega_rad_s, e_s)


def sigma_det_class_a() -> float:
    """Class A pin/readout channel: exact null (O11-3)."""
    return 0.0


def sigma_det_class_bc_upper(
    omega_rad_s: float,
    *,
    m_spring: float = M_E,
    r_spring: float | None = None,
) -> float:
    """Classes B/C upper bound via Rayleigh suppression."""
    return sigma_gamma_dm(omega_rad_s, m_spring=m_spring, r_spring=r_spring)


def phi_ab_steady_state(gamma_fill: float, gamma_snap: float) -> float:
    """P11-3 steady fill: phi = Gamma_fill / (Gamma_fill + Gamma_snap)."""
    g = max(0.0, gamma_fill)
    s = max(0.0, gamma_snap)
    den = g + s
    return g / den if den > 0 else 0.0


def phi_ab_at_time(
    t: float,
    gamma_fill: float,
    gamma_snap: float,
    *,
    phi0: float = 0.0,
) -> float:
    """Closed-form phi(t) for constant fill/snap rates."""
    return coherence_at_time(t, c0=phi0, gamma_form=gamma_fill, gamma_break=gamma_snap)


def ell_c_from_rho_dm(
    rho_dm_kg_m3: float,
    *,
    m_dm: float = M_E,
    fill_factor: float = 1.0,
) -> float:
    """
    Mesh reach ~ inter-spring spacing from DM density (O11-5 partial).
    MODEL until ab-initio filament geometry.
    """
    if rho_dm_kg_m3 <= 0 or m_dm <= 0:
        return ell_c_from_geometry()
    n = rho_dm_kg_m3 / m_dm
    spacing = n ** (-1.0 / 3.0)
    return max(coin_half_width(), fill_factor * spacing)


def circular_velocity_squared(m_enclosed_kg: float, radius_m: float) -> float:
    """v_c^2 = G M(<r) / r (Sec 11.4)."""
    if radius_m <= 0:
        return 0.0
    return G_NEWTON * m_enclosed_kg / radius_m


def enclosed_mass_flat_halo(m0_kg: float, r0_m: float, radius_m: float) -> float:
    """Illustrative flat rotation: M(<r) proportional to r for r > r0."""
    if radius_m <= r0_m:
        return m0_kg * (radius_m / r0_m) ** 2 if r0_m > 0 else 0.0
    return m0_kg * (radius_m / r0_m)


def gamma_unpin_arrhenius(
    nu0: float,
    delta_sep_j: float,
    t_env_k: float,
) -> float:
    """Gamma_unpin = nu_0 exp(-Delta_sep / k_B T_env)."""
    if t_env_k <= 0 or delta_sep_j <= 0:
        return 0.0
    return nu0 * math.exp(-delta_sep_j / (K_B * t_env_k))


def gamma_sep_rate(gamma_unpin: float, gamma_pin: float) -> float:
    """Gamma_sep = max(0, Gamma_unpin - Gamma_pin)."""
    return max(0.0, gamma_unpin - gamma_pin)


def q_transfer_rate(
    rho_nm_kg_m3: float,
    gamma_sep: float,
    e_gamma_bar_j: float,
    *,
    m_n: float = M_N,
) -> float:
    """Q = rho_NM (Gamma_sep / (m_N c^2)) E_gamma_bar (Sec 11.7.1)."""
    if m_n <= 0 or gamma_sep <= 0:
        return 0.0
    return rho_nm_kg_m3 * (gamma_sep / (m_n * C**2)) * e_gamma_bar_j


def sector_split_fractions(
    e_gamma_bar_j: float,
    e_spring_bar_j: float,
) -> tuple[float, float]:
    """(f_e, f_m) with f_e + f_m = 1 when E_N = E_gamma + E_spring."""
    e_n = e_gamma_bar_j + e_spring_bar_j
    if e_n <= 0:
        return 0.0, 0.0
    return e_gamma_bar_j / e_n, e_spring_bar_j / e_n


def rho_dot_frw(rho_kg_m3: float, p_pa: float, hubble_s: float) -> float:
    """Continuity: dot rho = -3 H (rho + p/c^2)."""
    return -3.0 * hubble_s * (rho_kg_m3 + p_pa / (C**2))


def rho_dot_de_const(rho_de_kg_m3: float, hubble_s: float) -> float:
    """w = -1 limit: dot rho_DE = 0."""
    _ = rho_de_kg_m3, hubble_s
    return 0.0


def u_gamma_free_density(n_gamma_m3: float, omega_mean_rad_s: float) -> float:
    """u_gamma,free = n_gamma hbar <omega> (J/m^3)."""
    return max(0.0, n_gamma_m3) * HBAR * max(0.0, omega_mean_rad_s)


def p_sea_dark_energy(u_gamma_j_m3: float) -> float:
    """P_sea,DE = u_gamma / 3 (Sec 11.5)."""
    return u_gamma_j_m3 / 3.0


def p_de_effective(rho_de_kg_m3: float, pi_s_pa: float) -> float:
    """p_DE = -rho_DE c^2 + Pi_s."""
    return -rho_de_kg_m3 * C**2 + pi_s_pa


def acceleration_condition_w(w: float) -> bool:
    """Cosmic acceleration requires w < -1/3."""
    return w < (-1.0 / 3.0)


def calibrate_pi0_for_w0(
    w0_target: float,
    rho_de0_c2: float,
) -> float:
    """Invert Pi_0 from w_0 = -1 + Pi_0/(rho_DE,0 c^2) (E check — O11-4)."""
    return (w0_target + 1.0) * rho_de0_c2


def report_dark_sector_geometry() -> str:
    omega_eV = omega_from_ev(1.0)
    e_s = e_spring_j()
    sig_eV = sigma_gamma_dm(omega_eV)
    phi = phi_ab_steady_state(gamma_fill=1.0, gamma_snap=0.5)
    lc_halo = ell_c_from_rho_dm(RHO_DM_HALO_LOCAL)
    delta_sep = HBAR * omega_bounce()
    gunpin = gamma_unpin_arrhenius(1.0e12, delta_sep, 300.0)
    gsep = gamma_sep_rate(gunpin, gamma_pin=0.0)
    q = q_transfer_rate(1.0e3, gsep, ev_to_joules(0.511))  # illustrative
    fe, fm = sector_split_fractions(ev_to_joules(0.255), ev_to_joules(0.256))
    w_de = w_eos_from_sea_pressure(1e-10, 1.0)
    pi0 = calibrate_pi0_for_w0(-0.99, 1.0)
    gf_phi = gamma_form_rate(1.0e-6, phi_ab=phi, ell_c=lc_halo)
    gf_empty = gamma_form_rate(1.0e-6, phi_ab=0.0, ell_c=lc_halo)
    lines = [
        "AETHOS dark sector geometry (Sec 11, Step 11 gate)",
        "=" * 56,
        f"E_spring (coin k_s)    = {joules_to_ev(e_s):.3f} MeV",
        f"sigma_gammaDM @ 1 eV   = {sig_eV:.3e} m^2  (bound {SIGMA_GAMMA_DM_EXP_UPPER:.1e})",
        f"sigma_det Class A      = {sigma_det_class_a():.1e}  (exact null)",
        f"phi_AB steady (1,0.5)  = {phi:.4f}",
        f"ell_c(rho_DM halo)     = {lc_halo:.3e} m",
        f"Gamma_sep (300 K demo) = {gsep:.3e} s^-1",
        f"Q demo                 = {q:.3e} J/m^3/s",
        f"f_e, f_m split         = {fe:.4f}, {fm:.4f}",
        f"w_DE (small Pi_s)     = {w_de:.6f}",
        f"Pi_0 for w_0=-0.99     = {pi0:.3e}  (E check)",
        f"Gamma_form phi=0 vs 1  = {gf_empty:.3e} / {gf_phi:.3e}",
        "",
        "P11-3: fill modulates Sec 6 Gamma_form; O11-5 ell_c(rho_DM) MODEL.",
        "O11-1–4: material calibration & dataset joint fit OPEN.",
        "",
    ]
    return "\n".join(lines)


# --- Section 12: Zeno / time / lattice M_lat (Step 12 gate, C6) ---

ACTIVE_SEED_REFERENCE = 100  # canonical bootstrap (E anchor convention, not FIT to 1836)

# E-check profiles from scripts/calibrate_discriminators.py (2026-06-05)
REFERENCE_NETWORK_COUNT = 80  # primes bootstrap: R_pe^pred ~ 1847 (0.6% vs CODATA)
REFERENCE_NETWORK_DEPTH = 3
F_COIN_HE3_DISCRIMINATOR = 0.405  # Lambda_3He/Lambda_4He ~ 1.075 (5-10% band)
F_COIN_HE4_DISCRIMINATOR = 0.5


def _is_small_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def frame_width_n(w0: float, primes: Sequence[int]) -> float:
    """w_n = w_0 / prod(p_k) — finite n keeps w_n > 0 (Sec 12.2)."""
    if w0 <= 0:
        return 0.0
    prod = 1
    for p in primes:
        if p <= 1:
            raise ValueError("prime factors must be >= 2")
        prod *= p
    return w0 / prod


def width_descent_positive_finite(w0: float, primes: Sequence[int]) -> bool:
    """No terminal zero-width frame at finite depth (Sec 12.3)."""
    return frame_width_n(w0, primes) > 0


def mixed_radix_address(indices: Sequence[int], radices: Sequence[int]) -> float:
    """x_n = sum i_k / prod_{j<=k} p_j on [0,1) (Sec 12.4)."""
    if len(indices) != len(radices):
        raise ValueError("indices and radices length mismatch")
    x = 0.0
    denom = 1.0
    for i_k, p_k in zip(indices, radices):
        if p_k <= 0 or i_k < 0 or i_k >= p_k:
            raise ValueError("invalid mixed-radix digit")
        denom *= p_k
        x += i_k / denom
    return x


def geometric_refinement_time_total(dt0: float, r: float, n_terms: int) -> float:
    """T_N = dt0 (1-r^n)/(1-r); T_inf = dt0/(1-r) for 0<r<1 (Sec 12.5)."""
    if n_terms <= 0:
        return 0.0
    if r <= 0 or r >= 1:
        raise ValueError("require 0 < r < 1")
    return dt0 * (1.0 - r**n_terms) / (1.0 - r)


def gamma_lorentz(v: float) -> float:
    if abs(v) >= C:
        return float("inf")
    return 1.0 / math.sqrt(1.0 - (v / C) ** 2)


def v_time_from_v_space(v_space: float) -> float:
    """v_time = sqrt(c^2 - v_space^2) — motion budget (Sec 12.7)."""
    if abs(v_space) >= C:
        return 0.0
    return math.sqrt(C**2 - v_space**2)


def motion_budget_residual(v_space: float) -> float:
    """v_space^2 + v_time^2 - c^2 (should be 0)."""
    vt = v_time_from_v_space(v_space)
    return v_space**2 + vt**2 - C**2


def d_tau_dt_kinematic(v: float) -> float:
    """d tau / dt = sqrt(1 - v^2/c^2) = 1/gamma."""
    g = gamma_lorentz(v)
    return 1.0 / g if math.isfinite(g) and g > 0 else 0.0


def f_clock_doppler(f0: float, v: float) -> float:
    """f = f_0 / gamma (Sec 12.6)."""
    g = gamma_lorentz(v)
    return f0 / g if math.isfinite(g) and g > 0 else 0.0


def bounce_crossing_time_lab(d_coin_m: float, v: float) -> float:
    """
    One inner-photon crossing of coin diameter d at electron drift v (Sec 2.5.1).

    T = d / sqrt(c^2 - v^2) = gamma * d/c.  Returns inf at |v| >= c.
    """
    if d_coin_m <= 0:
        raise ValueError("d_coin_m must be positive")
    if abs(v) >= C:
        return float("inf")
    return d_coin_m / math.sqrt(C**2 - v**2)


def v_time_static_metric(a_metric: float) -> float:
    """v_time = c sqrt(A) for static ds^2 = -A c^2 dt^2 + ... (Sec 12.8)."""
    if a_metric <= 0:
        return 0.0
    return C * math.sqrt(a_metric)


def v_space_static_metric(a_metric: float) -> float:
    if a_metric <= 0 or a_metric >= 1:
        return 0.0 if a_metric >= 1 else C
    return C * math.sqrt(1.0 - a_metric)


def v_flow_newtonian(mass_kg: float, radius_m: float) -> float:
    """v_flow = sqrt(2GM/r) — Sec 10/12 bridge."""
    return escape_speed(mass_kg, radius_m)


def v_time_from_gravity(radius_m: float, mass_kg: float) -> float:
    """Budget time component at r in Schwarzschild field."""
    return C * gravitational_time_dilation_factor(radius_m, mass_kg)


def prime_split_weight(p: int, p_max: int = 97) -> float:
    """P(p) = log(p)/log(P_max) on primes (Sec 12.3.1 — MODEL)."""
    if p_max <= 1 or p < 2 or p > p_max or not _is_small_prime(p):
        return 0.0
    return math.log(p) / math.log(p_max)


def lambda_descent_rate(gamma_obs: float, p_max: int = 97) -> float:
    """lambda_desc = Gamma_obs E[log p_k] (Sec 12.3.1)."""
    primes = [p for p in range(2, p_max + 1) if _is_small_prime(p)]
    e_log = sum(prime_split_weight(p, p_max) * math.log(p) for p in primes)
    return max(0.0, gamma_obs) * e_log


def width_under_descent(w0: float, gamma_obs: float, t: float, p_max: int = 97) -> float:
    """w(t) = w_0 exp(-lambda_desc t) > 0 for finite t."""
    lam = lambda_descent_rate(gamma_obs, p_max)
    return w0 * math.exp(-lam * max(0.0, t))


def f_clock_baryon_hz(m: float = M_E, l: float | None = None) -> float:
    """NM clock from bounce: f ~ omega_b / (2 pi)."""
    return omega_bounce(m, l) / (2.0 * math.pi)


def f_clock_dm_coherent() -> float:
    """No inner-photon mode => coherent DM clock null (Sec 12.9.1)."""
    return 0.0


def f_clock_dm_thermal(t_dm_k: float, *, m_spring: float = M_E, l: float | None = None) -> float:
    """Thermal tail clock ~ (k_B T/h) exp(-E_spring/(k_B T))."""
    if t_dm_k <= 0:
        return 0.0
    e_s = e_spring_j(m_spring, l)
    return (K_B * t_dm_k / H) * math.exp(-e_s / (K_B * t_dm_k))


def s_clock_suppression(t_dm_k: float, *, m: float = M_E, l: float | None = None) -> float:
    """S_clock = f_DM,therm / f_NM (Sec 12.9.1)."""
    f_nm = f_clock_baryon_hz(m, l)
    if f_nm <= 0:
        return 0.0
    return f_clock_dm_thermal(t_dm_k, m_spring=m, l=l) / f_nm


def chain_cascade_weight(chain: tuple[int, ...]) -> float:
    """
    Per-node lattice cascade load (C6):
    (k+1) segments × anchor-span ratio sum(chain)/chain[0].
    """
    k = len(chain)
    if k == 0:
        return 1.0
    p1 = chain[0]
    if p1 <= 0:
        return 1.0
    return (k + 1) * float(sum(chain)) / float(p1)


def m_lat_from_bootstrap_net(net: object) -> float:
    """M_lat from an ActiveNetwork100 instance (shared denominator logic)."""
    from aethos_active import BRANCHES_PER_VECTOR, VECTORS_PER_NODE, WINGS_PER_ROOM

    total = sum(chain_cascade_weight(n.chain) for n in net.nodes)  # type: ignore[attr-defined]
    origins = len(net._origin_index)  # type: ignore[attr-defined]
    denom = origins + BRANCHES_PER_VECTOR + VECTORS_PER_NODE
    if denom <= 0:
        return 1.0
    return total * WINGS_PER_ROOM / denom


def m_lat_from_active_network(
    *,
    count: int = ACTIVE_SEED_REFERENCE,
    origin_max_depth: int = 3,
    chain_species: object | None = None,
) -> float:
    """
    Lattice cascade multiplier from active anchor network (Step 12 / C6).

    M_lat = (sum chain weights) × WINGS / (N_origins + N_branches + N_vectors)
    Reference bootstrap: 100 nodes, primes, depth 3 → ~1491 (E-check vs 1488 gap).
    """
    from aethos_active import (
        BRANCHES_PER_VECTOR,
        VECTORS_PER_NODE,
        WINGS_PER_ROOM,
        ActiveNetwork100,
    )
    from aethos_sequences import SequenceKind

    species = chain_species if chain_species is not None else SequenceKind.PRIMES
    net = ActiveNetwork100.bootstrap(
        count=count,
        origin_max_depth=origin_max_depth,
        chain_species=species,  # type: ignore[arg-type]
    )
    return m_lat_from_bootstrap_net(net)


def m_lat_from_material_blob(
    blob: object,
    *,
    count: int = REFERENCE_NETWORK_COUNT,
    origin_max_depth: int = REFERENCE_NETWORK_DEPTH,
) -> float:
    """M_lat when anchor species follow ElectronBlob (C6 material path)."""
    from aethos_active import ActiveNetwork100

    net = ActiveNetwork100.bootstrap_from_blob(blob, count=count, origin_max_depth=origin_max_depth)  # type: ignore[arg-type]
    return m_lat_from_bootstrap_net(net)


def r_pe_model_with_lattice(
    *,
    count: int = ACTIVE_SEED_REFERENCE,
    origin_max_depth: int = 3,
    chain_species: object | None = None,
) -> float:
    """R_pe^model = R_pe^(0) × M_lat — Step 3 + Step 12 closure (C2 consequence check)."""
    return r_pe_spring_only() * m_lat_from_active_network(
        count=count,
        origin_max_depth=origin_max_depth,
        chain_species=chain_species,
    )


def wing_activation_analysis(
    *,
    count: int = REFERENCE_NETWORK_COUNT,
    origin_max_depth: int = REFERENCE_NETWORK_DEPTH,
) -> dict[str, float]:
    """
    Hidden pattern: 40 origins x 32 wings = 1280 slots; n active nodes => fraction 1/16 at n=80.

    Returns nodes_per_origin, wings_total, activation_fraction, role_cycles (count/5).
    """
    from aethos_origins import OriginTree

    n_orig = len(list(OriginTree.bootstrap(max_depth=origin_max_depth).walk()))
    wings_total = float(n_orig * 32)
    nodes_per_origin = count / n_orig if n_orig else 0.0
    wings_per_origin = 32.0
    return {
        "n_origins": float(n_orig),
        "wings_total": wings_total,
        "active_nodes": float(count),
        "nodes_per_origin": nodes_per_origin,
        "wing_fraction_per_origin": nodes_per_origin / wings_per_origin if wings_per_origin else 0.0,
        "global_activation_fraction": count / wings_total if wings_total else 0.0,
        "role_cycles": count / 5.0,
    }


def r_pe_model_reference_bootstrap(chain_species: object | None = None) -> float:
    """E-check optimum: primes, count=80, depth=3 → R_pe^pred ~ 1847 (see calibration_sheet)."""
    from aethos_sequences import SequenceKind

    species = chain_species if chain_species is not None else SequenceKind.PRIMES
    return r_pe_model_with_lattice(
        count=REFERENCE_NETWORK_COUNT,
        origin_max_depth=REFERENCE_NETWORK_DEPTH,
        chain_species=species,
    )


def report_discriminator_calibration() -> str:
    """E-check profiles: Ch 16/17 discriminators (calibration_sheet 2026-06-05)."""
    from aethos_origins import OriginTree

    r_ref = r_pe_model_reference_bootstrap()
    he_raw = lambda_he3_he4_ratio()
    he_cal = lambda_he3_he4_ratio_calibrated()
    err_r = 100.0 * abs(r_ref - R_PE) / R_PE
    n_orig = len(list(OriginTree.bootstrap(max_depth=REFERENCE_NETWORK_DEPTH).walk()))
    m4m3 = M_HE4 / M_HE3
    lines = [
        "AETHOS discriminator E-check (Ch 16/17, calibration_sheet)",
        "=" * 56,
        "Mass ratio R_pe = (pi^2/8) x M_lat:",
        f"  Spring factor pi^2/8     = {r_pe_spring_only():.6f}",
        f"  E-gap M_lat target       = {lattice_mass_multiplier():.2f}",
        f"  Reference bootstrap      primes n={REFERENCE_NETWORK_COUNT} depth={REFERENCE_NETWORK_DEPTH}",
        f"  Origins (1+3+9+27)       = {n_orig}  => M_lat = (32/52)*sum_mu",
        f"  Role ledger              n=80 = 16 x 5 (SOLO..FOUR_WAY balance)",
        f"  Wing activation          80/1280 = 2/32 = 1/16 per origin room",
        f"  R_pe^pred                = {r_ref:.2f}",
        f"  R_pe^E (CODATA)          = {R_PE:.2f}",
        f"  Relative error           = {err_r:.2f}%",
        "",
        "He isotope Lambda_3He / Lambda_4He = (f3/f4)*(m4/m3):",
        f"  Mass-only m4/m3          = {m4m3:.4f}",
        f"  Need f3/f4 for 1.075     = {1.075/m4m3:.4f}",
        f"  E-check f3/f4            = {F_COIN_HE3_DISCRIMINATOR/F_COIN_HE4_DISCRIMINATOR:.4f}",
        f"  Placeholder 0.75/0.15    -> {he_raw:.3f}  (reject)",
        f"  E-check profile          -> {he_cal:.4f}",
        "",
        "Why: scripts/pattern_why_discriminators.py",
        "Sweep: scripts/calibrate_discriminators.py",
        "",
    ]
    return "\n".join(lines)


def report_time_zeno_geometry() -> str:
    w0 = 1.0
    primes = (2, 3, 5)
    mlat_ref = m_lat_from_active_network(
        count=REFERENCE_NETWORK_COUNT,
        origin_max_depth=REFERENCE_NETWORK_DEPTH,
    )
    r_pred = r_pe_model_reference_bootstrap()
    lam = lambda_descent_rate(1.0 / TAU_N)
    lines = [
        "AETHOS time / Zeno / lattice geometry (Sec 12, Step 12 gate)",
        "=" * 56,
        f"w_3 (2,3,5 split)      = {frame_width_n(w0, primes):.6f}  (>0 finite)",
        f"mixed-radix x          = {mixed_radix_address((1, 0, 2), primes):.6f}",
        f"T_inf (r=0.5)          = {geometric_refinement_time_total(1.0, 0.5, 10_000):.4f} s",
        f"motion budget residual = {motion_budget_residual(0.5 * C):.3e}",
        f"d tau/dt @ v=c/2       = {d_tau_dt_kinematic(0.5 * C):.6f}",
        f"v_time @ Earth surf    = {v_time_from_gravity(R_EARTH, 5.972e24):.3e} m/s",
        f"lambda_desc (tau_n)    = {lam:.3e} s^-1",
        f"S_clock (DM @ 300K)    = {s_clock_suppression(300.0):.3e}",
        "",
        "Lattice mass ratio (C6, E-check bootstrap):",
        f"  R_pe^(0)             = {r_pe_spring_only():.6f}",
        f"  M_lat (n=80, d=3)    = {mlat_ref:.2f}",
        f"  R_pe^pred            = {r_pred:.2f}  (CODATA {R_PE:.2f})",
        "",
        "O12-1: P(p) micro-derivation OPEN; O12-2: rotating metric OPEN.",
        "",
    ]
    return "\n".join(lines)


def report_tunneling_regimes() -> str:
    lines = [
        "AETHOS tunneling regimes (Sec 7, P7-2)",
        "=" * 56,
        f"Soft example: Pi_pin={pi_pin_from_bias(1e18, 1e21):.4f} -> {classify_compression(0.01).value}",
        f"Hard example: Pi_pin={pi_pin_from_bias(1e21, 1e18):.4f} -> {classify_compression(0.99).value}",
        "",
        "Energy anchors (order of magnitude):",
        "  lab barrier     ~ eV–keV     (soft)",
        f"  m_e c^2         ~ {joules_to_ev(M_E * C**2):.3f} MeV",
        f"  Q_beta (neutron) ~ {Q_BETA_EV} MeV (hard escape scale)",
        "",
        "Transit with DM fill (phi_path=0.8, eta_DM=0.5):",
        f"  xi_shred: {xi_shred_from_field(1.0, 2.0):.3f} -> {xi_shred_with_dm(0.8, xi_shred_from_field(1.0, 2.0)):.3f}",
        "",
    ]
    return "\n".join(lines)


def report_measurement_geometry() -> str:
    b = 1.0e3
    tau = measurement_tau_window(0.04, 1.0e5)
    lam_geom = lambda_n_from_coin_gradient(b, tau, g_e=1.0)
    ratio = strong_measurement_ratio(b, g_e=1.0, kappa=0.0)
    c = calibrate_measurement_sg()
    a0, b0 = 0.0, math.pi / 4
    lines = [
        "AETHOS measurement geometry (Sec 5, Step 5 gate)",
        "=" * 56,
        f"L_0 (C1)              = {coin_half_width():.6e} m",
        f"E_obs micro (J)        = {e_obs_coupling_from_gradient(b):.3e}",
        f"omega_eff / Omega(0)   = {ratio:.3e}  (>>1 => axis pin)",
        f"Lambda_n (L_0 path)    = {lam_geom:.3e}",
        f"Lambda_n (SG cal)      = {c.lambda_n:.4f}",
        f"Kraus exp(-Lambda)     = {kraus_decoherence_factor(c.lambda_n):.3e}",
        f"pin p                  = {c.pin_p:.4f}",
        "",
        "Bell / O5-3 (C5):",
        f"  E QM (0, pi/4)       = {bell_correlation_qm(a0, b0):.6f}",
        f"  E sign sketch REJECT = {bell_correlation_coin_geometry(a0, b0, n_samples=80_000):.6f}",
        f"  E joint ripple PART  = {bell_correlation_joint_ripple_linear(a0, b0):.6f}  (half-scale)",
        f"  CHSH S (QM)          = {chsh_s_quantum():.6f}",
        "",
    ]
    return "\n".join(lines)


def report_measurement_calibration() -> str:
    c = calibrate_measurement_sg()
    lines = [
        "AETHOS measurement calibration (Sec 5, O5-2 partial)",
        "=" * 56,
        f"Reference SG: |dB/dz|={c.b_grad_z:.3e} T/m, L={c.l_mag} m",
        f"tau_m = L/v = {c.tau_m:.3e} s",
        f"E_obs (eff)     = {c.e_obs:.3e} J",
        f"g_E (fit)       = {c.g_e:.3e}  (target Lambda_n=5)",
        f"Lambda_n        = {c.lambda_n:.4f}",
        f"pin strength p  = {c.pin_p:.4f}",
        f"T_bounce (e)    = {bounce_period():.3e} s",
        "",
        "Bell / CHSH checks:",
        f"  E(0, pi/4) QM  = {bell_correlation_qm(0, math.pi/4):.6f}",
        f"  E sign sketch REJECT = {bell_correlation_coin_geometry(0, math.pi/4, n_samples=80_000):.6f}",
        f"  E joint ripple PART  = {bell_correlation_joint_ripple_linear(0, math.pi/4):.6f}",
        f"  CHSH S         = {chsh_s_quantum():.6f}  (expect 2*sqrt(2) ~ 2.828)",
        "",
    ]
    return "\n".join(lines)


def report_calibration() -> str:
    lines = [
        "AETHOS physics calibration (neutron pressure, Sec 4)",
        "=" * 56,
        f"tau_n (anchor)     = {TAU_N} s",
        f"Q_beta             = {Q_BETA_EV} MeV",
        f"Delta_m_np         = {DM_NP_EV} MeV",
        f"g_eff (fit)        = {G_EFF_NEUTRON:.4f}  (check {g_eff_from_mu_n():.4f})",
        f"K_f @ alpha=1      = {k_fusion_from_r_pe():.6f}",
        "",
    ]
    for gap in ("scale", "cavity", "q_cavity"):
        c = calibrate_neutron_pressure(gap=gap)
        lines.extend(
            [
                f"Mode: {c.mode}",
                f"  P_gap              = {joules_to_ev(c.p_gap_j):.4f} MeV",
                f"  omega_in0          = {c.omega_in0:.3e} rad/s",
                f"  alpha              = {joules_to_ev(c.alpha):.4f} MeV",
                f"  Gamma_obs          = {c.gamma_obs:.6e} s^-1",
                f"  dP/dt              = {c.dP_dt:.6e} W (effective)",
                f"  R_coin             = {c.r_coin:.3e} m",
                f"  Phi_obs (if sigma) = {c.phi_obs:.3e} m^-2 s^-1",
                f"  t_escape check     = {c.p_gap_j / c.dP_dt:.2f} s",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(report_spacetime_geometry())
    print()
    print(report_spacetime_wiring())
    print()
    print(report_coin_geometry())
    print()
    print(report_fusion_geometry())
    print()
    print(report_neutron_geometry())
    print()
    print(report_measurement_geometry())
    print()
    print(report_entanglement_geometry())
    print()
    print(report_tunneling_geometry())
    print()
    print(report_double_slit_geometry())
    print()
    print(report_atom_geometry())
    print()
    print(report_cosmic_geometry())
    print()
    print(report_dark_sector_geometry())
    print()
    print(report_time_zeno_geometry())
    print()
    print(report_discriminator_calibration())
    print(report_calibration())
    print()
    print(report_measurement_calibration())
    print()
    print(report_tunneling_regimes())
    print()
    print(report_vapor_spectrum())
