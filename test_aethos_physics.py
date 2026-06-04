"""
Section 4–5 physics calibration tests.
Run: python test_aethos_physics.py
"""

from __future__ import annotations

import math
import sys
import unittest

from aethos_physics import (
    H2_BOND_EV,
    H2_R0_M,
    MU_B,
    RYDBERG_EV,
    TAU_N,
    CompressionRegime,
    bell_correlation_coin_geometry,
    bell_correlation_joint_ripple_linear,
    bell_correlation_phi_fill,
    bell_correlation_qm,
    bounce_period,
    coin_half_width,
    coherence_steady_state,
    compton_wavelength,
    delta_kappa,
    ell_c_from_geometry,
    e_obs_coupling_from_gradient,
    entanglement_lifetime,
    f_bounce,
    gamma_break_rate,
    gamma_form_rate,
    j_ab_overlap,
    kraus_decoherence_factor,
    k_s_from_geometry,
    lambda_n_from_coin_gradient,
    measurement_tau_window,
    omega_bounce,
    omega_hopping,
    pi_pin_kappa,
    sigma_obs_geometry,
    strong_measurement_ratio,
    u_max_spring,
    C,
    E_CHARGE,
    HBAR,
    M_E,
    a_lm_coin,
    alpha_pressure_build,
    a0_wake_scale,
    bohr_radius_geometry,
    calibrate_c_b_h2,
    calibrate_f_part_dipole,
    cpl_from_sea_pressure,
    b_dipole_surface_t,
    b_neutron_star_dipole_t,
    double_well_u,
    flip_barrier_j,
    gravitational_time_dilation_factor,
    mu_cell_neutron_network,
    schwarzschild_radius,
    tau_flip_seconds,
    w_eos_from_sea_pressure,
    w_z_cpl,
    M_SUN,
    calibrate_pi0_for_w0,
    acceleration_condition_w,
    circular_velocity_squared,
    ell_c_from_rho_dm,
    enclosed_mass_flat_halo,
    e_spring_j,
    gamma_form_rate,
    gamma_sep_rate,
    gamma_unpin_arrhenius,
    k_sup_rayleigh,
    phi_ab_steady_state,
    q_transfer_rate,
    rho_dot_de_const,
    sector_split_fractions,
    sigma_det_class_a,
    sigma_gamma_dm,
    SIGMA_GAMMA_DM_EXP_UPPER,
    RHO_DM_HALO_LOCAL,
    omega_from_ev,
    R_EARTH,
    B_EARTH_SURFACE_T,
    M_NS_TYPICAL_KG,
    R_NS_TYPICAL_M,
    calibrate_g_e_for_lambda,
    calibrate_measurement_sg,
    calibrate_neutron_pressure,
    gamma_obs_for_t_escape,
    p_c_gap_joules,
    predict_neutron_escape,
    t_escape_pressure,
    demo_slit_fringe_intensity,
    e_bond_covalent_ev,
    hydrogen_energy_ev,
    interference_intensity,
    k_nl_geometry,
    lambda_he3_he4_ratio,
    path_mark_visibility,
    shell_capacity,
    subshell_capacity,
    sigma_wake_default,
    visibility_vs_pressure,
    chi_steady,
    chsh_s_quantum,
    classify_compression,
    classify_compression_from_coin,
    e_ref_from_geometry,
    kappa_bar_with_dm_path,
    kappa_wkb_from_h_x,
    transmission_soft_pipeline,
    u_bar_from_h_coin,
    g_eff_from_mu_n,
    k_fusion_from_r_pe,
    k_f_pin,
    lambda_n_from_sg,
    lattice_mass_multiplier,
    m_lat_from_active_network,
    r_pe_model_with_lattice,
    frame_width_n,
    width_descent_positive_finite,
    mixed_radix_address,
    geometric_refinement_time_total,
    motion_budget_residual,
    d_tau_dt_kinematic,
    gamma_lorentz,
    v_time_from_v_space,
    lambda_descent_rate,
    width_under_descent,
    f_clock_dm_coherent,
    s_clock_suppression,
    chain_cascade_weight,
    R_PE,
    length_energy_consistency_residual,
    l_p_min_spring,
    measurement_pin_probability,
    pi_pin_from_bias,
    r_pe_energy,
    r_pe_length,
    r_pe_spring_only,
    spin_projection_probability,
    t_eff_soft,
    t_wkb,
    xi_shred_with_dm,
    xi_shred_from_field,
    classify_spectrum,
    demo_white_vs_laser_spectra,
    mode_energy_ev,
    spectral_fill,
    visible_band_hz,
)


class TestNeutronCalibration(unittest.TestCase):
    def test_t_escape_matches_tau_n_scale_mode(self):
        c = calibrate_neutron_pressure(gap="scale")
        self.assertAlmostEqual(c.p_gap_j / c.dP_dt, c.tau_n, delta=1.0)

    def test_cavity_uses_c1_omega_b(self):
        c = calibrate_neutron_pressure(gap="cavity")
        self.assertAlmostEqual(c.omega_in0, omega_bounce(), places=3)
        self.assertAlmostEqual(c.r_coin, coin_half_width(), places=12)

    def test_alpha_from_geometry(self):
        a = alpha_pressure_build(1.0)
        self.assertAlmostEqual(a, HBAR * omega_bounce() * pi_pin_kappa(1.0), places=12)

    def test_t_escape_primary_law(self):
        p = p_c_gap_joules("dm_np")
        a = alpha_pressure_build(1.0)
        g = gamma_obs_for_t_escape(p, a, TAU_N)
        t = t_escape_pressure(p, a, g)
        self.assertAlmostEqual(t, TAU_N, delta=0.5)

    def test_predict_neutron_without_gamma_is_open(self):
        pred = predict_neutron_escape(p_c_kind="q_beta", kappa=1.0)
        self.assertIsNone(pred.t_escape_s)
        self.assertGreater(pred.alpha, 0.0)

    def test_predict_neutron_with_gamma_matches_tau(self):
        cal = calibrate_neutron_pressure(gap="cavity")
        pred = predict_neutron_escape(
            p_c_kind="dm_np", kappa=1.0, gamma_obs=cal.gamma_obs
        )
        self.assertAlmostEqual(pred.t_escape_s, TAU_N, delta=1.0)

    def test_g_eff_order(self):
        self.assertGreater(g_eff_from_mu_n(), 1.0e3)

    def test_k_fusion_near_one(self):
        self.assertGreater(k_fusion_from_r_pe(), 0.999)


class TestMeasurementGeometryStep5(unittest.TestCase):
    def test_e_obs_micro_uses_L0(self):
        b = 1.0e3
        self.assertAlmostEqual(
            e_obs_coupling_from_gradient(b),
            MU_B * b * coin_half_width(),
            places=12,
        )

    def test_strong_pin_ratio_with_calibrated_g_e(self):
        c = calibrate_measurement_sg()
        ratio = strong_measurement_ratio(c.b_grad_z, g_e=c.g_e, kappa=0.0)
        self.assertGreater(ratio, 1.0e6)

    def test_lambda_coin_gradient_positive(self):
        tau = measurement_tau_window(0.04, 1.0e5)
        self.assertGreater(lambda_n_from_coin_gradient(1.0e3, tau), 0.0)

    def test_kraus_suppression_at_strong_lambda(self):
        self.assertLess(kraus_decoherence_factor(5.0), 0.01)

    def test_joint_ripple_half_scale(self):
        a, b = 0.0, math.pi / 4
        self.assertAlmostEqual(
            bell_correlation_joint_ripple_linear(a, b),
            -0.5 * math.cos(a - b),
            places=9,
        )
        self.assertNotAlmostEqual(
            bell_correlation_joint_ripple_linear(a, b),
            bell_correlation_qm(a, b),
            places=2,
        )


class TestMeasurementSection5(unittest.TestCase):
    def test_cos_squared_law(self):
        self.assertAlmostEqual(spin_projection_probability(0.0), 1.0)
        self.assertAlmostEqual(spin_projection_probability(math.pi), 0.0, places=10)
        self.assertAlmostEqual(spin_projection_probability(math.pi / 2), 0.5)

    def test_mixed_is_half(self):
        self.assertAlmostEqual(spin_projection_probability(0.3, prep="mixed"), 0.5)

    def test_lambda_increases_with_field(self):
        lam_weak = lambda_n_from_sg(100.0, 0.04, 4e-7, g_e=1.0)
        lam_strong = lambda_n_from_sg(2000.0, 0.04, 4e-7, g_e=1.0)
        self.assertGreater(lam_strong, lam_weak)

    def test_g_e_calibration_hits_target_lambda(self):
        b, l, tau = 1.0e3, 0.04, 4e-7
        target = 5.0
        g = calibrate_g_e_for_lambda(target, b, l, tau)
        lam = lambda_n_from_sg(b, l, tau, g_e=g)
        self.assertAlmostEqual(lam, target, places=3)

    def test_strong_pin_probability(self):
        c = calibrate_measurement_sg()
        self.assertGreaterEqual(c.lambda_n, 4.9)
        self.assertGreater(c.pin_p, 0.49)

    def test_bounce_period_order(self):
        self.assertAlmostEqual(bounce_period(), 4.0 * coin_half_width() / C)


class TestEntanglementGeometryStep6(unittest.TestCase):
    def test_ell_c_is_compton(self):
        self.assertAlmostEqual(ell_c_from_geometry(), compton_wavelength())

    def test_sigma_obs_uses_L0(self):
        l0 = coin_half_width()
        self.assertAlmostEqual(sigma_obs_geometry(0.0), math.pi * l0**2)

    def test_j_ab_decays_with_distance(self):
        near = j_ab_overlap(0.0)
        far = j_ab_overlap(10.0 * ell_c_from_geometry())
        self.assertGreater(near, far)

    def test_coherence_steady_state(self):
        c = coherence_steady_state(2.0, 2.0)
        self.assertAlmostEqual(c, 0.5)

    def test_gamma_form_scales_with_phi(self):
        g0 = gamma_form_rate(0.0, phi_ab=0.0)
        g1 = gamma_form_rate(0.0, phi_ab=1.0)
        self.assertAlmostEqual(g0, 0.0)
        self.assertGreater(g1, 0.0)

    def test_bell_phi_fill_matches_qm_at_one(self):
        a, b = 0.0, math.pi / 4
        self.assertAlmostEqual(
            bell_correlation_phi_fill(a, b, 1.0),
            bell_correlation_qm(a, b),
            places=9,
        )

    def test_bell_half_fill_matches_joint_ripple(self):
        a, b = 0.3, 0.7
        self.assertAlmostEqual(
            bell_correlation_phi_fill(a, b, 0.5),
            bell_correlation_joint_ripple_linear(a, b),
            places=9,
        )


class TestBellGeometry(unittest.TestCase):
    def test_sign_half_plane_model_differs_from_qm(self):
        """External Block 5 sign integral is not yet a valid O5-3 closure."""
        qm = bell_correlation_qm(0.0, math.pi / 4)
        coin = bell_correlation_coin_geometry(0.0, math.pi / 4, n_samples=80_000)
        self.assertAlmostEqual(qm, -math.sqrt(2) / 2, places=6)
        self.assertNotAlmostEqual(coin, qm, places=1)

    def test_chsh_quantum_violation(self):
        s = chsh_s_quantum()
        self.assertAlmostEqual(abs(s), 2 * math.sqrt(2), places=3)
        self.assertGreater(abs(s), 2.0)


class TestAtomGeometryStep9(unittest.TestCase):
    def test_shell_capacity_n2(self):
        self.assertEqual(shell_capacity(2), 8)

    def test_subshell_p_capacity(self):
        self.assertEqual(subshell_capacity(1), 6)

    def test_hydrogen_ground_state(self):
        self.assertAlmostEqual(hydrogen_energy_ev(1), -RYDBERG_EV, places=3)

    def test_bohr_radius_much_larger_than_L0(self):
        self.assertGreater(bohr_radius_geometry() / coin_half_width(), 10.0)

    def test_k_nl_increases_with_n(self):
        self.assertGreater(k_nl_geometry(3, 0), k_nl_geometry(2, 0))

    def test_c_b_h2_calibration(self):
        c_b = calibrate_c_b_h2()
        self.assertGreater(c_b, 0.0)
        e = e_bond_covalent_ev(H2_R0_M, c_b)
        self.assertAlmostEqual(e, -H2_BOND_EV, delta=0.05)

    def test_a_lm_l0_normalized_order(self):
        val = a_lm_coin(0.0, 0.0, 0, 0)
        self.assertAlmostEqual(val, 1.0 / math.sqrt(4.0 * math.pi), places=6)


class TestCosmicGeometryStep10(unittest.TestCase):
    def test_schwarzschild_sun_order(self):
        rs = schwarzschild_radius(M_SUN)
        self.assertAlmostEqual(rs, 2953.0, delta=50.0)

    def test_earth_time_dilation_near_unity(self):
        dil = gravitational_time_dilation_factor(R_EARTH, 5.972e24)
        self.assertGreater(dil, 1.0 - 1e-9)
        self.assertLess(dil, 1.0)

    def test_double_well_minima_at_pm_one(self):
        u0 = 1.0
        self.assertAlmostEqual(double_well_u(1.0, u0), 0.0)
        self.assertAlmostEqual(double_well_u(-1.0, u0), 0.0)
        self.assertGreater(double_well_u(0.0, u0), 0.0)

    def test_tau_flip_increases_with_barrier(self):
        d = 1.0
        self.assertGreater(tau_flip_seconds(1.0, 2.0, d), tau_flip_seconds(1.0, 1.0, d))

    def test_w_eos_near_minus_one(self):
        w = w_eos_from_sea_pressure(1e-10, 1.0)
        self.assertAlmostEqual(w, -1.0, places=8)

    def test_cpl_w0_from_small_pi(self):
        w0, wa = cpl_from_sea_pressure(1e-10, 1.0, n=0.1)
        self.assertAlmostEqual(w0, -1.0 + 1e-10, places=12)
        self.assertAlmostEqual(w_z_cpl(w0, wa, 0.0), w0)

    def test_earth_f_part_calibration(self):
        mu_cell = mu_cell_neutron_network()
        v_core = 1.7e18
        n_n = 1.0e26
        f_part = calibrate_f_part_dipole(
            B_EARTH_SURFACE_T, R_EARTH, n_n, v_core, mu_cell
        )
        self.assertGreater(f_part, 0.0)
        self.assertLessEqual(f_part, 1.0)
        b = b_dipole_surface_t(f_part * n_n * v_core, mu_cell, R_EARTH)
        self.assertAlmostEqual(b, B_EARTH_SURFACE_T, delta=B_EARTH_SURFACE_T * 0.01)

    def test_neutron_star_b_field_large(self):
        b = b_neutron_star_dipole_t(M_NS_TYPICAL_KG, R_NS_TYPICAL_M, f_ns=1e-3)
        self.assertGreater(b, 1.0e7)


class TestDarkSectorGeometryStep11(unittest.TestCase):
    def test_phi_ab_steady_between_zero_and_one(self):
        phi = phi_ab_steady_state(2.0, 1.0)
        self.assertAlmostEqual(phi, 2.0 / 3.0)

    def test_k_sup_small_at_low_omega(self):
        e_s = e_spring_j()
        self.assertLess(k_sup_rayleigh(1.0e10, e_s), 1e-20)

    def test_sigma_gamma_dm_suppressed_vs_geom(self):
        omega_eV = omega_from_ev(1.0)
        sig = sigma_gamma_dm(omega_eV)
        geom = math.pi * coin_half_width() ** 2
        self.assertLess(sig, geom)
        self.assertLess(sig, SIGMA_GAMMA_DM_EXP_UPPER)

    def test_class_a_exact_null(self):
        self.assertEqual(sigma_det_class_a(), 0.0)

    def test_flat_halo_raises_velocity_with_radius(self):
        m0 = 1.0e40
        r0 = 1.0e4
        v1 = circular_velocity_squared(enclosed_mass_flat_halo(m0, r0, 2.0e4), 2.0e4)
        v2 = circular_velocity_squared(enclosed_mass_flat_halo(m0, r0, 4.0e4), 4.0e4)
        self.assertAlmostEqual(v1, v2, delta=1e-6 * v1)

    def test_gamma_sep_zero_at_zero_temperature(self):
        self.assertEqual(
            gamma_sep_rate(gamma_unpin_arrhenius(1e12, 1e-19, 0.0), 0.0), 0.0
        )

    def test_sector_fractions_sum_to_one(self):
        fe, fm = sector_split_fractions(1.0, 2.0)
        self.assertAlmostEqual(fe + fm, 1.0)
        self.assertAlmostEqual(fe, 1.0 / 3.0)

    def test_w_minus_one_rho_dot_zero(self):
        self.assertEqual(rho_dot_de_const(1.0, 2.2e-18), 0.0)

    def test_acceleration_condition(self):
        self.assertTrue(acceleration_condition_w(-0.9))
        self.assertFalse(acceleration_condition_w(-0.2))

    def test_phi_ab_modulates_gamma_form(self):
        lc = ell_c_from_rho_dm(RHO_DM_HALO_LOCAL)
        g0 = gamma_form_rate(1.0e-6, phi_ab=0.0, ell_c=lc)
        g1 = gamma_form_rate(1.0e-6, phi_ab=1.0, ell_c=lc)
        self.assertEqual(g0, 0.0)
        self.assertGreater(g1, 0.0)

    def test_ell_c_halo_much_larger_than_L0(self):
        self.assertGreater(ell_c_from_rho_dm(RHO_DM_HALO_LOCAL) / coin_half_width(), 1e3)


class TestDoubleSlitGeometryStep8(unittest.TestCase):
    def test_a0_wake_positive(self):
        self.assertGreater(a0_wake_scale(), 0.0)

    def test_interference_mu_zero_no_cross(self):
        a = 1.0 + 0.0j
        b = 1.0 + 0.0j
        self.assertAlmostEqual(interference_intensity(a, b, mu=0.0), 2.0)

    def test_opposite_phase_fringe_contrast(self):
        z = 0.05
        sep = 1.0e-6
        sig = sigma_wake_default(sep)
        i0 = demo_slit_fringe_intensity(0.0, z, sep, sigma_wake=sig, mu=1.0)
        i1 = demo_slit_fringe_intensity(sep / 3.0, z, sep, sigma_wake=sig, mu=1.0)
        self.assertGreater(i0, 0.0)
        rel = abs(i0 - i1) / max(i0, i1, 1.0e-30)
        self.assertGreater(rel, 0.05)

    def test_in_phase_higher_than_opposite_at_same_point(self):
        z = 0.05
        sep = 1.0e-6
        sig = sigma_wake_default(sep)
        y = sep / 5.0
        i_opp = demo_slit_fringe_intensity(y, z, sep, sigma_wake=sig, opposite_phase=True)
        i_same = demo_slit_fringe_intensity(y, z, sep, sigma_wake=sig, opposite_phase=False)
        self.assertGreater(i_same, i_opp)

    def test_path_mark_kills_visibility(self):
        self.assertLess(path_mark_visibility(1.0, 0.99), 0.1)

    def test_visibility_pressure_decays(self):
        self.assertGreater(visibility_vs_pressure(1.0, 1e-5, 0.0), visibility_vs_pressure(1.0, 1e-5, 1e5))

    def test_lambda_he3_he4_not_equal(self):
        self.assertGreater(lambda_he3_he4_ratio(), 1.5)


class TestTunnelingGeometryStep7(unittest.TestCase):
    def test_u_bar_positive(self):
        self.assertGreater(u_bar_from_h_coin(0.5), 0.0)

    def test_e_ref_is_hbar_omega_b(self):
        self.assertAlmostEqual(e_ref_from_geometry(), HBAR * omega_bounce(), places=6)

    def test_kappa_wkb_increases_with_compression(self):
        e = 1.0e3 * E_CHARGE
        k_lo = kappa_wkb_from_h_x(e, kappa=0.2)
        k_hi = kappa_wkb_from_h_x(e, kappa=0.8)
        self.assertGreaterEqual(k_hi, k_lo)

    def test_p7_2_soft_vs_hard_from_coin(self):
        self.assertEqual(
            classify_compression_from_coin(0.05), CompressionRegime.SOFT
        )
        self.assertEqual(
            classify_compression_from_coin(1.0), CompressionRegime.HARD
        )

    def test_dm_fill_increases_transmission(self):
        chi = 0.5
        l_bar = 1.0e-10
        k_fill = kappa_bar_with_dm_path(1.0e8, 1.0, eta_kappa=0.5)
        k_empty = kappa_bar_with_dm_path(1.0e8, 0.0, eta_kappa=0.5)
        t_fill = t_eff_soft(k_fill, l_bar, chi)
        t_empty = t_eff_soft(k_empty, l_bar, chi)
        self.assertGreater(t_fill, t_empty)

    def test_kappa_dm_unfilled_harder(self):
        k0 = kappa_bar_with_dm_path(1.0e9, 1.0, eta_kappa=0.5)
        k1 = kappa_bar_with_dm_path(1.0e9, 0.0, eta_kappa=0.5)
        self.assertGreater(k1, k0)


class TestTunnelingP72(unittest.TestCase):
    def test_soft_vs_hard_regime(self):
        self.assertEqual(classify_compression(0.1), CompressionRegime.SOFT)
        self.assertEqual(classify_compression(0.99), CompressionRegime.HARD)

    def test_pi_pin_limits(self):
        self.assertAlmostEqual(pi_pin_from_bias(0.0, 1.0), 0.0)
        self.assertAlmostEqual(pi_pin_from_bias(1.0, 0.0), 1.0)

    def test_dm_fill_reduces_xi_shred(self):
        base = xi_shred_from_field(1.0, 1.0)
        reduced = xi_shred_with_dm(1.0, base, eta_dm=0.5)
        self.assertLess(reduced, base)

    def test_t_eff_bounded_by_wkb(self):
        tw = t_wkb(1.0, 0.5)
        te = t_eff_soft(1.0, 0.5, chi_ss=0.4)
        self.assertAlmostEqual(te, tw * 0.4)

    def test_chi_steady(self):
        self.assertAlmostEqual(chi_steady(3.0, 1.0), 0.75)


class TestCoinGeometryC1(unittest.TestCase):
    def test_L_is_half_compton(self):
        l = coin_half_width()
        self.assertAlmostEqual(l, compton_wavelength() / 2.0)

    def test_omega_b_from_L(self):
        l = coin_half_width()
        self.assertAlmostEqual(omega_bounce(), math.pi * C / (2.0 * l))

    def test_k_s_from_omega(self):
        ob = omega_bounce()
        self.assertAlmostEqual(k_s_from_geometry(), M_E * ob**2)

    def test_bounce_period_four_L_over_c(self):
        self.assertAlmostEqual(bounce_period(), 4.0 * coin_half_width() / C)

    def test_pi_pin_high_at_full_compression(self):
        self.assertGreater(pi_pin_kappa(1.0), 0.9)

    def test_u_max_order_meV(self):
        ev = u_max_spring() / E_CHARGE
        self.assertGreater(ev, 1.0e5)
        self.assertLess(ev, 1.0e7)


class TestFusionGeometryStep3(unittest.TestCase):
    def test_k_f_pin_between_zero_and_one(self):
        k = k_f_pin()
        self.assertGreater(k, 0.3)
        self.assertLess(k, 0.5)

    def test_delta_omega_balance_at_k_f_pin(self):
        k = k_f_pin()
        d = delta_kappa(k)
        o = omega_hopping(k)
        self.assertAlmostEqual(d / o, 1.0, places=6)
        self.assertAlmostEqual(pi_pin_kappa(k), 0.5, places=6)

    def test_r_pe_spring_only_is_pi_sq_over_8(self):
        self.assertAlmostEqual(r_pe_spring_only(), math.pi**2 / 8.0, places=9)

    def test_l_p_min_ratio(self):
        l0 = coin_half_width()
        self.assertAlmostEqual(l_p_min_spring() / l0, 8.0 / math.pi**2, places=9)

    def test_length_energy_only_match_at_zero(self):
        for k in (0.0, 0.2, 0.41, 0.9):
            if k == 0.0:
                self.assertAlmostEqual(length_energy_consistency_residual(k), 0.0, places=9)
            else:
                self.assertNotAlmostEqual(length_energy_consistency_residual(k), 0.0, places=2)

    def test_r_pe_energy_at_k_pin_order_unity(self):
        r = r_pe_energy(k_f_pin())
        self.assertGreater(r, 1.5)
        self.assertLess(r, 2.5)

    def test_lattice_multiplier_order(self):
        m = m_lat_from_active_network()
        self.assertGreater(m, 1400.0)
        self.assertLess(m, 1600.0)
        r_pred = r_pe_model_with_lattice()
        self.assertGreater(r_pred, 1800.0)
        self.assertLess(r_pred, 1900.0)
        # E-check ratio matches legacy helper
        self.assertAlmostEqual(m, lattice_mass_multiplier(), delta=30.0)


class TestTimeZenoGeometryStep12(unittest.TestCase):
    def test_finite_width_never_zero(self):
        self.assertTrue(width_descent_positive_finite(1.0, (2, 3, 5, 7)))

    def test_motion_budget_zero_residual(self):
        self.assertAlmostEqual(motion_budget_residual(0.6 * C), 0.0, delta=1.0)

    def test_gamma_at_half_c(self):
        self.assertAlmostEqual(d_tau_dt_kinematic(0.5 * C), math.sqrt(3) / 2, places=6)

    def test_geometric_time_converges(self):
        t = geometric_refinement_time_total(1.0, 0.5, 100)
        self.assertAlmostEqual(t, 2.0, places=3)

    def test_mixed_radix_in_unit_interval(self):
        x = mixed_radix_address((1, 1, 1), (2, 2, 2))
        self.assertGreater(x, 0.0)
        self.assertLess(x, 1.0)

    def test_width_under_descent_positive(self):
        w = width_under_descent(1.0, 1.0 / TAU_N, 1.0)
        self.assertGreater(w, 0.0)
        self.assertLess(w, 1.0)

    def test_dm_coherent_clock_null(self):
        self.assertEqual(f_clock_dm_coherent(), 0.0)

    def test_s_clock_suppressed(self):
        self.assertLess(s_clock_suppression(300.0), 1e-20)

    def test_chain_cascade_weight_positive(self):
        self.assertGreater(chain_cascade_weight((2, 3, 5)), 1.0)

    def test_v_time_from_v_space_pythagoras(self):
        v_s = 0.8 * C
        v_t = v_time_from_v_space(v_s)
        self.assertAlmostEqual(v_s**2 + v_t**2, C**2, delta=1.0)


class TestVaporSpectrumP1v(unittest.TestCase):
    def test_mode_energy_red_light_order(self):
        # ~500 THz ~ 2 eV order
        e = mode_energy_ev(5.0e14)
        self.assertGreater(e, 1.5)
        self.assertLess(e, 3.0)

    def test_laser_vs_white_class(self):
        laser, white = demo_white_vs_laser_spectra()
        self.assertEqual(classify_spectrum(laser), "fundamental_packet")
        self.assertEqual(classify_spectrum(white), "partial_or_full_spectrum")

    def test_white_higher_fill_on_visible_band(self):
        laser, white = demo_white_vs_laser_spectra()
        f_lo, f_hi = visible_band_hz()
        phi_laser = spectral_fill(laser, f_lo, f_hi, reference=white)
        phi_white = spectral_fill(white, f_lo, f_hi, reference=white)
        self.assertAlmostEqual(phi_white, 1.0)
        self.assertLess(phi_laser, phi_white)


def run_tests() -> int:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
