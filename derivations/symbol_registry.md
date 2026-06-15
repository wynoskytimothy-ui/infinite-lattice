# AETHOS Symbol Registry

Single source of truth for symbols used across section derivations.


| Symbol                          | Meaning                                                   | Units                             | First section | Notes                                                 |
| ------------------------------- | --------------------------------------------------------- | --------------------------------- | ------------- | ----------------------------------------------------- |
| `h`                             | Planck constant                                           | J·s                               | 1             | `ħ = h/(2π)`                                          |
| `c`                             | Speed of light                                            | m/s                               | 1             | Motion budget constant                                |
| `E`                             | Energy                                                    | J                                 | 1             | `E=hf`, `E=pc` for photon                             |
| `p`                             | Momentum                                                  | kg·m/s                            | 1             | `p=h/λ`                                               |
| `f`                             | Frequency                                                 | Hz                                | 1             | Pump or wave frequency by context                     |
| `λ`                             | Wavelength                                                | m                                 | 1             | Disturbance wavelength                                |
| `ψ`                             | Disturbance amplitude                                     | normalized                        | 1             | Interpretation depends on section                     |
| `ρ`                             | Probability density (`                                    | ψ                                 | ²`)           | 1/volume                                              |
| `a_λ`, `a(f)`                   | Sea mode amplitude (vapor quantum weight)                 | field amp.                        | 1             | **P1-v**; `|a_\lambda|^2` → intensity                 |
| `B_full`                        | Spectral band for “full packet”                           | Hz (or ω)                         | 1             | `[f_min,f_max]` model band limit                      |
| `φ_B`                           | Spectral fill fraction on `B_full`                        | 0..1                              | 1             | `\phi_B=1` = full band structure                      |
| `Ψ_sea`                         | Sea disturbance `\sum_\lambda a_\lambda \epsilon_\lambda` | —                                 | 1             | Vapor decomposition                                   |
| `τ`                             | Proper time                                               | s                                 | 1             | Internal clock measure                                |
| `λ_C`                           | Full Compton wavelength                                   | m                                 | 1             | `h/(m_e c)`; lattice **cell** width                   |
| `ƛ_C`                           | Reduced Compton wavelength                                | m                                 | 1             | `ħ/(m_e c) = λ_C/(2π)`                                |
| `L`                             | Coin half-width (C1)                                      | m                                 | 1/2           | `λ_C/2`; bounce cavity — not cell width               |
| `t_cell`                        | Cell crossing time                                        | s                                 | 1             | `λ_C/c = h/(m_e c²)`                                  |
| `T_bounce`                      | Full electron pump period                                 | s                                 | 2             | `4L/c = 2 t_cell`                                     |
| `f_b`                           | Bounce / pump frequency                                   | Hz                                | 2             | `1/T_bounce = m_e c²/(2h)`                            |
| `A_k,B_k,C_k`                   | π-lattice legs at level `k`                               | dimensionless                     | 1             | Book/code map in Sec 1.10.5                           |
| `N_k`                           | Inscribed polygon vertex count                            | integer                           | 1             | `4·2^k`                                               |
| `γ`                             | Lorentz factor                                            | dimensionless                     | 1/2           | `1/sqrt(1-v²/c²)`                                     |
| `Γ_form`                        | Entanglement formation rate                               | 1/s                               | 6             | Model-fit parameter                                   |
| `Γ_break`                       | Entanglement break/decoherence rate                       | 1/s                               | 6             | Includes photon-observation effects                   |
| `C(t)`                          | Coherence/entanglement strength                           | 0..1                              | 6             | Dynamic state variable                                |
| `Φ_env`                         | Environmental photon flux                                 | 1/(m²·s)                          | 6             | Decoherence driver                                    |
| `σ_obs`                         | Observation/compression cross-section                     | m²                                | 6             | Effective parameter                                   |
| `G`                             | Gravitational constant                                    | SI                                | 10            | Gravity relations                                     |
| `r_s`                           | Schwarzschild radius                                      | m                                 | 10/12         | `2GM/c²`                                              |
| `rho_DM`                        | Dark matter mass density                                  | kg/m³                             | 11            | Gravitating nonluminous component                     |
| `rho_DE`                        | Dark energy density                                       | J/m³ (or equivalent mass density) | 11            | Accelerating-expansion component                      |
| `w`                             | Equation-of-state parameter `p/(rho c²)`                  | dimensionless                     | 10/11         | Cosmology acceleration criterion                      |
| `Q`                             | Effective transfer rate between normal/dark sectors       | density/time                      | 11            | Interacting-sector model parameter                    |
| `w_n`                           | Frame width at descent depth `n`                          | normalized interval length        | 12            | `w_n = w_0/prod p_k`                                  |
| `p_k`                           | Prime used at descent level `k`                           | integer                           | 12            | Subdivision radix                                     |
| `i_k`                           | Child index at descent level `k`                          | integer                           | 12            | `0 <= i_k <= p_k-1`                                   |
| `x_n`                           | Position after `n` descent levels                         | normalized coordinate             | 12            | `x_n = sum i_k/(prod p_j)`                            |
| `Delta t_n`                     | Time slice at refinement level `n`                        | s                                 | 12            | Geometric schedule in Zeno sum                        |
| `v_space`                       | Velocity component through space                          | m/s                               | 12            | Motion-budget form                                    |
| `v_time`                        | Velocity component through internal-time direction        | m/s (effective)                   | 12            | `v_time = c/gamma`                                    |
| `v_flow`                        | Effective gravitational flow speed                        | m/s                               | 10/12         | `sqrt(2GM/r)`                                         |
| `H_coin`                        | Effective electron coin Hamiltonian                       | J                                 | 2             | Minimal two-level trapped-photon dynamics             |
| `Delta(kappa)`                  | Side-energy asymmetry under compression state `kappa`     | rad/s (as `Delta`), or J/`hbar`   | 2             | Bias term in `H_coin`                                 |
| `Omega(kappa)`                  | Inter-side bounce/tunneling coupling                      | rad/s                             | 2             | Mixing term in `H_coin`                               |
| `E_obs(t)`                      | Observation/compression channel drive amplitude           | field-dependent                   | 2/5           | External measurement drive                            |
| `g_E`                           | Observation-drive coupling coefficient                    | model-dependent                   | 2             | Maps `E_obs` to coin bias                             |
| `M_{s,n}`                       | Kraus operator for outcome `s` along axis `n`             | dimensionless operator            | 5             | Effective measurement channel                         |
| `Lambda_n`                      | Integrated measurement strength on axis `n`               | dimensionless                     | 5             | `2 integral Gamma_n(t) dt`                            |
| `Gamma_n(t)`                    | Axis-resolved dephasing/compression rate                  | 1/s                               | 5/6           | Driven by observation channel                         |
| `A_eff`                         | Effective geometric interaction area of coin              | m²                                | 6             | `pi R_coin^2`                                         |
| `R_coin`                        | Effective coin radius                                     | m                                 | 6             | Geometry scale parameter                              |
| `S_res(omega)`                  | Spectral resonance overlap factor                         | dimensionless                     | 6             | Lorentzian probe/pump overlap                         |
| `Gamma_2`                       | Effective linewidth/dephasing parameter                   | 1/s                               | 6             | Sets resonance width                                  |
| `O_AB`                          | Pair phase-lock order parameter                           | dimensionless complex             | 6             | `⟨exp(i(phi_A-phi_B-pi))⟩`                            |
| `J_AB`                          | Pair overlap/matching envelope factor                     | dimensionless                     | 6             | Includes distance/frequency/axis matching             |
| `k_lock`                        | Phase-lock formation rate scale                           | 1/s                               | 6             | Prefactor in `Gamma_form` closure                     |
| `ell_c`                         | Effective coherence length for locking                    | m                                 | 6             | Appears in `exp(-d/ell_c)`; may track DM mesh (P11-3) |
| `phi_AB`                        | DM ripple fill fraction on path A→B                       | 0..1                              | 6,11          | Continuous string fill (P11-3)                        |
| `Gamma_fill`                    | DM channel fill rate                                      | 1/s                               | 11            | `d phi/dt` growth term                                |
| `Gamma_snap`                    | DM filament snap / drain rate                             | 1/s                               | 11            | Reduces `phi`                                         |
| `eta_DM`                        | DM assist factor in soft tunneling                        | dimensionless                     | 7             | Modifies `xi_shred` (Sec 7.3.4)                       |
| `chi`                           | Post-barrier compactness order parameter                  | 0..1                              | 7             | `chi=1` recaptured pump                               |
| `k_s`                           | Spring stiffness constant                                 | N/m (effective)                   | 2             | Hooke law `T=k_s u`                                   |
| `u(x,t)`                        | Spring elongation field along coin axis                   | m                                 | 2             | Mechanical displacement                               |
| `T(x,t)`                        | Spring tension field                                      | N (effective)                     | 2             | `T=k_s u`; detection proxy                            |
| `alpha`                         | Tension-amplitude coupling constant                       | model-dependent                   | 2             | `u=alpha Re(psi)`                                     |
| `U_spring`                      | Stored spring energy                                      | J                                 | 2             | `(1/(2k_s)) int T^2 dx`                               |
| `K_f`                           | Fusion threshold compression                              | dimensionless                     | 3             | `K>=K_f` triggers fused branch                        |
| `k_el`                          | Elastic compression stiffness                             | model units                       | 3             | Pre-fusion spring constant                            |
| `k_pl`                          | Post-fusion stiffness                                     | model units                       | 3             | `k_pl >> k_el`                                        |
| `U_lock`                        | Fusion locking energy offset                              | J                                 | 3             | Added in fused branch potential                       |
| `Delta_f`                       | Fused-state bias splitting                                | rad/s (as `Delta`)                | 3             | Post-fusion `H_p` term                                |
| `M_K`                           | Effective mass of compression mode                        | kg (effective)                    | 3             | Conjugate to `P_K`                                    |
| `P_K`                           | Compression momentum                                      | kg·m/s (effective)                | 3             | Conjugate to `K`                                      |
| `Gamma_fuse`                    | Fusion transition rate                                    | 1/s                               | 3             | Arrhenius-like barrier crossing                       |
| `R_pe`                          | Proton/electron mass ratio                                | dimensionless                     | 3             | `m_p/m_e`; model form `1/(1-alpha K_f)`               |
| `L_0`                           | Open coin span scale                                      | m                                 | 3             | Electron-side maximum length                          |
| `L_p`                           | Fused coin span scale                                     | m                                 | 3             | `L_0(1-alpha K_f)`                                    |
| `alpha`                         | Compression-to-length coupling                            | dimensionless                     | 3             | In `L(K)=L_0(1-alpha K)`                              |
| `K_0`                           | Open-state compression baseline                           | dimensionless                     | 3             | Electron reference (~0)                               |
| `Phi_d`                         | Proton drain potential                                    | J/C (effective)                   | 3             | `Q_p/(4 pi eps_eff r)`                                |
| `Phi_e`                         | Electron pump potential                                   | J/C (effective)                   | 3             | Outward source potential                              |
| `epsilon_eff`                   | Effective sea permittivity                                | F/m                               | 3             | Leading map `epsilon_0`                               |
| `chi_sea`                       | Sea response factor                                       | dimensionless                     | 3             | `epsilon_eff = epsilon_0 chi_sea`                     |
| `Q_p`, `Q_e`                    | Effective drain/pump source charges                       | C (effective)                     | 3             | `Q_p=+Ze`, `Q_e=-e`                                   |
| `rho_q`                         | Effective source density in sea Poisson law               | C/m³                              | 3             | Drives `Phi`                                          |
| `k_e^eff`                       | Effective Coulomb constant                                | N·m²/C²                           | 3             | `1/(4 pi epsilon_eff)`                                |
| `P`                             | Neutron trapped-electron pressure variable                | J (effective)                     | 4             | Excess stored spring+bias energy                      |
| `P_c`                           | Pressure escape threshold                                 | J (effective)                     | 4             | Decay trigger level                                   |
| `P_0`                           | Pressure reference offset                                 | J (effective)                     | 4             | Free-electron baseline                                |
| `Pi_pin`                        | Pinning strength factor                                   | 0..1                              | 4             | `|Delta_eff|/(|Delta_eff|+Omega)`                     |
| `Delta_eff`                     | Effective side bias under outer observation               | rad/s (as `Delta`)                | 4             | `Delta(kappa)+2 g_obs E_bar`                          |
| `g_obs`                         | Outer-photon compression coupling                         | model-dependent                   | 4             | Drives neutron observation term                       |
| `E_bar`                         | Mean outer observation field                              | field-dependent                   | 4             | Constant-observation limit                            |
| `omega_in0`                     | Inner pump reference frequency                            | rad/s                             | 4             | Trapped electron baseline                             |
| `alpha`                         | Pressure-build rate coefficient                           | J                                 | 4             | `hbar omega_in0 Pi_pin`                               |
| `beta`                          | Pressure-relief coefficient                               | 1/s                               | 4             | Network sharing term scale                            |
| `R_share`                       | Network pressure-sharing relief rate                      | 1/s                               | 4             | `eta N C_N` in nuclei                                 |
| `C_N`                           | Nuclear trapped-electron sharing factor                   | dimensionless                     | 4/9           | Function of `(N,Z,A)`                                 |
| `C_0`                           | Sharing amplitude scale                                   | dimensionless                     | 4             | Leading `C_N` prefactor                               |
| `N_0`                           | Neutron-count turn-on scale                               | dimensionless                     | 4             | In `1-exp(-N/N_0)`                                    |
| `N-Z_0`                         | Stability-valley width for `N-Z`                          | dimensionless                     | 4             | Gaussian width parameter                              |
| `R_0`                           | Radius damping scale                                      | m (or fm)                         | 4             | In `exp(-R_0/R_A)`                                    |
| `r_0`                           | Nuclear radius constant                                   | m (or fm)                         | 4/9           | `R_A=r_0 A^(1/3)`                                     |
| `b_net`                         | Network binding correction scale                          | MeV (fit)                         | 4/9           | `B_share=-b_net N C_N`                                |
| `eta`                           | Sharing coupling efficiency                               | 1/s per unit                      | 4             | In `R_share=eta N C_N`                                |
| `U_bar`                         | Effective local barrier energy scale                      | J                                 | 7             | From coin+spring+`Delta_eff`                          |
| `V_bar`                         | Barrier compression drive potential                       | J                                 | 7             | `hbar g_E E_bar sigma_z`                              |
| `E_bar`                         | Barrier-region compression field                          | field-dependent                   | 7             | Drives `Delta_eff`                                    |
| `xi_shred`                      | Shredding fraction                                        | 0..1                              | 7             | Reduces `Omega`, raises `m_eff`                       |
| `lambda_m`                      | Shredding mass-coupling constant                          | dimensionless                     | 7             | In `M(xi)=1+lambda_m xi`                              |
| `E_ref`                         | Shredding reference field scale                           | field-dependent                   | 7             | Normalizes `xi_shred`                                 |
| `m_eff`                         | Effective tunneling mass                                  | kg                                | 7             | State-dependent in barrier                            |
| `A_L`, `A_R`                    | Left/right sea wake amplitudes                            | field amplitude                   | 8             | `A_s = A_0 K_s e^{i phi_s}`                           |
| `S_s`                           | Slit-local sea source density                             | amp²/volume                       | 8             | `eta_wake                                             |
| `G`                             | Sea propagation kernel                                    | 1/length                          | 8             | Retarded Green envelope                               |
| `eta_wake`                      | Pump-to-sea coupling efficiency                           | dimensionless                     | 8             | Wake amplitude prefactor                              |
| `ell_wake`                      | Wake spatial damping length                               | m                                 | 8             | In Green exponential                                  |
| `sigma_wake`                    | Slit wake transverse width                                | m                                 | 8             | Gaussian factor in `K_s`                              |
| `phi_L`, `phi_R`                | Left/right pump phases                                    | rad                               | 8             | `phi_R=phi_L+pi` when locked                          |
| `K_L`, `K_R`                    | Left/right geometric wake kernels                         | 1/length                          | 8             | Slit-to-detector envelope                             |
| `mu_cell`                       | Effective magnetic moment per neutron cell                | A·m²                              | 10            | `g_eff mu_B <sigma_z> Pi_pin` avg                     |
| `g_eff`                         | Effective g-factor for trapped-electron moment            | dimensionless                     | 10            | Magnetization scale                                   |
| `N_eff`                         | Participating neutron count in core network               | dimensionless                     | 10            | `f_part n_n V_c`                                      |
| `f_part`                        | Participation fraction (planet/core)                      | 0..1                              | 10            | Coherent network fraction                             |
| `f_NS`                          | Neutron-star participation fraction                       | 0..1                              | 10            | Compact-object network fraction                       |
| `xi_NS`                         | Neutron-star moment coupling factor                       | dimensionless                     | 10            | Core pinning/alignment scale                          |
| `m(t)`                          | Global magnetic orientation order parameter               | [-1,1]                            | 10            | Dipole polarity state                                 |
| `n_n`                           | Neutron number density in core region                     | 1/m³                              | 10            | Coarse-grain input                                    |
| `V_c`                           | Core volume                                               | m³                                | 10            | Region integrated for `N_eff`                         |
| `P_core`                        | Aggregate core trapped-electron pressure                  | J (effective)                     | 4/10          | Geodynamo stress variable                             |
| `P_eq`                          | Core pressure equilibrium offset                          | J (effective)                     | 10            | Reference in `Delta U(P)`                             |
| `Gamma_obs,core`                | Core observation/decoherence rate                         | 1/s                               | 10            | `sigma_obs Phi_obs,core`                              |
| `alpha_core`                    | Core pressure-build coefficient                           | J                                 | 4/10          | `hbar omega_in0 Pi_pin` average                       |
| `beta_core`                     | Core pressure-relief coefficient                          | 1/s                               | 4/10          | Network sharing scale                                 |
| `eta_core`                      | Core sharing efficiency                                   | 1/s per unit                      | 10            | In `R_share,core`                                     |
| `zeta_P`                        | Pressure-to-barrier coupling                              | J⁻¹                               | 10            | Lowers `Delta U` as `P_core` rises                    |
| `Delta U_0`                     | Baseline reversal barrier                                 | J                                 | 10            | Kramers barrier scale                                 |
| `D_core`                        | Core fluctuation/noise scale                              | J                                 | 10            | Thermal + observation variance                        |
| `nu_th`                         | Core thermal attempt factor                               | 1/s                               | 10            | Enters `D_core`                                       |
| `chi_D`                         | Observation-noise coupling                                | J·s                               | 10            | Scales `Var(Gamma_obs,core)`                          |
| `tau_flip`                      | Magnetic reversal timescale                               | s                                 | 10            | Paleomagnetic statistic target                        |
| `P_sea,DE`                      | Dark-energy sea pressure                                  | Pa (effective)                    | 10/11         | `(1/3) u_gamma,free`                                  |
| `u_gamma,free`                  | Freed-photon energy density                               | J/m³                              | 10/11         | `n_gamma hbar<omega>`                                 |
| `n_gamma,free`                  | Freed inner-photon number density                         | 1/m³                              | 10/11         | Driven by `f_e Q`                                     |
| `Pi_s(z)`                       | Sea-pressure fluctuation term                             | Pa                                | 10/11         | Perturbs `w(z)` away from -1                          |
| `w_0`, `w_a`                    | CPL equation-of-state fit params                          | dimensionless                     | 10/11         | `w=w_0+w_a z/(1+z)`                                   |
| `A(x)`                          | Static metric time component                              | dimensionless                     | 12            | `g_00=-A c^2`                                         |
| `Phi(x)`                        | Newtonian potential field                                 | J/kg                              | 12            | Weak-field limit of `A`                               |
| `chi(r,theta)`                  | Frame-drag budget factor                                  | dimensionless                     | 12            | Kerr-type extension sketch                            |
| `sigma_gammaDM`                 | Photon–dark-matter effective cross-section                | m²                                | 11            | `sigma_geom K_sup`                                    |
| `sigma_geom`                    | Geometric spring cross-section                            | m²                                | 11            | `pi R_spring^2`                                       |
| `K_sup`                         | Suppression kernel (no inner photon)                      | dimensionless                     | 11            | `(hbar omega/E_spring)^4`                             |
| `S_res,DM`                      | DM resonance overlap factor                               | dimensionless                     | 11            | Set to `0` in model                                   |
| `q_spring`                      | Effective spring–field coupling charge                    | C (effective)                     | 11            | `~ e/L_spring`                                        |
| `m_spring`                      | Effective spring-mode mass                                | kg                                | 11            | Enters `E_spring`                                     |
| `R_spring`                      | Effective spring radius                                   | m                                 | 11            | Geometric scale in `sigma_geom`                       |
| `sigma_max^exp`                 | Experimental cross-section bound                          | m²                                | 11            | Falsifier cap                                         |
| `Q`                             | NM→DM/DE transfer source term                             | J/m³/s                            | 11            | `rho_NM Gamma_sep bar E_gamma /(m_N c^2)`             |
| `Gamma_sep`                     | Spring–photon separation rate                             | 1/s                               | 11            | `Gamma_unpin-Gamma_pin`                               |
| `Gamma_pin`                     | Inner-photon pinning rate                                 | 1/s                               | 11            | Section 5/6 channel                                   |
| `Gamma_unpin`                   | Environment-driven unpin rate                             | 1/s                               | 11            | Arrhenius form                                        |
| `Delta_sep`                     | Separation barrier energy                                 | J                                 | 11            | `~ hbar omega_b` scale                                |
| `nu_0`                          | Attempt frequency for unpin                               | 1/s                               | 11            | Microscopic prefactor                                 |
| `f_e`, `f_m`                    | DE/DM energy split fractions                              | dimensionless                     | 11            | `f_m+f_e=1`                                           |
| `E_N`, `E_spring`, `E_gamma,in` | Event energy bookkeeping                                  | J                                 | 11            | One-unit split                                        |
| `f_clock^DM,coh`                | Coherent DM internal clock rate                           | Hz                                | 12            | Set to `0` in model                                   |
| `f_clock^DM,therm`              | Thermal tail DM clock rate                                | Hz                                | 12            | Arrhenius-like bound                                  |
| `f_clock^NM`                    | Normal-matter internal clock                              | Hz                                | 12            | `omega_b/(2pi)`                                       |
| `S_clock`                       | DM/NM clock suppression factor                            | dimensionless                     | 12            | `f_clock^DM,therm/f_clock^NM`                         |
| `v_time^DM`                     | DM motion-budget time component                           | m/s                               | 12            | `2 pi L_eff f_clock^DM`                               |
| `N_obs`                         | Observation-number bookkeeping                            | dimensionless                     | 4             | Conserved in `\nu` map                                |
| `Q_beta`                        | Beta-decay Q value                                        | J                                 | 4             | `(m_n-m_p-m_e)c^2`                                    |
| `g_eff`                         | Effective neutron moment factor                           | dimensionless                     | 4             | `\mu_n=-g_{eff}(2m_e/m_n)\mu_N`                       |
| `Lambda_leak`                   | Proton-screening leakage factor                           | dimensionless                     | 4             | Magnetic-moment bridge                                |
| `Gamma_partner`                 | Partner acquisition rate                                  | 1/s                               | 8             | `\sum n_i\sigma_i v_i p_i`                            |
| `f_coin,i`                      | Coin-availability factor                                  | dimensionless                     | 8             | Species `i`                                           |
| `sigma_e,i`                     | Entanglement cross-section                                | m²                                | 8             | Species `i`                                           |
| `chi`                           | Coin compactness order parameter                          | dimensionless                     | 7             | Recoherence dynamics                                  |
| `Gamma_rec`, `Gamma_sh`         | Recoherence/shred rates                                   | 1/s                               | 7             | Barrier exit                                          |
| `tau_rec`                       | Recoherence timescale                                     | s                                 | 7             | `1/(\Gamma_{rec}+\Gamma_{sh})`                        |
| `T_eff`                         | Effective transmission                                    | dimensionless                     | 7             | `T_{WKB}\chi_{ss}`                                    |
| `eta_share`                     | Entanglement-cost sharing fraction                        | dimensionless                     | 7             | Partner decoherence                                   |
| `Delta V`                       | Visibility shift from sharing                             | dimensionless                     | 7             | Partner discriminator                                 |
| `tau_re`                        | Re-entanglement recovery time                             | s                                 | 8             | `1/(Gamma_form+Gamma_break)`                          |
| `v_re`                          | Re-entanglement speed proxy                               | 1/s                               | 8             | `1/tau_re`                                            |
| `N_dof`                         | Internal mode count (molecule)                            | dimensionless                     | 8             | Fullerene scaling                                     |
| `eta_AB`                        | Bond overlap amplitude                                    | dimensionless                     | 9             | `\int psi_A^* psi_B d^3r`                             |
| `C_b`                           | Covalent coupling constant                                | dimensionless                     | 9             | Bond-energy closure                                   |
| `A_lm`                          | Angular envelope on coin shell                            | —                                 | 9             | From phase winding                                    |
| `chi`                           | Topological charge chirality                              | dimensionless                     | 2             | `q=q_0 chi`                                           |
| `q_0`                           | Charge unit magnitude                                     | C                                 | 2             | `e` scale                                             |
| `f_clock^p`                     | Proton internal clock rate                                | Hz                                | 3             | Set to `0` in model                                   |
| `S_p(omega)`                    | Proton clock sideband power                               | —                                 | 3             | Null discriminator                                    |
| `lambda_desc`                   | Prime-descent rate                                        | 1/s                               | 12            | `Gamma_obs E[log p]`                                  |
| `P_max`                         | Max radix prime in split                                  | dimensionless                     | 12            | Detector resolution                                   |
| `T_S`, `k_S`                    | Sea tension field/stiffness                               | N/m                               | 1             | Born map on `S`                                       |
| `u_S`, `rho_S`                  | Sea energy/vacuum density                                 | J/m³                              | 1             | Cutoff integral                                       |
| `Pi_vac`                        | Locked-mode vacuum fraction                               | dimensionless                     | 1             | `\rho_Lambda` interface                               |
| `L_cell`                        | Sea cell cutoff scale                                     | m                                 | 1             | UV regularization                                     |


