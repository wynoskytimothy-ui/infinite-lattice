# AETHOS Cross-Reference Matrix (Sections 1–12)

Rows = source section. Columns = sections that must be rechecked when source changes.

| Source | Recheck targets | Why |
|---|---|---|
| 1 Photon sea | 2–12 | Base ontology, photon math, causality assumptions |
| 2 Electron | 3–12 | Pump/clock mechanism reused everywhere |
| 3 Proton | 4,9,10,11 | Fusion, drain mechanics, mass/charge structure |
| 4 Neutron | 9,10,11 | Trapped layers, pressure, decay and stability |
| 5 Measurement | 6,7,8,9,11,12 | Observation/decoherence definitions |
| 6 Entanglement | 8,9,10 | Correlation, pairing limits, coherence loss |
| 7 Tunneling | 8,9,10 | Barrier scaling and high-energy transitions |
| 8 Double slit | 5,6,11 | Coherence/decoherence and partner sourcing |
| 9 Atom | 10,11 | Matter structure to cosmic aggregation |
| 10 Cosmic scales | 11,12 | Gravity/time-dilation/global dynamics |
| 11 Dark sector | 10,12 | Cosmology + observability constraints |
| 12 Zeno/time | 2,5,10,11 | Time foundations and dilation closure |

## Closed link ledger (current pass)

These cross-links were explicitly closed in this pass:

- `2 -> 12` clock-law closure:
  - `f_b=f_{b0}/gamma`
  - `v_int=sqrt(c^2-v^2)` mapped to Section-12 `v_time`
- `10 -> 12` gravity-time closure:
  - `v_flow=sqrt(2GM/r)`
  - `d tau / dt = sqrt(1-2GM/(rc^2))` mapped to motion-budget identity
- `11 -> 12` dark-clock constraint:
  - `f_clock^DM,coh = 0` from spring-only ontology
  - thermal tail `f_clock^DM,therm` and `S_clock` bound exported to Section 12.9.1
  - explicitly not identified with SR photon-null `d tau = 0`
- `2 -> 3` fusion Hamiltonian closure:
  - `H_e` uses Section-2 `H_coin`
  - `K>=K_f` quenches `Omega` and maps to fused `H_p`
- `2 -> 8` wake-kernel closure:
  - `A_L,A_R` sourced from pump frequency `omega_b` and `|psi_s|^2` sea drive
  - opposite-phase lock `phi_R=phi_L+pi` imported from Section 6
- `4 -> 10` geodynamo pressure bridge:
  - `P_core`, `alpha`, `Gamma_obs`, `C_N` mapped to `Delta U(P)` and `tau_flip`
- `10 -> 11` dark-energy bridge:
  - `P_sea,DE`, `w(z)`, and CPL map exported to Section 11 fit layer
- `2,5,6 -> 11` dark-sector microdynamics:
  - `S_res,DM=0` gives `sigma_gammaDM=sigma_geom K_sup`
  - `Gamma_pin`/`Gamma_unpin` define `Gamma_sep` and transfer `Q`
- `4 -> 9,10` neutron bridges:
  - `\gamma_{obs}\leftrightarrow\nu` decay/capture map (O4-3)
  - `\mu_n=-g_{eff}(2m_e/m_n)\mu_N` magnetic closure (O4-4)
- `6,8 -> 7,8` entanglement/tunneling:
  - `\Gamma_{partner}` composition law (O8-1)
  - `\tau_{rec}`, `T_{eff}`, `\Delta V` sharing law (O7-2/3)
  - `\tau_{re}` electron/fullerene scaling (O8-3)
- `2,3,4,9` matter stack:
  - `q=q_0 chi` charge map (O2-3)
  - `f_{clock}^p=0` proton-clock nulls (O3-4)
  - `E_{bond}`, `\psi_{nlm}` chemistry bridges (O9-2/3)
- `1,2,5,12` foundations:
  - sea Born `P_S\propto T_S^2` (O1-1)
  - `\rho_{S,eff}=\Pi_{vac}u_S/c^2` (O1-3)
  - observation prime-split descent (O12-1)
- `12 -> 10` GR budget extension:
  - static metric rule `v_time=c sqrt(A)` generalizes Schwarzschild `v_flow` case

## Update checklist (run every section pass)

- [x] Update changed section derivation file
- [x] Recheck matrix targets
- [x] Sync symbol names with `symbol_registry.md`
- [x] Log unresolved tensions in `conflict_log.md`
- [x] Mark affected section statuses in `README.md`
