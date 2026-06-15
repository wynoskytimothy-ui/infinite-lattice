'use strict';

const {
  buildDoc, saveDoc, H1, H2, P, PR, EQ, EQ_NUM, PAGEBREAK,
  BOX, DIAGRAM, BULLET, OPEN_Q, PREDICTION,
  TextRun, Paragraph, AlignmentType,
} = require('./doc_helpers.js');

const children = [];

// PART V
children.push(new Paragraph({ spacing: { before: 2400 }, children: [new TextRun('')] }));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 480 },
  children: [new TextRun({ text: 'PART V', size: 36, font: 'Georgia', color: '888888', italics: true })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 1200 },
  children: [new TextRun({ text: 'ATOMS AND CHEMISTRY', size: 48, bold: true, font: 'Georgia' })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [new TextRun({
    text: 'Chapter 15 builds atoms and chemistry on anchored hydrogenic physics plus coin/pump and infinite 3D spring-plane (Part II) interpretations.',
    size: 22, italics: true, font: 'Georgia', color: '555555',
  })],
}));
children.push(PAGEBREAK());

// CHAPTER 10
children.push(H1('Chapter 15: The Atom'));

children.push(BOX([
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { line: 360, after: 0 },
    children: [new TextRun({
      text: 'The atom is geometric resonance of the cosmic ocean around a nuclear drain — not a separate mystery layer. Anchored hydrogenic math (Bohr, Rydberg, 2n², SEMF) carries the numbers; coins, pumps, and entangled inner photons carry the picture.',
      size: 22, italics: true, font: 'Georgia',
    })],
  }),
]));

children.push(H2('10.1  Why the Electron Does Not Collapse'));

children.push(PR(
  'Classically an electron should spiral into the nucleus. Three anchored brakes apply: (1) ',
  { text: 'quantization', bold: true },
  ' — only standing-wave orbits with L = nℏ survive; (2) ',
  { text: 'uncertainty', bold: true },
  ' — confining the coin below a_0 raises momentum cost; (3) ',
  { text: 'Pauli', bold: true },
  ' — fermions cannot pile into one state. **MODEL** add-on: the inner photon must bounce in resonance with the proton drain; non-resonant orbits shed energy into the sea and decay.'
));

children.push(H2('10.2  Bohr Radius — a₀ = ƛ_C/α'));

children.push(PR(
  'Standing wave on a circular orbit: 2πr = nλ = nh/(m_e v) → m_e v r = nℏ. Coulomb balance ke²/r² = m_e v²/r. Eliminating v gives r_n = n²ℏ²/(k e² m_e). At n = 1:'
));

children.push(EQ_NUM('a_0 = ℏ²/(k e² m_e) = ƛ_C/α ≈ 5.292×10⁻¹¹ m', '10.1'));
children.push(EQ_NUM('α = k e²/(ℏ c) = e²/(4πε₀ℏc) ≈ 1/137.036', '10.2'));
children.push(EQ_NUM('ƛ_C = ℏ/(m_e c) = λ_C/(2π)  [reduced Compton, C1]', '10.2a'));

children.push(PR(
  'The orbital scale is the Compton scale divided by EM coupling α — ',
  { text: 'ANCHORED', bold: true },
  '. Lattice reading: α measures how tightly the inner photon couples to the sea against nuclear drain; 1/α is how spread-out the coin must be to balance bounce pressure vs pull — **MODEL** (why α ≈ 1/137 remains **OPEN**).'
));

children.push(H2('10.3  Hydrogen Energy Levels and Spectra'));

children.push(EQ_NUM('r_n = n² a_0', '10.3'));
children.push(EQ_NUM('E_n = −(1/2) k e²/r_n = −m_e c² α²/(2n²) = −13.606 eV/n²', '10.4'));
children.push(EQ_NUM('1/λ = R_H (1/n_f² − 1/n_i²),   R_H ≈ 1.097×10⁷ m⁻¹', '10.5'));

children.push(PR(
  'Transitions emit photons with E = |E_{n_i} − E_{n_f}| — Balmer (n_f = 2), Lyman (n_f = 1), etc. match measurement to many decimals — **ANCHORED**. **MODEL**: each n is a stable inner-photon bounce pattern locked to the drain; spectral lines are polarizer-selected packet energies when the pattern steps down (Ch 6.6 preview).'
));

children.push(H2('10.4  Shell Capacity 2n²'));

children.push(EQ_NUM('N_max(n) = Σ_{ℓ=0}^{n−1} 2(2ℓ+1) = 2n²', '10.6'));

children.push(BULLET('n = 1: 2 (K shell, H–He)'));
children.push(BULLET('n = 2: 8 (L shell, Li–Ne)'));
children.push(BULLET('n = 3: 18 full shell; 3s+3p only 8 in row 3 (Aufbau)'));
children.push(BULLET('n = 4: 32 full; row 4 = 4s + 3d + 4p = 18'));

children.push(PR(
  'Pauli: two electrons per (n, ℓ, m_ℓ) with opposite spin — **PROVEN** counting. **MODEL**: each quantum label is an independent resonance mode of the sea around the drain; paired spins = opposite inner-photon bounce phases on the same mode.'
));

children.push(H2('10.5  Orbital Shapes — s, p, d, f'));

children.push(PR(
  'Angular part of hydrogenic ψ: spherical harmonics Y_ℓ^m(θ,φ), eigenmodes of L² on a sphere — **ANCHORED**. ℓ = 0 sphere; ℓ = 1 three dumbbells; ℓ = 2 five cloverleafs; ℓ = 3 seven f modes. Node count = ℓ. **MODEL / PARTIAL** (O9-3): these are standing-wave patterns of sea disturbance around a spherical proton drain — full derivation from coin geometry without importing Y_ℓ^m is **OPEN**.'
));

children.push(EQ_NUM('L² Y = ℓ(ℓ+1)ℏ² Y,   L_z Y = m_ℓ ℏ Y', '10.7'));

children.push(H2('10.6  Periodic Table and Aufbau'));

children.push(PR(
  'Fill order (Madelung): 1s < 2s < 2p < 3s < 3p < 4s < 3d < 4p < … — **ANCHORED** chemistry. Noble-gas closures: He 2, Ne 10, Ar 18, Kr 36, Xe 54, Rn 86, Og 118. Row sizes 2, 8, 8, 18, 18, 32 follow which (n, ℓ) blocks open at each energy. **MODEL**: 4s fills before 3d because s penetration couples the inner photon closer to the drain — lower energy, not arbitrary ordering.'
));

children.push(PR(
  'Columns share outer configuration: Group 1 (one s electron, reactive donors); Group 17 (seven valence, acceptors); Group 18 (closed shell, inert). Reactivity = how far outer spring polarizer is from a stable closed pattern.'
));

children.push(H2('10.7  Covalent Bonds'));

children.push(EQ_NUM('E_bond = U_C(r_0) − C_b ℏω_b |η_AB|² Π_pin', '10.8'));
children.push(EQ_NUM('D_e(H₂) ≈ 4.48 eV  [E-check vs 2×(−13.6) eV atoms]', '10.9'));

children.push(PR(
  'Heitler–London singlet sharing lowers total energy vs two isolated H atoms (−27.2 eV → −31.7 eV classically cited). **MODEL**: overlapping coins entangle inner photons; synchronized opposite-phase pumps combine wakes constructively. Bond order scales with shared channels (H–H 4.5 eV, C≡C ~8.7 eV, N≡N ~9.8 eV — **E** tabulations). Microscopic η_AB from geometry — **PARTIAL** (O9-2).'
));

children.push(H2('10.8  Nuclear Binding — SEMF'));

children.push(EQ_NUM('B(A,Z) = a_V A − a_S A^{2/3} − a_C Z(Z−1)/A^{1/3} − a_A (A−2Z)²/A + δ', '10.10'));

children.push(BULLET('a_V ≈ 15.5 MeV — volume (each nucleon joins network)'));
children.push(BULLET('a_S ≈ 17.2 MeV — surface (fewer neighbors at edge)'));
children.push(BULLET('a_C ≈ 0.72 MeV — Coulomb proton repulsion'));
children.push(BULLET('a_A ≈ 23.3 MeV — N≠Z asymmetry penalty'));
children.push(BULLET('δ — pairing (even-even bonus)'));

children.push(PR(
  'Coefficients are empirical fits — **E / ANCHORED** phenomenology. **MODEL** (C4): binding from trapped-electron entanglement network across neutrons/protons; extension B_total = B_SEMF + B_share(N,Z) with C_N(N,Z) — **PARTIAL** (O9-1).'
));

children.push(H2('10.9  Iron Peak, Fusion, and Fission'));

children.push(PR(
  'B/A peaks near Fe-56 (~8.79 MeV/nucleon) — **ANCHORED** astrophysics. A < 56: fusion releases energy (pp chain, CNO). A > 56: fusion costs energy; fission of U-238 can release energy by moving toward higher B/A. Stars stop fusing at iron because the entanglement+Coulomb balance has no exothermic path upward — **MODEL** reading of the iron peak.'
));

children.push(H2('10.10  N/Z Stability Valley'));

children.push(EQ_NUM('∂B/∂Z|_A = 0  →  Z_opt(A) ≈ A/2 / (1 + 0.00772 A^{2/3})', '10.11'));

children.push(PR(
  'Light nuclei: N ≈ Z. Heavy nuclei: extra neutrons dilute Coulomb without adding repulsion. Checks: A = 4 → Z_opt ≈ 2; A = 12 → 6; A = 56 → ~25–26 (Fe Z = 26); A = 238 → ~92 (U Z = 92) — **E** valley. Beyond uranium: Coulomb wins; α-decay and instability — **ANCHORED** trend.'
));

children.push(H2('10.11  Master Equations'));

children.push(BOX([
  new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: 'a_0 = ƛ_C/α     E_n = −13.606 eV/n²     N(n) = 2n²', font: 'Consolas', size: 18 })] }),
  new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: '1/λ = R_H(1/n_f² − 1/n_i²)     Y_ℓ^m orbitals', font: 'Consolas', size: 18 })] }),
  new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: 'B(A,Z) SEMF     B/A_max ≈ 8.8 MeV at Fe-56', font: 'Consolas', size: 18 })] }),
  new Paragraph({ spacing: { after: 0 }, children: [new TextRun({ text: 'Z_opt = A/2/(1 + 0.00772 A^{2/3})     D_e(H₂) ≈ 4.5 eV', font: 'Consolas', size: 18 })] }),
], { shaded: true }));

children.push(PR(
  'Chemistry, in this picture, is the art of entangling inner photons between atoms. The periodic table maps how many ways the ocean can resonate around increasing nuclear drains — anchored shell math plus **MODEL** mechanism.'
));

children.push(OPEN_Q('Why α ≈ 1/137?', 'OPEN — dimensionless **E** anchor; deep lattice origin not closed.'));
children.push(OPEN_Q('Derive Y_ℓ^m amplitudes from coin geometry alone?', 'OPEN (O9-3) — partial progress for ℓ ≤ 1 only.'));
children.push(PAGEBREAK());

// PART VI
children.push(new Paragraph({ spacing: { before: 2400 }, children: [new TextRun('')] }));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 480 },
  children: [new TextRun({ text: 'PART VI', size: 36, font: 'Georgia', color: '888888', italics: true })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 1200 },
  children: [new TextRun({ text: 'COSMOLOGY', size: 48, bold: true, font: 'Georgia' })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [new TextRun({
    text: 'Chapters 16–19: Higgs reference clock (v), dark sector without sync (P11), lattice proper time (Sec 12), gravity as cosmic synchronizer (Sec 10/18).',
    size: 22, italics: true, font: 'Georgia', color: '555555',
  })],
}));
children.push(PAGEBREAK());

// CHAPTER 11
children.push(H1('Chapter 16: The Higgs Boson'));

children.push(BOX([
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { line: 360, after: 0 },
    children: [new TextRun({
      text: 'SM Higgs sector (**E**) with MODEL identification: v = 246 GeV as sea intrinsic tension scale. Mass m = gv/√2 for coupled fermions.',
      size: 22, italics: true, font: 'Georgia',
    })],
  }),
]));

children.push(H2('11.1  Higgs Potential'));
children.push(EQ_NUM('V(H) = −μ²|H|² + λ|H|⁴', '11.1'));
children.push(EQ_NUM('v = 246 GeV', '11.2'));

children.push(H2('11.2  Mass from Coupling'));
children.push(EQ_NUM('m = g v / √2', '11.3'));
children.push(BULLET('e: g_e ~ 2.94×10⁻⁶ → 0.511 MeV'));
children.push(BULLET('μ, τ, top: increasing g — **E**'));
children.push(BULLET('Photon: g_γ = 0 — no coin structure to couple'));

children.push(H2('11.3  Three Generations'));
children.push(PR('Three fermion generations — **E**. Lattice resonance-scale story = **MODEL** (not derived here).'));

children.push(H2('11.4  Higgs Boson Mass'));
children.push(EQ_NUM('m_H² = 2λv²  →  λ ≈ 0.129', '11.4'));

children.push(H2('11.5  Sea Tension as Reference Clock'));
children.push(PR(
  'v = 246 GeV is the sea\'s intrinsic tension scale — **E** anchor. **MODEL**: all lattice pumps calibrate bounce rates against this reference; gravity wells then synchronize local clocks to shared potential (Chapter 19). DM couples weakly to inner-photon modes (P11-1) but still gravitates.'
));

children.push(PREDICTION('No large invisible Higgs branching — consistent with LHC (**ANCHORED** so far).'));
children.push(PAGEBREAK());

// CHAPTER 12
children.push(H1('Chapter 17: Dark Matter and Dark Energy'));

children.push(BOX([
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { line: 360, after: 0 },
    children: [new TextRun({
      text: 'P11-1: DM = spring without inner photon (σ_γDM suppressed). P11-2: DE channel from freed inner-photon sea pressure. P11-3: DM mesh fill φ_AB for long paths.',
      size: 22, italics: true, font: 'Georgia',
    })],
  }),
]));

children.push(H2('12.1  Dark Matter — Spring Without Inner Photon'));

children.push(PR(
  'Dark matter is not a new particle species bolted onto gravity. It is the ',
  { text: 'same lattice structure as matter', bold: true },
  ' with the engine removed. An electron has coin + spring + membrane + trapped inner photon. Remove the trapped photon. The spring-and-coin skeleton remains — it still curves spacetime — but there is nothing inside to pump, polarize, or release packets.'
));

children.push(PR(
  'Through the polarizer picture (Chapter 11): dark matter is an '
  { text: 'empty polarizer', bold: true },
  '. Observation photons arrive, the spring can align to the probe axis, but there is no inner photon to filter. No packet released. No ionization cascade. No detector click. Shine all the light you want — σ_γDM is suppressed because there is no EM engine, not because the structure is invisible to gravity.'
));

children.push(...DIAGRAM([
  '',
  '   ELECTRON (visible)              DARK MATTER (invisible EM)',
  '   ╔═══════════════╗               ╔═══════════════╗',
  '   ║  coin + pump  ║               ║  coin + spring║',
  '   ║  ∿∿ ● ∿∿∿∿   ║               ║   (no ●)      ║',
  '   ╚═══════════════╝               ╚═══════════════╝',
  '   polarizer + light               polarizer, no light inside',
  '',
], 'Figure 12.1: DM = spring lattice without inner photon (P11-1).'));

children.push(EQ_NUM('∇²Φ = 4πG(ρ_b + ρ_DM)', '12.1'));
children.push(EQ_NUM('σ_{γDM} ≈ σ_geom K_sup(ω) ≪ σ_max^{exp}', '12.2'));
children.push(PR(
  'Gravitates (ρ_DM > 0); effectively dark to EM — **MODEL/PARTIAL** (O11-1). Forms diffuse halos because without pumps it cannot clump into tight luminous structures.'
));

children.push(H2('12.1a  Why DM Does Not Synchronize'));

children.push(PR(
  'Gravity synchronizes clocks only where trapped inner photons maintain bounce ledgers (Ch 19). DM has no inner photon ⇒ f_clock^DM,coh ≈ 0 (Ch 17): it curves Φ and fills φ_AB mesh paths for long Bell trajectories, but never joins the EM entanglement tick network. Luminous matter in a well shares dilation; DM is gravitational scaffolding without a clock to align — **MODEL**.'
));

children.push(H2('12.2  Dark Energy'));
children.push(EQ_NUM('w(z) ≈ −1  (today)', '12.3'));
children.push(PR('P11-2 pressure channel. CPL fits allow small w_a drift — do not claim w = −1 exact at all z (**Sec 11**).'));

children.push(H2('12.3  Energy Budget'));
children.push(BULLET('Baryons + EM matter ~ 5%'));
children.push(BULLET('ρ_DM ~ 27%'));
children.push(BULLET('ρ_DE ~ 68% — **E** cosmology'));

children.push(PREDICTION('Direct EM DM detection remains null at predicted σ_γDM suppression — test ongoing.'));
children.push(PAGEBREAK());

// CHAPTER 13
children.push(H1('Chapter 18: The Lattice Clock'));

children.push(BOX([
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { line: 360, after: 0 },
    children: [new TextRun({
      text: 'Proper time = pump tick count. C1: f_b = m_e c²/(2h), T_bounce = 2 t_cell, c = λ_C/t_cell. Motion budget unifies SR (Sec 12).',
      size: 22, italics: true, font: 'Georgia',
    })],
  }),
]));

children.push(H2('13.1  The Lattice as Cosmic Clock'));

children.push(PR(
  'The π lattice is not only address space — it is the tick structure of proper time. Each inner-photon bounce advances one cell crossing; one full pump period is two crossings:'
));

children.push(EQ_NUM('τ ∝ N_{bounce} × T_{bounce}', '13.1'));
children.push(EQ_NUM('T_{bounce} = 4L/c = 2 t_{cell}', '13.1a'));
children.push(EQ_NUM('t_{cell} = λ_C/c = h/(m_e c²) ≈ 8.09×10⁻²¹ s', '13.1b'));
children.push(EQ_NUM('f_b = 1/T_{bounce} = m_e c²/(2h) ≈ 6.18×10¹⁹ Hz', '13.1c'));

children.push(PR(
  'Iteration depth k on the π tree tracks ',
  { text: 'address refinement', bold: true },
  ' (Chapter 2); link k to elapsed bounce count N_bounce is **MODEL** — do not equate k with rest energy. Coordinate time t is the lab cosmic clock; proper time τ is private bounce ledger per coin.'
));

children.push(H2('13.2  Entanglement = Shared Tick'));

children.push(PR(
  'Pair-produced or jointly prepared electrons begin at the same lattice anchor. Each bisection they take opposite ±B at the ',
  { text: 'same', bold: true },
  ' k — one shared tick, mirrored y (Chapter 12.7). Entanglement is cosmic synchronization: both inner photons advance k together until Γ_break or a measurement splits the trajectory.'
));

children.push(H2('13.3  Decoherence = Tick Mismatch'));

children.push(PR(
  'After separation each electron may bind a new sea partner at depth k_env ≠ k_local. Phase reconciliation takes time τ_reconcile ~ |Δk| t_cell. Large |Δk| → faster loss of fringe visibility. Environment gas (Ch 14) raises Γ_partner and scrambles k relationships — exponential V(P) envelope with stepped structure underneath at t_cell resolution — **MODEL**.'
));

children.push(H2('13.4  Iteration-Depth Uncertainty'));

children.push(EQ_NUM('Δk · Δt ≳ ℏ/(2 m_e c²)   [conjecture, OPEN O12-1]', '13.2'));
children.push(EQ_NUM('Δφ ∼ (k_1 − k_2) × (phase per tick)   [partner phase slip]', '13.2a'));

children.push(PR(
  'Standard Heisenberg ΔxΔp ≥ ℏ/2 remains **ANCHORED**. The lattice adds: we cannot know every partner\'s iteration depth in the cosmic entanglement web without disturbing it — **MODEL** explanation for irreducible spread in sequential measurements, not a claim that outcomes are classical-hidden (Bell still blocks that). "Randomness" at the apparatus = ignorance of geometric partner-phase facts we did not track.'
));

children.push(H2('13.5  SR Time Dilation'));
children.push(EQ_NUM('dτ = dt √(1 − v²/c²)', '13.3'));
children.push(EQ_NUM('f_b(v) = f_b/γ', '13.4'));

children.push(PR(
  'Moving coins lengthen diagonal bounce paths → fewer ticks per lab second. At v → c internal tick rate → 0 while coordinate time t continues — **ANCHORED** SR; lattice reading: iteration counter slows, cosmic clock does not.'
));

children.push(H2('13.6  Why c is Constant'));
children.push(EQ_NUM('c = λ_C / t_cell', '13.5'));
children.push(PR('Cell geometry fixed — **D** from Ch 1 (not observer-dependent).'));

children.push(H2('13.7  Motion Budget'));
children.push(EQ_NUM('v_{space}² + v_{time}² = c²', '13.6'));
children.push(EQ_NUM('v_{time} = c/γ = √(c² − v²)', '13.7'));
children.push(BULLET('Massive at rest: v_space=0, v_time=c (full internal tick rate).'));
children.push(BULLET('Photon: v_space=c, no trapped pump — no private τ ledger.'));

children.push(H2('13.8  Arrow of Time'));

children.push(PR(
  'Bisection geometry is reversible (Ch 2.7), but macroscopic arrows follow forward-only bounce and iteration counters — recovering past k would require undoing released packets and re-forming past entanglements (coordination at cosmic scale). Arrow of time = cumulative forward motion of private ledgers in an irreversible sea — **MODEL** (thermodynamic link **PARTIAL**).'
));

children.push(H2('13.9  Gravity Couples Coordinate and Proper Time'));

children.push(PR(
  'Chapter 19 adds the gravitational layer: at fixed Φ all trapped pumps share the same √(1−2GM/rc²) factor — coordinate t vs private τ split is **ANCHORED** GR; reading that shared slowdown as shell-by-shell clock synchronization is **MODEL**. Free space: scattered k and fast decoherence. Bound in a well: common tick rate per radius, structure can persist.'
));

children.push(PREDICTION(
  'Zeptosecond decoherence and Born statistics: stepped structure at ~t_cell and T_bounce, smoothing to exponential above ~10⁻¹⁸ s — **TEST** (Ch 6.2a, 17.8).'
));
children.push(PAGEBREAK());

// CHAPTER 14
children.push(H1('Chapter 19: Gravity as Cosmic Clock Synchronization'));

children.push(BOX([
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { line: 360, after: 0 },
    children: [new TextRun({
      text: 'Free electrons carry scattered iteration histories (Ch 18). Gravity binds matter into shared potential wells and sets a common tick rate per shell — anchored redshift and GPS; "synchronizer of structure" is the lattice MODEL on top.',
      size: 22, italics: true, font: 'Georgia',
    })],
  }),
]));

children.push(H2('14.1  Chaos vs Order — Free vs Bound'));

children.push(PR(
  'Isolated electrons in the sea: private bounce ledgers, mismatched partner depths k, rapid decoherence — ',
  { text: 'chaos', bold: true },
  ' at the entanglement layer (Ch 11–13). Bound systems (atoms, solids, planets, stars) sit in shared potentials. **MODEL thesis**: gravity is not only attractive geometry; it is the mechanism that ',
  { text: 'aligns local lattice tick rates', bold: true },
  ' so structure can persist. Einstein field equations from this alone — **OPEN**.'
));

children.push(H2('14.2  Anchored Clock Law'));

children.push(EQ_NUM('dτ/dt = √(1 − 2GM/(rc²))', '14.1'));
children.push(EQ_NUM('f_{clock}(r) = f_b √(1 − 2GM/(rc²))', '14.2'));
children.push(EQ_NUM('v_{flow} = √(2GM/r)', '14.3'));

children.push(PR(
  'At fixed radius r every electron feels the ',
  { text: 'same', bold: true },
  ' dilation factor — **ANCHORED** static GR (Sec 10/12). Earth surface: f/f₀ ≈ 1 − 7×10⁻¹⁰. GPS satellites correct ~38 µs/day — empirical proof clocks differ by potential, not by electron history alone.'
));

children.push(PR(
  '**MODEL** reading: inward sea flow v_flow toward drains (mass) nudges scattered k values toward a common local rate — like aligning second hands that still started at different counts but now run at identical speed per shell.'
));

children.push(H2('14.3  Hierarchy of Synchronized Scales'));

children.push(BULLET('Atomic: Coulomb well + QM; unified internal pump band — **ANCHORED** chemistry.'));
children.push(BULLET('Molecular/crystal: bonds + phonons lock neighbors — **E** + **MODEL** entanglement overlap.'));
children.push(BULLET('Planetary: 10⁴⁹ atoms in one Φ(r); rotation and tides as macro rhythm — **MODEL** narrative.'));
children.push(BULLET('Stellar: fusion equilibrium = pressure–gravity balance — **ANCHORED**; "star as clock network" — **MODEL**.'));
children.push(BULLET('Galactic / cosmic: density waves, CMB uniformity — phenomenology **ANCHORED**; lattice sync story **MODEL**.'));
children.push(PR('Section 10 planetary magnetism, geodynamo, and stellar cycles live in `section_10_planetary_cosmic_scales.md` — not fully expanded in this chapter yet.'));

children.push(H2('14.4  Stability and Decay as Sync Loss'));

children.push(PR(
  'Free neutron τ ~ 879 s vs stable bound neutron — **ANCHORED**. **MODEL**: free state lacks nuclear entanglement network to hold inner-photon phase; bound nucleon shares nuclear-scale well. High temperature breaks molecular sync before bonds re-lock — disorder as unsynchronized clocks. Gas / liquid / solid as weak / partial / strong tick coupling — **MODEL**, not a replacement for statistical mechanics.'
));

children.push(H2('14.5  Dark Matter — No Clock to Sync'));

children.push(PR(
  'DM springs lack inner photons (Ch 17) ⇒ no private τ ledger, f_clock^DM,coh ≈ 0. They curve space and fill φ_AB mesh paths but do not join the EM entanglement tick network. Halos provide gravity without luminous synchronization — **MODEL/PARTIAL** (O11, O12-3).'
));

children.push(H2('14.6  Higgs Sea Tension as Reference'));

children.push(PR(
  'Local wells synchronize rates relative to the sea\'s tension scale v (Ch 16). Massless photons have no coin pump; massive species bounce at rates tied to v and local Φ. Reference clock + gravity well = two-layer time story — **MODEL**.'
));

children.push(H2('14.7  Entanglement Lifetime in Wells'));

children.push(PR(
  'Γ_break rises with environmental photons (Ch 12). **MODEL**: deeper wells give common dilation, partially suppressing relative Δk drift between partners — longer τ_E in principle. Test: compare Bell decoherence at matched pressure but different gravitational potential (hard; effect small on Earth).'
));

children.push(EQ_NUM('τ_{sync} ∼ t_{cell}/|ΔΦ/c²|   [order-of-magnitude MODEL, not derived]', '14.4'));

children.push(H2('14.8  CMB and Cosmic Clock'));

children.push(PR(
  'CMB temperature uniformity — **ANCHORED** (ΔT/T ~ 10⁻⁵). Standard: inflation + causal patches. **MODEL** alternative: early universal well synchronized clocks before expansion; imprint preserved in CMB — not proven; do not claim inflation is replaced without calculation.'
));

children.push(H2('14.9  Master Picture'));

children.push(BOX([
  new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: 'FREE → scattered k, fast decoherence', font: 'Consolas', size: 18 })] }),
  new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: 'BOUND → shared √(1−2GM/rc²) per shell', font: 'Consolas', size: 18 })] }),
  new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: 'STRUCTURE = nested clocks (atom ⊂ molecule ⊂ … ⊂ cosmos)', font: 'Consolas', size: 18 })] }),
  new Paragraph({ spacing: { after: 0 }, children: [new TextRun({ text: 'GRAVITY = synchronizer (MODEL) + curvature (ANCHORED redshift)', font: 'Consolas', size: 18 })] }),
], { shaded: true }));

children.push(OPEN_Q('Derive Einstein equations from lattice synchronization?', 'OPEN — redshift and motion-budget algebra partial (Sec 10/12).'));
children.push(PREDICTION('Precision clocks + entanglement hubs at different potentials: measure combined redshift and τ_E drift — small on Earth, larger near compact objects.'));
children.push(PAGEBREAK());

(async () => {
  const doc = buildDoc('V-VI', 'Parts V–VI — Atoms and Cosmology (Ch 15–19)', children);
  await saveDoc(doc, 'book/output/05_Atoms_and_Cosmology.docx');
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
