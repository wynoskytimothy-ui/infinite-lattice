'use strict';

const {
  buildDoc, saveDoc, H1, H2, H3, P, PR, EQ, EQ_NUM, CAPTION, PAGEBREAK,
  BOX, DIAGRAM, BULLET, OPEN_Q, PREDICTION,
  TextRun, Paragraph, AlignmentType,
} = require('./doc_helpers.js');

const children = [];

// ===================== PART III HEADER =====================
children.push(new Paragraph({ spacing: { before: 2400 }, children: [new TextRun('')] }));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 0, after: 480 },
  children: [new TextRun({ text: 'PART III', size: 36, font: 'Georgia', color: '888888', italics: true })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 0, after: 1200 },
  children: [new TextRun({ text: 'PARTICLES', size: 56, bold: true, font: 'Georgia' })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 240, after: 0 },
  children: [new TextRun({
    text: 'Chapters 8–10 build electron, proton, and neutron inside the lattice — the actual 3D arena (Part II, Ch 3–7) — plus coin geometry on the spring plane.',
    size: 22, italics: true, font: 'Georgia', color: '555555',
  })],
}));
children.push(PAGEBREAK());

// =====================================================================
// CHAPTER 3: THE ELECTRON
// =====================================================================
children.push(H1('Chapter 8: The Electron'));

children.push(BOX([
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { line: 360, after: 0 },
    children: [new TextRun({
      text: 'Membrane = packet boundary; spring = polarizer (spirals out when soft, flattens to coin when hard). Four states WH/WS/BH/BS; measurement compression flips axes WS→BH. f_b from trapped-photon pump sets mass and proper time.',
      size: 22, italics: true, font: 'Georgia',
    })],
  }),
]));

children.push(H2('3.1  The Coin–Spring–Membrane–Photon Structure'));

children.push(PR(
  'An electron occupies one lattice cell of width ',
  { text: 'λ_C = h/(m_e c)', bold: true },
  ' (full Compton wavelength). Inside, four coupled components (C1 geometry):'
));

children.push(BULLET('COIN — two faces (white/black); opposite coil spins on the two sides.'));
children.push(BULLET('SPRING — connects faces; active polarizer on z-plane (Ch 3–7, Ch 11).'));
children.push(BULLET('MEMBRANE — boundary film, always opposite color to the visible face.'));
children.push(BULLET('TRAPPED PHOTON — engine; bounce drives membrane and clock.'));

children.push(H2('3.1a  The Spring Is an Active Polarizer'));

children.push(PR(
  'The spring is not a passive container. It is an ',
  { text: 'active filter', bold: true },
  ' — the same role a polarizer plays for light. A polarizer has an axis. Light aligned with that axis passes; light perpendicular is absorbed; light at angle θ passes with probability cos²(θ/2). The polarizer does not merely block — it ',
  { text: 'reorients', italics: true },
  ' what passes to match its axis.'
));

children.push(PR(
  'The electron spring works the same way. Before observation the inner photon is in superposition across the coin. When a probe photon arrives from the observer, it pushes along direction n̂ and ',
  { text: 'sets the spring\'s axis to n̂', bold: true },
  '. The spring becomes a polarizer along the observer\'s compression direction. The inner photon must then align with that axis (release a detectable packet) or anti-align (stay trapped on the opposite face). Probabilities cos²(θ/2) and sin²(θ/2) — Malus\'s law — are ',
  { text: 'ANCHORED', italics: true },
  ' in standard QM; reading them as spring-tension² deposition is ',
  { text: 'PARTIAL', italics: true },
  ' (O1-1, O2-2, full treatment Chapter 11).'
));

children.push(H3('3.1a.1  Membrane = Packet, Spring = Polarizer'));

children.push(PR(
  'The ',
  { text: 'membrane', bold: true },
  ' is the visible packet boundary — the film whose color reports where the trapped photon last deposited tension. The ',
  { text: 'spring', bold: true },
  ' is the active polarizer: same lattice strings as the coin faces, but wound as opposing spirals. Extended spring = low tension (',
  { text: 'soft', italics: true },
  '); flattened spring = high tension (',
  { text: 'hard', italics: true },
  '). Macroscopic coil springs are large-scale alignments of this same spiral geometry — **MODEL**.'
));

children.push(H3('3.1a.2  Why Compression Looks Like a Coin'));

children.push(PR(
  'A spiraled spring has volume: string boundaries occupy both faces and the axis gap. Unilateral compression folds those boundaries onto one geometric plane — the structure ',
  { text: 'looks', italics: true },
  ' like a flat coin, but it is not a solid disc. It is a collapsed spiral. No external potential-energy reservoir is postulated; tension is intrinsic lattice string alignment. Release does not "push" from stored energy — the trapped photon resumes its deterministic bounce path, and recursive geometry ',
  { text: 'obliges', italics: true },
  ' the strings to spiral outward and reopen the cavity for the next cycle — **MODEL**.'
));

children.push(...DIAGRAM([
  '',
  '   SPIRALED (soft)                    COMPRESSED (hard, coin-like)',
  '        ╭──∿──╮                            ╔═══════╗',
  '       ╱   │   ╲                           ║   ●   ║  ← packet pinned',
  '      ∿    │    ∿   spring spirals out     ╠═══════╣  ← spring flat',
  '       ╲   │   ╱                           ║       ║',
  '        ╰──∿──╯                            ╚═══════╝',
  '   ample bounce room                  boundaries on one plane',
  '',
], 'Figure 3.1a: Soft spiral vs hard flattened coin illusion.'));

children.push(H2('3.1b  Four Mechanical States'));

children.push(PR(
  'Two binary axes — ',
  { text: 'membrane side', italics: true },
  ' (White / Black) and ',
  { text: 'spring tension', italics: true },
  ' (Soft / Hard) — give four deterministic coin states. These replace abstract spin labels with geometry; empirical spin correlations remain **ANCHORED**.'
));

children.push(BULLET('WH — White-Hard: white face flat/compressed, photon pinned at white.'));
children.push(BULLET('WS — White-Soft: white membrane visible, spring spiraled open (low tension).'));
children.push(BULLET('BH — Black-Hard: black face flat/compressed, photon pinned at black.'));
children.push(BULLET('BS — Black-Soft: black membrane visible, spring spiraled open.'));

children.push(PR(
  'Free pump cycle (no external polarizer): photon alternates pinned faces — ',
  { text: 'WH ↔ BS ↔ WH ↔ BS', bold: true },
  ' (hard at impact side, soft inflate on opposite). Mid-transit the spring is fully soft: superposition of both sides — **MODEL** map to Dirac components (derivation audit §3.4). Lattice address quadrants (1−A, ±B) track π-tree position; WH/WS/BH/BS track coin membrane + spring tension — related layers, not identical labels (Ch 1 §1.7b).'
));

children.push(H2('3.1c  Measurement Compression and Axes Flip'));

children.push(PR(
  'A polarizer (Chapter 11) applies '
  { text: 'unilateral', bold: true },
  ' compression along axis n̂. Example path: electron in WS (white-soft, spiraled). Compression folds membrane + spring to a plane (coin silhouette). At maximum compression the mechanical axes ',
  { text: 'flip', bold: true },
  ': white-soft becomes black-hard on release. One deterministic path shift — what standard physics logs as a spin outcome. Angled polarizers still deposit tension as cos²(θ/2) — **ANCHORED** Malus; branch counting underneath — **PARTIAL** (O1-1).'
));

children.push(...DIAGRAM([
  '',
  '   MEASUREMENT CYCLE (unilateral compression)',
  '',
  '   (1) WS stable          (2) compressing       (3) max flat coin',
  '       ╭─white─╮              ╔═════╗               ╔═══╗',
  '       ∿ soft  ∿              ║  ●  ║               ║ ● ║  axes flip',
  '       ╰─black─╯              ╚═════╝               ╚═══╝  at limit',
  '',
  '   (4) release spiral  →  BH (black-hard, spiraling out)',
  '       ╭─black─╮',
  '       ╲ hard ╱   photon now pinned black; spring re-expands',
  '       ╰─white╯',
  '',
], 'Figure 3.1b: WS → BH via polarizer compression and axes flip.'));

children.push(H2('3.1d  Max Compression — Ball State'));

children.push(PR(
  'Unilateral compression = measurement / flip. ',
  { text: 'Bilateral', bold: true },
  ' compression from both sides simultaneously drives the coin to maximum density: membrane closes into a sphere (ball state). Inner photon oscillates at highest confined frequency — maximum massive packing. This is the route to proton fusion (Chapter 9): spring elastic limit exceeded, faces fuse, pump locks. χ = 1 compactness (Chapter 13). Ordinary observation stays below this threshold; electron returns to spiraled pump after single-sided release.'
));

children.push(...DIAGRAM([
  '',
  '   LIFECYCLE SUMMARY',
  '',
  '   WS/BH spiraled  ──unilateral──►  flat coin  ──flip──►  opposite hard face',
  '        ▲                              │',
  '        │         release spiral       │ bilateral max',
  '        └──────────────────────────────┴──►  BALL (sphere) ──► fusion (Ch 9)',
  '',
], 'Figure 3.1c: Spiraled ↔ coin ↔ ball lifecycle.'));

children.push(...DIAGRAM([
  '',
  '                         White face',
  '                    ╔═══════════════╗  ← membrane',
  '                    ║       ●       ║',
  '                    ║      ╱│╲      ║  ← inner photon',
  '                    ║     ╱ │ ╲     ║',
  '                    ║    ╱  │  ╲    ║  ← spring',
  '                    ║   ╱   │   ╲   ║',
  '                    ╚═══════════════╝',
  '                         Black face',
  '',
  '   Cell width:  λ_C = 2.426 × 10⁻¹² m',
  '   Cavity span: 2L = λ_C   (L = λ_C/2)',
  '   Pump period: T_bounce = 4L/c ≈ 1.62 × 10⁻²⁰ s',
  '   f_b = 1/T_bounce ≈ 6.18 × 10¹⁹ Hz',
  '',
], 'Figure 3.1: Electron coin in one Compton cell (C1).'));

children.push(H2('3.2  The Trapped Inner Photon'));

children.push(PR(
  'The inner photon is trapped by spring tension. One ',
  { text: 'full pump cycle', bold: true },
  ' (white → black → white) takes:'
));

children.push(EQ_NUM('T_bounce = 4L/c = 2 t_cell ≈ 1.619 × 10⁻²⁰ s', '3.1'));
children.push(EQ_NUM('t_cell = λ_C/c = h/(m_e c²) ≈ 8.09 × 10⁻²¹ s', '3.1a'));
children.push(EQ_NUM('f_b = 1/T_bounce = m_e c²/(2h) ≈ 6.178 × 10¹⁹ Hz', '3.2'));

children.push(PR(
  'This is the zitterbewegung frequency in Dirac theory — here it is the literal pump rate. Energy per cycle:'
));

children.push(EQ_NUM('E = h f_b = m_e c² ≈ 0.511 MeV', '3.3'));

children.push(PR(
  'Each completed bounce advances the electron\'s internal proper-time counter. Lattice address refinement (Chapter 2) supplies the discrete geometry; the pump supplies the clock.'
));

children.push(H2('3.3  Mass from Bounce Frequency'));

children.push(EQ_NUM('m = h f_b / c²', '3.4'));

children.push(PR(
  'Mass measures how many pump cycles fit into the rest-energy budget. Heavier species have shorter cavities and higher f_b. The proton case (Chapter 9) locks the pump instead of freely bouncing.'
));

children.push(H2('3.4  Spin from Coil Geometry'));

children.push(PR(
  'The two coin faces spin in opposite directions — a coil. Net angular momentum is half-integer: a 360° turn in space advances the pump only halfway through the spinor cycle (720° for identity).'
));

children.push(BULLET('WH / BH (hard pinned face) → definite spin projection (+½ / −½).'));
children.push(BULLET('WS / BS (soft spiraled) → open cavity; transit superposition.'));
children.push(BULLET('Pump alternation WH ↔ BS; polarizer flip WS → BH (§3.1c).'));

children.push(PR(
  'The four-component Dirac spinor can be read as amplitudes for coin + orientation states — interpretation ',
  { text: 'MODEL', italics: true },
  ', equation ',
  { text: 'ANCHORED', italics: true },
  ' (§3.6).'
));

children.push(H2('3.5  Compton Wavelength'));

children.push(EQ_NUM('λ_C = h/(m c) = 2L', '3.5'));
children.push(EQ_NUM('ƛ_C = ħ/(m c) = λ_C/(2π)', '3.6'));

children.push(PR('For the electron: λ_C = 2.426 × 10⁻¹² m; ƛ_C = 3.862 × 10⁻¹³ m.'));

children.push(H2('3.6  Time Dilation and the Speed Limit'));

children.push(EQ_NUM('f_b(v) = f_{b0}/γ = f_{b0}√(1 − v²/c²)', '3.7'));
children.push(EQ_NUM('v_int = √(c² − v²)', '3.8'));

children.push(PR(
  'A moving coin forces the inner photon on a longer diagonal path; fewer bounces per lab second. At v → c the bounce cannot close — no pump, no electron. Links to Chapter 18 motion budget.'
));

children.push(H2('3.7  The Dirac Equation (Interpretation)'));

children.push(EQ_NUM('iℏ ∂ψ/∂t = (c α·p + β m c²) ψ', '3.9'));

children.push(PR(
  'Standard physics treats this as fundamental. Here ψ packages coin-state amplitudes; α, β are inter-side transition operators; mc² is stored bounce energy. ',
  { text: 'Mechanism = MODEL', bold: true },
  '; empirical predictions match Dirac where tested.'
));

children.push(H2('3.8  Charge'));

children.push(PR(
  'Negative charge correlates with an open, expansive pump state (spring extended, structure pushes into the sea). Positrons are mirrored branches. Constitutive map q = q₀χ — ',
  { text: 'PARTIAL', italics: true },
  ' (O2-3); e magnitude is empirical anchor.'
));

children.push(H2('3.9  Open Questions'));

children.push(OPEN_Q(
  'Why is m_e c² exactly 0.511 MeV?',
  'In this framework, that is equivalent to fixing f_b from C1 geometry. Deeper origin of Yukawa/Higgs coupling to the electron is Chapter 16; g_e remains an empirical anchor at this pass.'
));

children.push(PREDICTION(
  'Fresh electrons within the first t_cell ≈ 8 × 10⁻²¹ s after creation may show discrete pump statistics before thermalization over ~10¹⁹ cycles. Testable as zeptosecond pair-production spectroscopy improves.'
));

children.push(PAGEBREAK());

// =====================================================================
// CHAPTER 4: THE PROTON
// =====================================================================
children.push(H1('Chapter 9: The Proton'));

children.push(BOX([
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { line: 360, after: 0 },
    children: [new TextRun({
      text: 'The proton forms when electron compression crosses fusion threshold K_f: the coin fuses, the spring disappears, and the trapped photon locks into structure. Three regions of the fused body map to quark zones; f_clock^p = 0.',
      size: 22, italics: true, font: 'Georgia',
    })],
  }),
]));

children.push(H2('4.1  Fusion Beyond the Elastic Limit'));

children.push(PR(
  'Normal compression pins the inner photon (observation). ',
  { text: 'Extreme', bold: true },
  ' compression past K_f fuses the two faces — irreversible proton phase (P3-1). The bounce engine shuts off; energy becomes structural mass.'
));

children.push(...DIAGRAM([
  '',
  '        Electron (K < K_f)          Proton (K ≥ K_f)',
  '',
  '        ╔═══════════╗               ╔═══════════╗',
  '        ║  ∿ photon ║               ║ + + + + + ║  top pole',
  '        ╠═══════════╣               ╠═══════════╣  equator',
  '        ║  ∿∿∿∿∿∿∿   ║               ║ - - - - - ║',
  '        ╚═══════════╝               ╚═══════════╝  bottom pole',
  '',
  '     spring extended                  spring gone — fused',
  '',
], 'Figure 4.1: Elastic electron vs fused proton.'));

children.push(H2('4.2  Quark Regions'));

children.push(BULLET('Top pole: positive region → up (+2/3).'));
children.push(BULLET('Equator: neutral band → down (−1/3).'));
children.push(BULLET('Bottom pole: positive region → up (+2/3).'));
children.push(BULLET('Sum: +2/3 − 1/3 + 2/3 = +1 ✓'));

children.push(PR(
  'Quarks are not separate particles stuffed inside — they are ',
  { text: 'zones', bold: true },
  ' of one fused coin. Fractional charges are geometric bookkeeping (**MODEL**), not ad hoc labels.'
));

children.push(H2('4.3  Mass Ratio m_p/m_e ≈ 1836 (C2)'));

children.push(PR(
  'Geometry-first rule (architecture mandate C2): derive compression threshold K_f from coin/spring limits — ',
  { text: 'do not', bold: true },
  ' define K_f by fitting 1836.'
));

children.push(EQ_NUM('R_pe^{(0)} = π²/8 ≈ 1.23  [spring-only geometry]', '4.1'));
children.push(EQ_NUM('R_pe = R_pe^{(0)} × ℳ_lat  [lattice multiplier, Ch 2 / C6]', '4.2'));
children.push(EQ_NUM('R_pe^E = 1836.152…  [CODATA — consequence check]', '4.3'));

children.push(PR(
  'Measured m_p c² = 938.272 MeV. The large ratio needs ℳ_lat from the active lattice network, not a single FIT inversion K_f := (R−1)/R.'
));

children.push(H2('4.4  No Internal Proton Clock'));

children.push(EQ_NUM('f_{clock}^{(p)} = 0', '4.4'));

children.push(PR(
  'No free trapped-photon bounce. Proton still has magnetic moment and spin response — but no electron-style pump sidebands (O3-4 null test).'
));

children.push(H2('4.5  Stability'));

children.push(PR(
  'Fused structure has no lower-energy decay path; no trapped electron to release. Lifetime > 10³⁴ years (experimental bound) follows from fused-barrier topology.'
));

children.push(H2('4.6  Open Questions'));

children.push(OPEN_Q(
  'Why three zones and not two or four?',
  'Three-zone pole–equator–pole fusion matches baryon +1 charge and stability in this model. Meson (quark–antiquark) and exotic baryons are outside this minimal picture — open extension.'
));

children.push(PAGEBREAK());

// =====================================================================
// CHAPTER 5: THE NEUTRON
// =====================================================================
children.push(H1('Chapter 10: The Neutron'));

children.push(BOX([
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { line: 360, after: 0 },
    children: [new TextRun({
      text: 'Neutron = proton + trapped electron + outer observation photon γ_obs. Outer layer pins the inner pump; pressure builds until escape at P_c — primary lifetime mechanism (C3). Beta decay releases electron and effective neutrino channel.',
      size: 22, italics: true, font: 'Georgia',
    })],
  }),
]));

children.push(H2('5.1  Layered Structure'));

children.push(...DIAGRAM([
  '',
  '                    Free neutron',
  '                 ╔═════════════════════╗',
  '                 ║   γ_obs (outer)     ║  ← constant observation',
  '                 ║  ┌───────────────┐  ║',
  '                 ║  │ electron coin │  ║  ← inner pump still runs',
  '                 ║  │  ∿ inner ∿    │  ║',
  '                 ║  └───────────────┘  ║',
  '                 ║      fused proton   ║',
  '                 ╚═════════════════════╝',
  '',
], 'Figure 5.1: Neutron = p + e⁻ + γ_obs (P4-1).'));

children.push(H2('5.2  Mass Excess'));

children.push(EQ_NUM('m_n c² = 939.565 MeV', '5.1'));
children.push(EQ_NUM('m_p c² = 938.272 MeV', '5.2'));
children.push(EQ_NUM('Δm c² = 1.293 MeV', '5.3'));
children.push(EQ_NUM('Q_β = Δm c² − m_e c² ≈ 0.782 MeV', '5.4'));

children.push(PR('Q_β is the beta-decay energy release — matches outer-layer + binding bookkeeping (**E** anchor).'));

children.push(H2('5.3  Magnetic Moment'));

children.push(EQ_NUM('μ_n = −1.913 μ_N', '5.5'));
children.push(EQ_NUM('μ_n = −g_{eff} (2m_e/m_n) μ_N', '5.5a'));

children.push(PR(
  'Negative μ_n from trapped-electron leakage dominating screened proton contribution — ',
  { text: 'PARTIAL', italics: true },
  ' fit (O4-4); g_eff ≈ 1.76×10³.'
));

children.push(H2('5.4  Beta Decay'));

children.push(EQ_NUM('n → p + e⁻ + ν̄_e', '5.6'));

children.push(PR(
  'Outer γ_obs escape maps to the effective neutrino channel (MODEL O4-3). Lepton number and energy conserved; Q_β ≈ 0.782 MeV.'
));

children.push(H2('5.5  Free Lifetime — Pressure Escape (C3)'));

children.push(PR(
  { text: 'Primary mechanism (C3):', bold: true },
  ' outer observation interrupts the inner pump → pressure P(t) rises → escape at P_c.'
));

children.push(EQ_NUM('t_{escape} ≈ (P_c − P_0)/(α Γ_{obs}) ~ τ_n', '5.7'));

children.push(PR(
  'Calibrated to τ_n ≈ 879 s (bottle/beam averages differ by ~10 s). Standard Model Fermi theory gives comparable Γ — useful ',
  { text: 'comparison layer', italics: true },
  ', not the primary geometric derivation in this book.'
));

children.push(H2('5.6  Stability in Nuclei'));

children.push(PR(
  'Inside nuclei, network pressure sharing and entanglement with neighboring neutrons suppress single-particle escape — free τ_n pathology is lifted. N/Z valley stability follows balance of shared pressure channels (MODEL).'
));

children.push(H2('5.7  Open Questions'));

children.push(OPEN_Q(
  'Bottle vs beam τ_n discrepancy (~10 s)?',
  'Different environmental Γ_obs and entanglement coupling may shift effective P_c slightly. Testable by controlled surroundings — MODEL prediction.'
));

children.push(PREDICTION(
  'Higher ambient electron density (plasma, dense gas) may weakly extend free-neutron lifetime by altering Γ_obs — small effect, neutron-storage experiment.'
));

children.push(PAGEBREAK());

(async () => {
  const doc = buildDoc('III', 'Part III — Particles (Ch 8–10)', children);
  await saveDoc(doc, 'book/output/03_Particles_Electron_Proton_Neutron.docx');
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
