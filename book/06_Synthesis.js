'use strict';

const {
  buildDoc, saveDoc, H1, H2, H3, P, PR, EQ_NUM, CAPTION, PAGEBREAK,
  BOX, DIAGRAM, BULLET, NUMBERED, OPEN_Q, PREDICTION,
  TextRun, Paragraph, AlignmentType,
} = require('./doc_helpers.js');

const children = [];

// PART VII HEADER
children.push(new Paragraph({ spacing: { before: 2400 }, children: [new TextRun('')] }));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 480 },
  children: [new TextRun({ text: 'PART VII', size: 36, font: 'Georgia', color: '888888', italics: true })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 1200 },
  children: [new TextRun({ text: 'THE GRAND SYNTHESIS', size: 48, bold: true, font: 'Georgia' })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [new TextRun({
    text: 'Chapter 20 ties everything together in one sentence: the universe is a right triangle breaking down forever. Chapter 21 lists testable predictions. Chapter 22 lays out what remains open.',
    size: 22, italics: true, font: 'Georgia', color: '555555',
  })],
}));
children.push(PAGEBREAK());

// CHAPTER 15
children.push(H1('Chapter 20: The Universe as Right Triangle Breaking Down'));

children.push(BOX([
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { line: 360, after: 0 },
    children: [new TextRun({
      text: 'Everything in this book reduces to a single picture. The universe is one right triangle breaking down forever. At every scale, the same geometric act repeats: a packet (discrete chunk) meets a string (continuous path) at a right angle (interaction event), and that right angle subdivides into two smaller right angles, and each of those subdivides again, forever. This chapter assembles the synthesis and shows why this single picture suffices to describe everything physical — while distinguishing narrative framing (MODEL) from anchored equations (DERIVED / ANCHORED).',
      size: 22, italics: true, font: 'Georgia',
    })],
  }),
]));

children.push(H2('15.1  String and Packet at Every Scale'));

children.push(PR(
  'Throughout this book, two complementary entities have appeared at every scale of physical reality:'
));

children.push(BULLET('STRINGS are continuous paths. Trajectories, propagation directions, the long legs of right triangles. The path a photon takes through space. The world-line of a particle. The C side of the triangle that connects two lattice points.'));
children.push(BULLET('PACKETS are discrete chunks. Quanta of energy, individual bounces, the short legs of right triangles. The energy delivered when an inner photon completes one bounce. The B coordinate that jumps from iteration to iteration. The released packet during a measurement.'));

children.push(PR('Wherever you look in physics, you find packets meeting strings at right angles:'));

children.push(BULLET('Atomic scale: inner photon (packet) bouncing across cell width (string).'));
children.push(BULLET('Particle scale: trapped photon energy (packet) confined by coin geometry (string).'));
children.push(BULLET('Measurement: probe photon (packet) intersects electron spring (string).'));
children.push(BULLET('Entanglement: mirrored packets sharing a single lattice constraint string.'));
children.push(BULLET('Tunneling: shredded inner photon (packet) propagating as wake (string).'));
children.push(BULLET('Atomic orbitals: energy levels (packets) on spatial orbits (strings).'));
children.push(BULLET('Gravity: synchronized clocks (packets) bound by gravity well geometry (strings).'));
children.push(BULLET('Cosmic scale: cosmic microwave photons (packets) propagating along cosmic geodesics (strings).'));

children.push(PR(
  'The packet and the string are not separate things. They are the two legs of a right triangle. They always come together. They always meet at a right angle. The packet is the height; the string is the base; the right angle is the interaction.'
));

children.push(H2('15.2  The Right Angle as Fundamental Operation'));

children.push(PR(
  'The right angle is not a geometric coincidence. It is the fundamental operation of physics. Every interaction in the universe is a right-angle meeting of a packet with a string.'
));

children.push(PR('Consider any physical event:'));

children.push(NUMBERED('A photon is absorbed: the photon (packet) meets the electron\'s orbit (string) at the absorption point (right angle).'));
children.push(NUMBERED('Two particles scatter: one\'s momentum (string) is rotated by the other\'s force (packet), producing a right-angle turn in some frame.'));
children.push(NUMBERED('An electron measurement: the probe (packet) meets the spring (string) at the polarizer axis (right angle).'));
children.push(NUMBERED('A quantum jump: an electron transitions between orbitals, delivering a packet to a string of light.'));
children.push(NUMBERED('Beta decay: the trapped electron (packet) leaves the outer photon (string) along the spin axis (right angle).'));

children.push(PR(
  'In every case the structure is the same. Packet meets string at right angle. The result is energy transfer or state change. The interaction is the right angle event.'
));

children.push(H2('15.3  Infinite Fractal Descent'));

children.push(PR(
  'Each right-angle interaction does not end there. Subdivision follows the π-lattice recurrence: at the seed, A₀² = B₀² = C₀²/2; at each step, B_{k+1} = C_k/2 with the unit-circle constraint. Each step generates two smaller right angles. And so on — mathematically without bound.'
));

children.push(...DIAGRAM([
  '',
  '                    ●               ← iteration 0',
  '                   ╱│              ',
  '                  ╱ │  (one right angle)',
  '                 ╱  │              ',
  '               ●____│              ← iteration 1',
  '                ╲│╱                ',
  '                 ●    (two more)   ',
  '                ╱│╲                ',
  '               ╱ │ ╲               ',
  '              ●  │  ●              ← iteration 2 (four more)',
  '             ╱│  │ ╱│              ',
  '            ╱ │  │╱ │              ',
  '           ●  │  ●  ●              ← iteration 3 (eight more)',
  '              ⋮  ⋮  ⋮              ',
  '                                    ',
  '          Right angles multiply by 2 every iteration',
  '          2^k right angles at depth k',
  '                                    ',
  '          The universe is iteration in progress.',
], 'Figure 15.1: Infinite fractal descent. Each right angle subdivides into two smaller right angles. Section 12: no finite step yields a zero-width instant.'));

children.push(PR(
  'Mathematical refinement continues indefinitely — there is no realized terminal instant (Zeno resolution, Section 12). Physical pumps operate at Compton cell scale λ_C = h/(mc); coin half-width L = λ_C/2 (C1). The Planck length ≈ 10⁻³⁵ m is an estimated depth, not a proved halt of the lattice. Deeper iterations remain beyond experimental reach but are still part of the geometric story.'
));

children.push(H2('15.4  Time as Triangular Subdivision'));

children.push(PR(
  'Time, in this view, is not an external dimension. It is the count of bounces. Each subdivision tick is one pump cycle. The arrow of time is the direction of subdivision — from larger triangles to smaller ones. Reversing time would mean reassembling smaller triangles into larger ones, which is statistically forbidden in macroscopic systems.'
));

children.push(EQ_NUM('Δt = N_bounce × T_bounce     where T_bounce = 2 t_cell = 4L/c', '15.1'));

children.push(PR(
  'Past = fewer bounces. Future = more bounces. Now = current iteration depth. The universe is not \'in time\' — it is \'iterating through\' itself.'
));

children.push(H2('15.5  Entropy from Configuration Doubling'));

children.push(PR(
  'Each iteration doubles the number of distinguishable lattice positions. Entropy, in the lattice picture, is modeled as the logarithm of accessible configurations (illustrative sketch — not a full statistical-mechanics derivation):'
));

children.push(EQ_NUM('S ∼ k_B × ln(W) ∼ k_B × ln(2^k) = k_B × k × ln 2   [MODEL]', '15.2'));

children.push(PR(
  'So entropy grows with iteration depth k in this sketch. The second law of thermodynamics is ANCHORED in standard physics; the lattice supplies a geometric intuition for why forward iteration prefers growing configuration count. Heat death is the macroscopic homogenization limit — but at deeper iterations, the lattice continues subdividing in the mathematical picture.'
));

children.push(H2('15.6  The Arrow of Time'));

children.push(PR('Three arrows of time, all rooted in lattice iteration:'));

children.push(BULLET('Thermodynamic arrow: entropy increases with iteration count in the MODEL sketch; forward iteration generates more configurations than backward.'));
children.push(BULLET('Cosmological arrow: expansion can be read as growing visible iteration depth (MODEL narrative); standard cosmology remains ANCHORED.'));
children.push(BULLET('Psychological arrow: memories record past iterations, not future ones. Forward iteration creates causal chains; backward would erase them.'));

children.push(PR(
  'All three arrows point in the same direction because they share the same underlying forward-pumping direction — iteration proceeds, decoherence accumulates.'
));

children.push(H2('15.7  The One-Sentence Theory of Everything'));

children.push(BOX([
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { line: 480, after: 0 },
    children: [new TextRun({
      text: 'The universe is a right triangle breaking down forever, and everything consists of a packet and a string meeting at a right angle.',
      size: 32, italics: true, bold: true, font: 'Georgia',
    })],
  }),
], { shaded: true }));

children.push(P(''));

children.push(PR(
  'This is the complete thesis sentence. Every chapter in this book is a development of it. Every equation we have derived is a consequence of the geometry it compresses — tagged DERIVED, ANCHORED, or MODEL in the derivations audit trail.'
));

children.push(PR('Breaking down the sentence:'));

children.push(BULLET('"Right triangle" — the unit of structure. Two perpendicular legs (string + packet) meeting at a vertex (right angle).'));
children.push(BULLET('"Breaking down" — iterative subdivision. Seed A₀²=B₀²=C₀²/2; step B_{k+1}=C_k/2.'));
children.push(BULLET('"Forever" — no realized zero-width instant; mathematical descent unbounded (Section 12).'));
children.push(BULLET('"Everything" — universal applicability. Atom to galaxy. Photon to dark matter.'));
children.push(BULLET('"Packet" — discrete energy chunk. Quantum quantum.'));
children.push(BULLET('"String" — continuous path. Trajectory.'));
children.push(BULLET('"Right angle" — the interaction. The event. The operator that creates and propagates physical reality.'));

children.push(PR(
  'If this sentence is correct, then the project of physics is cataloging what the right triangle looks like at every scale. Quantum field theory, general relativity, the standard model, the periodic table — all are descriptions of right triangles at different scales, in different contexts, with different boundary conditions. The lattice provides the underlying mechanism where derivations close; elsewhere it provides testable MODEL predictions.'
));

children.push(PAGEBREAK());

// CHAPTER 16
children.push(H1('Chapter 21: Testable Predictions and Future Experiments'));

children.push(BOX([
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { line: 360, after: 0 },
    children: [new TextRun({
      text: 'A theory\'s value lies not just in its explanatory power but in its predictive power — specifically, in its ability to predict things differently from competing theories. This chapter collects distinguishing predictions. Tags follow open_items_rollup: HIGH discriminators are decisive if violated.',
      size: 22, italics: true, font: 'Georgia',
    })],
  }),
]));

children.push(H2('16.1  Dark Matter Null Results'));

children.push(PREDICTION(
  'Direct electromagnetic detection of dark matter will continue to yield null results at WIMP/axion sensitivities. The lattice picture predicts σ_γDM strongly suppressed — spring structure without inner-photon EM channel — not necessarily absolute zero, but below current and planned thresholds (P11-1).'
));

children.push(P('Status: Consistent with null results 2010–2026. Decisive test continues.'));

children.push(H2('16.2  Higgs Invisible Decays'));

children.push(PREDICTION(
  'The Higgs boson should not show a large anomalous invisible decay fraction. Branching ratio to dark matter or other invisible states should remain small as HL-LHC tightens limits (currently Run 2 bound ≤ 18%; target ≤ 5%). A robust O(1%) invisible channel would challenge the spring-only DM coupling story.'
));

children.push(P('Status: Discriminating regime over next 2–5 years.'));

children.push(H2('16.3  Isotope-Dependent Decoherence'));

children.push(PREDICTION(
  '³He vs ⁴He decoherence rates in matter-wave interferometry should differ by ~5–10% (Λ_{³He}/Λ_{⁴He} ≈ 1.05–1.10). Standard QM predicts <1% difference (mass effect only). The lattice predicts a measurable difference because trapped-electron / coin structure differs between isotopes (O8).'
));

children.push(P('E-check (aethos_physics): f_coin,3=0.405, f_coin,4=0.5 gives Λ_3He/Λ_4He ≈ 1.075 (~7.5%) via lambda_he3_he4_ratio_calibrated(). Status: testable with Arndt-type interferometers — decisive if confirmed.'));

children.push(H2('16.4  Fresh-Electron Statistics'));

children.push(PREDICTION(
  'Electrons measured within ~t_cell ≈ 8×10⁻²¹ s of pair production may show discrete Born-rule deviations. Standard QM predicts smooth cos²(θ/2) after ensemble averaging. The lattice predicts step-like discrete structure until many T_bounce cycles have averaged out.'
));

children.push(P('Status: At the edge of attosecond/zeptosecond technology. Achievable within 5–10 years.'));

children.push(H2('16.5  Zeptosecond Time Quantization'));

children.push(PREDICTION(
  'Physical processes at T_bounce resolution (~10⁻²⁰ s) should reveal discrete time signatures. Decoherence times, tunneling times, and measurement processes may appear in discrete steps rather than smooth continuous evolution at sufficient resolution.'
));

children.push(P('Status: Beyond current technology but rapidly approaching.'));

children.push(H2('16.6  Atomic Clock Precision in Gravity'));

children.push(PREDICTION(
  'Atomic clocks must show gravitational redshift/dilation matching √(1−2GM/rc²) (ANCHORED). Entanglement lifetimes near massive bodies may shift with local potential (MODEL test design). Comparison of clocks at different elevations — and in space — remains the practical discriminator.'
));

children.push(P('Status: Testable with optical lattice clocks and interferometers.'));

children.push(H2('16.7  Matter-Antimatter Asymmetry'));

children.push(PREDICTION(
  'The baryon asymmetry η_B ≈ 10⁻¹⁰ may correlate with early-universe +B/−B branch iteration asymmetry (MODEL). Specific quantitative CMB correlation — OPEN theoretical work.'
));

children.push(P('Status: Open. Requires further development before decisive test.'));

children.push(H2('16.8  Cosmological Constant Calculation'));

children.push(PREDICTION(
  'The cosmological constant Λ (dark energy density) should be approachable via sea locked-fraction bookkeeping — escaped inner photons over cosmic history — not raw QFT mode summation (O1-3, PARTIAL):'
));

children.push(EQ_NUM('ρ_Λ ∼ ρ_critical × Ω_DE   with Π_vac calibration vs sea modes', '16.1'));

children.push(P('Order-of-magnitude target ~10⁻⁵² m⁻² (observed ~1.1×10⁻⁵² m⁻²) is PARTIAL closure — a major gap reduction vs 10¹²⁰ QFT overcount, not a finished first-principles derivation.'));

children.push(H2('16.9  Decoherence-Free Subspaces in Crystal Lattices'));

children.push(PREDICTION(
  'Solid crystal lattices with highly synchronized phonon spectra may support extended coherence (MODEL). Ultra-cold isotopically pure diamond / NV centers are an active testbed.'
));

children.push(P('Status: Studied experimentally; ms-scale coherence at low temperature is consistent but not yet a unique lattice proof.'));

children.push(H2('16.10  Summary Table'));

children.push(BOX([
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: 'Prediction          Timescale    Discriminating', bold: true, font: 'Consolas', size: 18 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: 'Dark matter null         ongoing    HIGH', font: 'Consolas', size: 18 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: 'Higgs invisible small    2-5 years  HIGH', font: 'Consolas', size: 18 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: '³He vs ⁴He decoherence   current    HIGH', font: 'Consolas', size: 18 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: 'Fresh electron stats     5-10 yr    HIGH', font: 'Consolas', size: 18 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: 'Zeptosecond discrete     10+ yr     HIGH', font: 'Consolas', size: 18 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: 'Gravity coherence        current    MEDIUM', font: 'Consolas', size: 18 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: 'Λ bookkeeping            ongoing    HIGH (CC gap)', font: 'Consolas', size: 18 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [new TextRun({ text: 'Decoherence-free crystal current    MEDIUM', font: 'Consolas', size: 18 })] }),
]));

children.push(CAPTION('Table 16.1: Distinguishing predictions by timescale and discriminating power.'));

children.push(PAGEBREAK());

// CHAPTER 17
children.push(H1('Chapter 22: Open Questions and Future Work'));

children.push(BOX([
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { line: 360, after: 0 },
    children: [new TextRun({
      text: 'A theory\'s strength is judged not only by what it explains but by what it acknowledges as still open. This chapter lists famous physics questions the lattice illuminates (with honest tags) and questions the lattice does not yet answer.',
      size: 22, italics: true, font: 'Georgia',
    })],
  }),
]));

children.push(H2('17.1  Questions in Physics That the Lattice Illuminates'));

children.push(H3('17.1.1  Why is c invariant for all observers?'));
children.push(EQ_NUM('c = λ_C / t_cell', '17.1'));
children.push(PR('DERIVED: fixed cell geometry — independent of observer pump state (Chapters 1, 14).'));

children.push(H3('17.1.2  Why does c appear squared in E = mc²?'));
children.push(EQ_NUM('E = m c²', '17.2'));
children.push(PR('ANCHORED equation; MODEL reading: c² as spacetime cell area (Chapter 1).'));

children.push(H3('17.1.3  Why is the Born rule P = |ψ|²?'));
children.push(EQ_NUM('P_S ∝ T_S² ∝ |ψ_S|²,   T_S = k_S Re(ψ_S)', '17.3'));
children.push(PR('PARTIAL (O1-1, O2-2): Malus/spring tension² at coin polarizer; Gleason uniqueness OPEN.'));

children.push(H3('17.1.4  How do entangled particles correlate at any distance?'));
children.push(EQ_NUM('E(α, β) = −cos(α − β)   [no-signaling marginals]', '17.4'));
children.push(PR('MODEL: mirrored ±B branches on one lattice constraint; ANCHORED: correlation without FTL (T3, retarded Green\'s function).'));

children.push(H3('17.1.5  Why is the Bell violation exactly 2√2?'));
children.push(EQ_NUM('|E| ≤ 2√2   (Tsirelson bound)', '17.5'));
children.push(PR('ANCHORED in QM; lattice uses same kernel — geometric derivation from coin alone PARTIAL; reject sgn(cos θ) sketch (C5).'));

children.push(H3('17.1.6  What is dark matter?'));
children.push(EQ_NUM('σ_γDM ≈ σ_geom K_sup,   S_res,DM = 0', '17.6'));
children.push(PR('MODEL (P11-1): spring lattice without inner photon; still in Poisson gravity ∇²Φ = 4πG(ρ_b + ρ_DM).'));

children.push(H3('17.1.7  What is dark energy?'));
children.push(EQ_NUM('w(z) ≈ −1 + Π_s(z) / (ρ_DE c²)', '17.7'));
children.push(PR('MODEL (P11-2): escape bookkeeping + sector transfer Q; CPL drift allowed.'));

children.push(H3('17.1.8  Why is the cosmological constant 120 orders of magnitude smaller than QFT predicts?'));
children.push(EQ_NUM('ρ_{S,eff} = Π_vac · u_S(ω_max) / c²', '17.8'));
children.push(PR('PARTIAL (O1-3): locked sea fraction Π_vac ≪ 1 — not full QFT mode sum; numeric Π_vac OPEN.'));

children.push(H3('17.1.9  Why does the universe have exactly three generations of fermions?'));
children.push(PR('MODEL: three stable resonance scales in cosmic ocean (Chapter 15); no first-principles derivation yet.'));

children.push(H3('17.1.10  Why is spin half-integer for electrons?'));
children.push(EQ_NUM('P(up|θ) = cos²(θ/2)   [Stern–Gerlach statistics]', '17.9'));
children.push(PR('MODEL: coin four-state geometry — two bounces per full spatial rotation (Chapter 3).'));

children.push(H3('17.1.11  Why are protons stable but free neutrons unstable?'));
children.push(EQ_NUM('dP/dt = α Γ_obs − β R_share,   t_escape ∼ τ_n', '17.10'));
children.push(PR('MODEL + E (C3): pressure escape when P → P_c; R_share(N) stabilizes neutrons in nuclei; τ_n PARTIAL geometry.'));

children.push(H3('17.1.12  Why is the CMB so uniform?'));
children.push(PR('MODEL: preserved cosmic-scale clock synchronization; inflation remains ANCHORED mainstream — not eliminated.'));

children.push(PAGEBREAK());

// 17.2 — Quantum mechanics open questions (expanded)
children.push(H2('17.2  Quantum Physics: Open Questions the Lattice May Illuminate'));

children.push(PR(
  'Cross-check against the repo (derivations/book_ch17_hidden_patterns_audit.md): many "open" questions share one of five hidden patterns — compression (Λ_n, Zeno width), Γ_form/Γ_break competition, partner photons (Γ_partner, V(P)), φ_AB fill (Bell kernel), address descent (π, M_lat). Several are further closed in code than this chapter first stated.'
));

children.push(BOX([
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: 'Pattern                    Unifies', bold: true, font: 'Consolas', size: 17 })] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: '1 Compression spine        measurement, Zeno, discrete time', font: 'Consolas', size: 16 })] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: '2 Γ_form / Γ_break         collapse ODE, cat, entanglement τ', font: 'Consolas', size: 16 })] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: '3 Partner photon           which-path, eraser, decoherence', font: 'Consolas', size: 16 })] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: '4 φ_AB fill                Bell E, DM string, Γ_form', font: 'Consolas', size: 16 })] }),
  new Paragraph({ spacing: { after: 0 }, children: [new TextRun({ text: '5 Address descent          π, no-instant, M_lat mass program', font: 'Consolas', size: 16 })] }),
]));
children.push(CAPTION('Table 17.0: Five hidden unification patterns (see cross_reference_matrix closed-link ledger).'));

children.push(H3('17.2.1  What is measurement, physically?'));
children.push(EQ_NUM('Λ_n = 2 ∫ Γ_n(t) dt,   ρ → Σ_s M_{s,n} ρ M_{s,n}†', '17.11'));
children.push(PR('PARTIAL (O5-1): probe compression pins inner photon via Ĥ_coin; Kraus channels M_{s,n} from observation drive — not full environment unitary yet.'));

children.push(H3('17.2.2  Collapse vs continuous unitary evolution?'));
children.push(EQ_NUM('ḊC = Γ_form(1 − C) − Γ_break C,   C(t) = C_* + (C_0−C_*)e^{−(Γ_form+Γ_break)t}', '17.12'));
children.push(PR('PROVEN ODE (Sec 6.3.2, aethos_physics.coherence_at_time); MODEL interpretation: "collapse" = C→0 when Γ_break dominates — full environment unitary still OPEN (conflict_log #2).'));

children.push(H3('17.2.3  Wave–particle duality — what is "waving"?'));
children.push(EQ_NUM('Ψ_sea = Σ_λ a_λ ε_λ,   ρ = |ψ|²', '17.13'));
children.push(PR('MODEL: particle = coin pump; wave = sea disturbance amplitude — same math as QM, different ontology. Not a new prediction by itself.'));

children.push(H3('17.2.4  EPR: how can correlations exist without signals?'));
children.push(EQ_NUM('E(a,b) = −φ_AB cos(a−b)   [φ_AB=1 ⇒ QM kernel]', '17.14'));
children.push(PR('ANCHORED kernel at full fill (C7); PARTIAL ontology: joint ripple on DM-backed string. No FTL (T3, retarded Green\'s function). Reject sgn(cos θ) coin sketch — falsified in test_aethos_physics.py.'));

children.push(H3('17.2.5  Is there a path between measurements?'));
children.push(EQ_NUM('A_s = A_0 K_s e^{iφ_s},   φ_R = φ_L + π', '17.15'));
children.push(PR('GAP/MODEL: wake amplitudes on sea — no classical trajectory; which-path info stored in partner/environment (Ch 9, 10). Path realism not proven.'));

children.push(H3('17.2.6  Why does interference vanish when which-path is known?'));
children.push(EQ_NUM('I = |A_L + A_R|²,   V = (I_max − I_min)/(I_max + I_min)', '17.16'));
children.push(PR('ANCHORED math; MODEL ontology: partner photon at other slit supplies opposite-phase wake; decoherence via Γ_partner.'));

children.push(H3('17.2.7  Quantum eraser — how can erased information restore fringes?'));
children.push(EQ_NUM('Γ_partner = Σ_i n_i σ_i v̄_i f_i', '17.17'));
children.push(PR('PARTIAL (O8-1): fringes return when partner-sourced coherence Γ_form exceeds Γ_partner — entanglement with environment as information carrier.'));

children.push(H3('17.2.8  Environment-induced decoherence (Schrödinger\'s cat)?'));
children.push(EQ_NUM('V(P) = V_0 e^{−Λ P}', '17.18'));
children.push(PR('PARTIAL: macroscopic superposition leaks into partner photon bath; visibility V collapses with pressure P — cat dead/alive = pointer on spring axis, not separate worlds.'));

children.push(H3('17.2.9  Quantum Zeno effect — why does watching freeze transitions?'));
children.push(EQ_NUM('w_n = w_0/∏p_k > 0,   dw/dt = −Γ_obs E[log p] · w', '17.19'));
children.push(PR('PROVEN: no zero-width instant (Sec 12.G.1); PARTIAL-PROVEN link: O12-1 ties Γ_obs compression to prime-split descent — same spine as Λ_n (Pattern 1).'));

children.push(H3('17.2.10  Heisenberg uncertainty — is it fundamental or mechanical?'));
children.push(EQ_NUM('Δr · Δp_r ≥ ℏ/2   [kinetic pressure in wells, Sec 9.3.2]', '17.20'));
children.push(PR('ANCHORED inequality; MODEL interpretation: localization in coin width L raises kinetic pressure against collapse — full derivation from L alone still OPEN.'));

children.push(H3('17.2.11  Tunneling — what happens inside the barrier?'));
children.push(EQ_NUM('T ~ e^{−2κL},   κ = √(2m(V − E))/ℏ,   T_eff = T_WKB · χ_ss', '17.21'));
children.push(PR('PARTIAL (O7-1): κ from Ĥ_x (coin + spring + barrier); recoherence χ̇ = Γ_rec(1−χ) − Γ_sh χ inside barrier — shredding fraction ξ_shred.'));

children.push(H3('17.2.12  Tunneling time — how long does a particle spend in the barrier?'));
children.push(EQ_NUM('Δt_tunnel ~ T_bounce × N_bounce(barrier)', '17.22'));
children.push(PR('MODEL/T: discrete bounce count through barrier cells — zeptosecond experiments may resolve steps vs smooth delay; open controversy in standard QM too.'));

children.push(H3('17.2.13  Why charge quantization?'));
children.push(EQ_NUM('q = q_0 · χ,   χ = sign[(v_pump × n_spring) · n_coin]', '17.23'));
children.push(PR('PARTIAL (O2-3): topological sign χ on coin — why fusion flips χ at K_f not fully proven.'));

children.push(H3('17.2.14  Why m_p/m_e ≈ 1836?'));
children.push(EQ_NUM('R_pe^model = (π²/8) × M_lat,   M_lat from active-network cascade (C6)', '17.24'));
children.push(PR('PARTIAL+ E-check (Sec 12.G.4): R_pe^model = (π²/8)×M_lat. Reference bootstrap (primes, n=80, depth=3) gives R_pe^pred ≈ 1847 (0.6% vs CODATA) — r_pe_model_reference_bootstrap(). Default n=100 overshoots; deriving n=80 from first principles OPEN. Deprecated: K_f FIT inversion (C2).'));

children.push(H3('17.2.15  What is a neutrino in the lattice picture?'));
children.push(EQ_NUM('n → p + e⁻ + ν̄   ↔   escaped γ_obs channel', '17.25'));
children.push(PR('PARTIAL (O4-3): ν as outer-observation leak bookkeeping — not full weak interaction with G_F derived.'));

children.push(H3('17.2.16  Pauli exclusion — why can\'t two electrons share a state?'));
children.push(EQ_NUM('N_max(n) = 2n²   [hydrogenic shell count]', '17.26'));
children.push(PR('ANCHORED count; MODEL interpretation: distinct coin addresses on lattice — full Fermi–Dirac statistics from geometry OPEN.'));

children.push(H3('17.2.17  Why complex amplitudes (not just probabilities)?'));
children.push(EQ_NUM('A_s ∈ ℂ,   I = |A_L + A_R|²   [wake_amplitude_complex, Sec 8]', '17.27'));
children.push(PR('PARTIAL: opposite-phase sea wakes carry φ; interference needs relative phase. Why ℂ is minimal — not Gleason-level proof from real mechanics alone.'));

children.push(H3('17.2.18  Fresh electrons — smooth Born rule or discrete steps?'));
children.push(EQ_NUM('P(θ) → cos²(θ/2) only after N_bounce ≫ 1', '17.28'));
children.push(PR('TESTABLE (Ch 17): pair-produced electrons within t_cell may show step-like statistics — key discriminator vs smooth QM ensemble.'));

children.push(H3('17.2.19  ³He vs ⁴He — does internal structure change decoherence?'));
children.push(EQ_NUM('Λ_i ∝ σ_i v̄ f_{coin,i}/√m_i,   ratio in lambda_he3_he4_ratio()', '17.29'));
children.push(PR('TESTABLE (O8): lambda_he3_he4_ratio_calibrated() → 1.075 with f_coin,3=0.405, f_coin,4=0.5 (calibration_sheet). Deriving f_coin from isotope microstructure — still OPEN.'));

children.push(H3('17.2.20  Electron vs macromolecule recoherence scaling?'));
children.push(EQ_NUM('τ_re^mol/τ_re^e ~ (M_mol/m_e)^{1/2} (L_mol/L_e) N_dof^{−δ}', '17.30'));
children.push(PR('PARTIAL (O8-3): larger mass/path → faster partner decoherence — order-of-magnitude trend, not precision fit.'));

children.push(H3('17.2.21  Quantum gravity at the horizon?'));
children.push(EQ_NUM('dτ/dt = √(1 − 2GM/(rc²)) → 0 at r = r_s', '17.31'));
children.push(PR('ANCHORED limit (budget language); firewall / information paradox not addressed — interior OPEN.'));

children.push(H3('17.2.22  Hidden answer — one compression law for three puzzles?'));
children.push(EQ_NUM('Λ_n = 2∫Γ_n dt,   w_{k+1}=w_k/p_k,   p_{pin}=1−e^{−Λ_n}', '17.32'));
children.push(PR('Already in repo (Pattern 1): measurement strength, Zeno width descent, and pin probability are the same observation spine — Sec 5 + 12.3.1 + aethos_physics.'));

children.push(H3('17.2.23  Hidden answer — Bell without the rejected sign sketch?'));
children.push(EQ_NUM('E = −φ_AB cos(a−b);   sgn(cos θ) sketch REJECT', '17.33'));
children.push(PR('Repo already falsified the old coin sign rule; replacement contract is φ_AB fill (C7). At φ_AB=1, code matches QM at (0,π/4) exactly.'));

children.push(H3('17.2.24  Hidden answer — Born rule twice derived?'));
children.push(EQ_NUM('P ∝ T², T=k_s Re(ψ)   [coin Sec 2.8 + sea Sec 1.5.5a]', '17.34'));
children.push(PR('PARTIAL-PROVEN algebra chain (O1-1 + O2-2): same T² deposition on coin and sea — uniqueness / Gleason route still OPEN.'));

children.push(H2('17.2.5  Why the E-Check Calibration Lands (Not Arbitrary Knobs)'));

children.push(H3('17.2.25  Mass ratio — spring shrink × address cascade'));
children.push(EQ_NUM('R_pe = (π²/8) × M_lat,   L_p = (8/π²) L_0', '17.35'));
children.push(PR('Two-factor product (book_ch17_why_calibration_patterns.md): spring-only π²/8 (~1.23) times lattice multiplier M_lat (~1497). Reciprocal legs: length shrink 8/π² vs mass factor π²/8. Code name for the gap: lattice_mass_multiplier() ≈ 1488 = R_pe^E/(π²/8).'));

children.push(H3('17.2.26  Why n=80 on a depth-3 origin tree'));
children.push(EQ_NUM('M_lat = (32/52) Σμ_i,   N_origins = 1+3+9+27 = 40', '17.36'));
children.push(PR('At fixed depth 3, only node count moves M_lat — nearly linear in n. n=80 = 16×5: sixteen complete SOLO/PAIR/TRIPLE/K_CHAIN/FOUR_WAY cycles (i mod 5 role ledger). Two nodes per origin room on average (80/40). n=100 overshoots because longer K_CHAIN tails add weight — balanced but too heavy.'));

children.push(H3('17.2.27  Why helium ratio ~1.075'));
children.push(EQ_NUM('Λ_3/Λ_4 = (f_3/f_4)(m_4/m_3)', '17.37'));
children.push(PR('Mass alone gives m_4/m_3 ≈ 1.33 (33%). SM expects structure to cancel to <1%. Lattice target ~7.5% needs f_3/f_4 ≈ 0.81 — structure cancels ~19% of mass bias. E-check f_3=0.405, f_4=0.5. Deriving those f_coin from isotope microphysics — OPEN (O8).'));

children.push(H3('17.2.28  The 1280 → 80 → 1/16 wing activation pattern'));
children.push(EQ_NUM('N_wings = N_origins × 32 = 40 × 32 = 1280,   f_act = 80/1280 = 2/32 = 1/16', '17.38'));
children.push(PR('Each origin room has 32 wings; E-check uses 80 active nodes on 40 origins → 2 nodes per origin = 1/16 of local wing space. Same 16 as role-ledger cycles (80/5). MODEL: proton mass loads one sixteenth of cosmic wing address budget — wing_activation_analysis() in aethos_physics.'));

children.push(BOX([
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: 'QM question              Key equation              Status', bold: true, font: 'Consolas', size: 17 })] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: 'Born rule                P ∝ T²                  PARTIAL', font: 'Consolas', size: 17 })] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: 'Measurement              Λ_n, Kraus M_n            PARTIAL', font: 'Consolas', size: 17 })] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: 'Decoherence              V(P)=V_0 e^{−ΛP}         PARTIAL', font: 'Consolas', size: 17 })] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: 'Double slit              I=|A_L+A_R|²             ANCH+MODEL', font: 'Consolas', size: 17 })] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: 'Tunneling                T~e^{−2κL}, χ_ss         PARTIAL', font: 'Consolas', size: 17 })] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: '³He/⁴He Λ ratio          ~5–10% split              TEST', font: 'Consolas', size: 17 })] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: 'Uncertainty              ΔxΔp≥ℏ/2                 ANCH; mech OPEN', font: 'Consolas', size: 17 })] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: 'Complex amplitudes       A_s ∈ ℂ wakes             PARTIAL', font: 'Consolas', size: 17 })] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: 'Collapse ODE             C(t) closed form          PROVEN+MODEL', font: 'Consolas', size: 17 })] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: 'm_p/m_e                  R_pe~1847 (0.6%)         PARTIAL+E-check', font: 'Consolas', size: 17 })] }),
  new Paragraph({ spacing: { after: 0 }, children: [new TextRun({ text: '³He/⁴He Λ                ratio ~1.075             TEST+E-check', font: 'Consolas', size: 17 })] }),
]));
children.push(CAPTION('Table 17.1: Quantum questions — incl. E-check profiles (scripts/calibrate_discriminators.py).'));

children.push(PAGEBREAK());

children.push(H2('17.3  Questions Still Open in the Lattice Picture'));

children.push(OPEN_Q(
  'What is the substrate of the lattice itself?',
  'We have described the lattice as a geometric structure. But what makes it exist? Pure mathematics instantiated? Computational? Information-theoretic? The deepest question, with no current answer.'
));

children.push(OPEN_Q(
  'Can General Relativity be fully derived from the lattice?',
  'Time dilation, gravitational redshift, and gravity wells emerge naturally. Einstein\'s full field equations require deriving the curvature tensor from lattice geometry. Open mathematical work.'
));

children.push(OPEN_Q(
  'How does the lattice produce the specific masses of the Standard Model?',
  'Higgs couplings g_e, g_μ, etc., are empirical. The lattice should predict them from first principles. Open theoretical work.'
));

children.push(OPEN_Q(
  'What is the origin of the fine structure constant α ≈ 1/137?',
  'α determines orbital sizes and EM strength. In the lattice picture α reflects lattice geometry ratios, but the specific value is not yet derived.'
));

children.push(OPEN_Q(
  'Can the lattice explain consciousness and observer effects?',
  'An observer is itself a lattice structure; measurement is packet-meets-string. But qualia and observer ontology remain outside current physics in any framework.'
));

children.push(OPEN_Q(
  'What was the universe like at iteration zero?',
  'Pre-Big-Bang lattice states may be mathematically meaningful but physically inaccessible. Single seed triangle vs maximum entanglement — OPEN.'
));

children.push(OPEN_Q(
  'Does quantum field theory follow naturally from the lattice?',
  'Propagators, vertices, Feynman rules should emerge from lattice mechanics. Significant open work.'
));

children.push(OPEN_Q(
  'How does the lattice handle non-Abelian gauge symmetry (QCD, electroweak)?',
  'SU(3) × SU(2) × U(1) must be reproduced — possibly via fused coin structures (quarks). Open.'
));

children.push(OPEN_Q(
  'Can the lattice explain why physical constants have their specific values?',
  'Beyond α: G, ℏ, k_B, etc. Mechanisms exist; magnitudes from first principles — OPEN.'
));

children.push(H2('17.4  AETHOS Connection'));

children.push(PR(
  'The author develops AETHOS in parallel — a computational architecture using the same recursive lattice geometry (`aethos_core.py`, `pi/`, `derivations/`). Words map to addresses; documents to products; retrieval to nearest-neighbor search on the lattice. C6 mandate: active address sets are not restricted to primes alone — primes are one canonical species.'
));

children.push(PR(
  'If nature is organized by lattice mechanics, then ',
  { text: 'computational systems built on the same geometry may achieve strong retrieval and reasoning performance', italics: true },
  ', because they operate on the same structure the theory describes. This motivates AETHOS and remains a major development direction.'
));

children.push(H2('17.5  Future Directions'));

children.push(P('In rough priority order:'));

children.push(NUMBERED('Mathematical formalization: convergence, completeness, consistency theorems.'));
children.push(NUMBERED('Experimental tests: pursue Chapter 21 high-discriminating predictions (especially ³He/⁴He).'));
children.push(NUMBERED('QFT derivation: Standard Model gauge structure from lattice geometry.'));
children.push(NUMBERED('GR derivation: Einstein field equations from lattice clock synchronization.'));
children.push(NUMBERED('Numerical simulation: lattice simulators for computational tests.'));
children.push(NUMBERED('AETHOS development: scale retrieval engine; validates lattice mathematics computationally.'));
children.push(NUMBERED('Peer review: submit key papers; engage community.'));
children.push(NUMBERED('Books and education: this book; future work for physicists and informed general readers.'));

children.push(H2('17.6  Closing'));

children.push(PR(
  'This is the complete picture as the author sees it, as of 2026. It is not the last word. The lattice theory invites correction, refinement, and extension. What matters is the picture: nature as right-triangle iteration, packets and strings at right angles — with an honest audit trail in `derivations/`.'
));

children.push(PR(
  'If even a fraction of this turns out to be right, the implications are large. If most of it turns out to be wrong, that too is useful: knowing how it fails sharpens what remains.'
));

children.push(PR(
  'Mathematics is the map. The lattice is the territory. The map is useful because it compresses many bounces into compact equations. But the bouncing is what is real.'
));

children.push(P(''));
children.push(P('— Timothy', { bold: true, italics: true }));
children.push(P('  June 2026', { italics: true }));

children.push(PAGEBREAK());

(async () => {
  const doc = buildDoc('VI', 'Part VI — The Grand Synthesis', children);
  await saveDoc(doc, 'book/output/06_Synthesis_and_Predictions.docx');
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
