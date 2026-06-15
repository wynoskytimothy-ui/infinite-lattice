'use strict';

const {
  buildDoc, saveDoc, H1, H2, P, PR, EQ, EQ_NUM, PAGEBREAK,
  BOX, DIAGRAM, BULLET, NUMBERED, OPEN_Q, PREDICTION,
  TextRun, Paragraph, AlignmentType,
} = require('./doc_helpers.js');

const children = [];

// PART IV HEADER
children.push(new Paragraph({ spacing: { before: 2400 }, children: [new TextRun('')] }));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 480 },
  children: [new TextRun({ text: 'PART IV', size: 36, font: 'Georgia', color: '888888', italics: true })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 1200 },
  children: [new TextRun({ text: 'QUANTUM MECHANICS DEMYSTIFIED', size: 48, bold: true, font: 'Georgia' })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [new TextRun({
    text: 'Chapters 11–14 give mechanical pictures for measurement, entanglement, tunneling, and interference — infinite 3D spring plane (Part II Ch 3–7) underneath; checked against section_05–08 and C5.',
    size: 22, italics: true, font: 'Georgia', color: '555555',
  })],
}));
children.push(PAGEBREAK());

// =====================================================================
// CHAPTER 6: MEASUREMENT
// =====================================================================
children.push(H1('Chapter 11: Measurement and Observation'));

children.push(BOX([
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { line: 360, after: 0 },
    children: [new TextRun({
      text: 'Measurement is physical compression of the coin. The spring acts as an axis-selective polarizer; probabilities follow P(up|θ) = cos²(θ/2) — anchored QM, interpreted here via Malus/spring tension (partial derivation O1-1).',
      size: 22, italics: true, font: 'Georgia',
    })],
  }),
]));

children.push(H2('6.1  Measurement Is a Polarizer Cycle'));

children.push(PR(
  'Measurement is not a single mysterious collapse. It is a ',
  { text: 'sequence', bold: true },
  ' — the same physics as a polarizer, applied to the inner photon inside the coin. The observer does not passively watch; the observer\'s probe ',
  { text: 'sets the spring\'s axis', bold: true },
  ', and the spring decides what packets get released and what stays trapped.'
));

children.push(P('The complete cycle:'));
children.push(NUMBERED('Observer sends a probe photon with momentum along compression axis n̂_obs.'));
children.push(NUMBERED('Probe push reorients the spring axis to n̂_obs (typically faster than one bounce: τ_align ~ ℏ/E_obs, compare T_bounce ≈ 2t_cell).'));
children.push(NUMBERED('Spring acts as polarizer P_n̂ = |n̂⟩⟨n̂| on the inner photon state |χ⟩.'));
children.push(NUMBERED('Inner photon passes (packet released along n̂) with P_pass = cos²(θ/2), or is blocked (trapped, opposite orientation) with P_block = sin²(θ/2).'));
children.push(NUMBERED('Detector registers the packet — not the whole electron. The coin retains residual inner-photon energy E_retained = E_inner sin²(θ/2).'));
children.push(NUMBERED('Sea / Higgs coupling returns equilibrium mass after the brief extraction — **MODEL** (without this, repeated measurements would drain m_e).'));
children.push(NUMBERED('Spring axis relaxes; inner photon resumes bouncing when compression ends.'));

children.push(PR(
  'Geometric detail (Ch 8 §8.1c): unilateral compression folds the spiraled spring onto z-plane (coin silhouette); at maximum compression the side+tension axes ',
  { text: 'flip', bold: true },
  ' — e.g. WS → BH on release. Bilateral max compression closes the cavity to a ball state (fusion threshold, Ch 9).'
));

children.push(...DIAGRAM([
  '',
  '   Before          Probe arrives        Axis locked',
  '   ╔═══════╗       ╔═══════╗           ╔═══════╗',
  '   ║   │   ║  γ    ║   ╲   ║           ║    ╲n̂║',
  '   ║   │   ║  →    ║    ╲  ║    →      ║     ╲║',
  '   ╚═══════╝       ╚═══════╝           ╚═══════╝',
  '',
], 'Figure 6.1: Compression reorients the spring polarizer.'));

children.push(H2('6.2  Born Rule / Malus Law'));

children.push(EQ_NUM('|χ⟩ = cos(θ_i/2)|↑⟩ + e^{iφ} sin(θ_i/2)|↓⟩', '6.1'));
children.push(EQ_NUM('P(up|θ) = |⟨n̂|χ⟩|² = cos²(θ/2)', '6.3'));
children.push(EQ_NUM('P(down|θ) = sin²(θ/2)', '6.4'));

children.push(PR(
  { text: 'Status:', bold: true },
  ' cos²(θ/2) is ',
  { text: 'ANCHORED', italics: true },
  ' (standard QM). Identification with classical Malus through spring tension² ∝ |ψ|² is ',
  { text: 'MODEL / PARTIAL', italics: true },
  ' (O1-1, O5-2).'
));

children.push(H2('6.2a  Born Rule as Lattice Counting (Mechanical Layer)'));

children.push(PR(
  'Circular functions ',
  { text: 'describe', italics: true },
  ' the limit; triangle iteration ',
  { text: 'explains', bold: true },
  ' it (Chapter 2.10.5). When the spring polarizer aligns to axis n̂ at angle θ from the inner photon\'s prior branch, ask: on the unit-circle address tree, what fraction of ±B leaves at depth k lie on the "pass" side of the new axis?'
));

children.push(EQ_NUM('f_{pass}(k, θ) = #{±B branches aligned with n̂} / #{all branches at depth k}', '6.2a'));
children.push(EQ_NUM('lim_{k→∞} f_{pass}(k, θ) = cos²(θ/2)   [Malus / Born envelope]', '6.2b'));

children.push(PR(
  'At k = 0 (anchor only): outcome is 0 or 1 — fully discrete. At k = 1 (three quadrant points): counts depend on θ stepwise. After many pump ticks (f_b ~ 10¹⁹ Hz, t_cell ~ 8×10⁻²¹ s) the electron has traced ~10⁴ lattice steps per attosecond and ~1.9×10⁴ steps per hydrogen 1s orbital period — enough for the envelope to look smooth. ',
  { text: 'Fresh', bold: true },
  ' electrons within ~t_cell of creation may still show stepwise statistics — **TEST** (Ch 17.8). Gleason uniqueness from geometry alone — **OPEN** (O1-1).'
));

children.push(PR(
  'Mechanical recipe: keep cos²(θ/2) as the measured limit; underneath, each measurement samples one bifurcation branch (+B or −B) set by polarizer axis. Do not delete cos from the book — show it as the continuum envelope of discrete bounce accounting.'
));

children.push(H2('6.3  Energy Bookkeeping — Why the Electron Survives'));

children.push(EQ_NUM('E_packet = E_inner × cos²(θ/2) = m_e c² × cos²(θ/2)', '6.5'));
children.push(EQ_NUM('E_retained = E_inner × sin²(θ/2)', '6.6'));
children.push(EQ_NUM('E_packet + E_retained = E_inner  (conserved)', '6.7'));

children.push(PR(
  'If the entire inner photon escaped, the electron would disintegrate. The polarizer mechanism fixes that: only a ',
  { text: 'packet', bold: true },
  ' — the component aligned with n̂ — leaves the coin toward the detector. The perpendicular component stays trapped. The detector ionizes and amplifies the packet; to the apparatus it looks like "the particle arrived," but what arrived is a packet of electron-energy, not the whole coin structure.'
));

children.push(PR(
  'Naive coin shrink d_new ∝ |sin(θ/2)| would change mass every shot. Resolution (',
  { text: 'MODEL', italics: true },
  '): the released packet is reabsorbed from the sea after detection; Higgs/sea tension restores equilibrium size. Net flow: E_inner → packet → detector → sea → coin. Mass m_e is stable across many observations because the sea is the energy reservoir.'
));

children.push(H2('6.4  What the Detector Actually Measures'));

children.push(PR(
  'A Stern–Gerlach magnet sets n̂_obs = ẑ. "Spin up" means the inner photon ',
  { text: 'passed', bold: true },
  ' the spring polarizer — a packet released toward +z. "Spin down" means it was ',
  { text: 'blocked', bold: true },
  ' along +z; the packet went the other way (or no packet on that arm). Each arm clicks when a packet hits it. The electron (spring + residual inner photon) continues, now polarized along the last axis. Measure the same axis again: same result. Measure a perpendicular axis: 50/50 — previous information erased because the spring re-aligned.'
));

children.push(H2('6.5  Consecutive Polarizers — The 1/4 Transmission'));

children.push(PR(
  'Three polarizers: ẑ, then x̂, then ẑ again. After the first, the inner photon is along ẑ. At x̂ the angle is 90°, so P_pass = cos²(45°) = 1/2. The ẑ-information is ',
  { text: 'gone', bold: true },
  ' — the photon is now along x̂. At the third polarizer (ẑ again), another 1/2. Total: 1/4. Classical intuition ("still ẑ-polarized, should all pass") fails because polarizer 2 ',
  { text: 'erased', bold: true },
  ' the memory. Same as unpolarized light through 0°–45°–90° sheets. No mystery — sequential polarizer action.'
));

children.push(EQ_NUM('P_total = cos²(45°) × cos²(45°) = 1/4  [ẑ → x̂ → ẑ]', '6.8'));

children.push(H2('6.6  Atomic Spectra Preview'));

children.push(PR(
  'In atoms (Chapter 15), each orbital geometry sets a different spring polarizer axis. Only packet energies allowed by that geometry escape efficiently — hence discrete spectral lines. Different orbital = different polarizer = different allowed packet sizes. **MODEL** link; quantitative E_bond calibration open.'
));

children.push(H2('6.7  Open Questions'));

children.push(OPEN_Q(
  'Is Malus derivation complete from spring geometry alone?',
  'Partial: T_S² deposition map exists (Sec 1.5.5a); Gleason uniqueness still OPEN (O1-1).'
));

children.push(PREDICTION(
  'Measurements separated by < t_cell ≈ 8×10⁻²¹ s may show incomplete polarizer alignment — discrete steps before smooth cos²(θ/2) statistics.'
));

children.push(PAGEBREAK());

// =====================================================================
// CHAPTER 7: ENTANGLEMENT
// =====================================================================
children.push(H1('Chapter 12: Entanglement'));

children.push(PR(
  'Entanglement in standard QM is a joint state vector. Here it is mirrored addresses on one unit circle and swap meets on Ψ = (z, ζ) (Ch 5): two inner photons, ±B_k partners. They do not calculate cos(α−β); geometry forces joint outcomes — **MODEL** mechanism, **ANCHORED** Bell predictions (Ch 1 §1.7, Ch 2 §2.6, Ch 5 §5.2).'
));

children.push(BOX([
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { line: 360, after: 0 },
    children: [new TextRun({
      text: 'Entanglement is dynamic: locks form (Γ_form) and break (Γ_break) continuously under ambient photons. Correlations E(α,β) = −cos(α−β) and CHSH |S| = 2√2 are anchored QM; lattice mechanism uses joint substrate ripples + opposite pump phases — C5.',
      size: 22, italics: true, font: 'Georgia',
    })],
  }),
]));

children.push(H2('7.1  What Is Shared? — Singlet and the Sea'));

children.push(EQ_NUM('|Ψ⟩ = (1/√2)(|↑↓⟩ − |↓↑⟩)', '7.1'));

children.push(PR(
  'Two electrons from pair production share one cosmic ocean connection. Their inner photons are not two independent coins with a classical tag — they are one ',
  { text: 'joint state', bold: true },
  ' |Ψ⟩. Always opposite: if A is up, B is down. Total spin zero. Each spring is a polarizer; but the polarizers read the ',
  { text: 'same', italics: true },
  ' underlying orientation geometry, not separate hidden dice.'
));

children.push(EQ_NUM('A: (1−A_k, +B_k)    B: (1−A_k, −B_k)', '7.2–7.3'));

children.push(PR(
  'Mirror lattice coordinates (Ch 2) preview this: shared x_k, opposite ±B_k. Full lattice closure adds joint substrate ripples and fill φ_{AB} on the connective path (P11-3) — ',
  { text: 'MODEL', italics: true },
  ', not only static mirror pairs.'
));

children.push(H2('7.2  Polarizer Mechanics on the Singlet'));

children.push(PR(
  'Alice sets polarizer axis α. Spring A aligns; inner photon A passes or blocks with Malus probabilities. When A passes along |α⟩, B is forced into |α+π⟩ — opposite on the circle. Bob\'s polarizer at β then sees angle (α+π)−β between B\'s state and his axis. Sequential polarizer logic on a ',
  { text: 'shared', bold: true },
  ' state produces the quantum correlations — not a signal sent between them.'
));

children.push(H2('7.3  Joint Probabilities and E(α,β) = −cos(α − β)'));

children.push(EQ_NUM('P(A=+, B=+) = (1/4)(1 − cos(α−β))', '7.4'));
children.push(EQ_NUM('P(A=+, B=−) = (1/4)(1 + cos(α−β))', '7.5'));
children.push(EQ_NUM('P(A=−, B=+) = (1/4)(1 + cos(α−β))', '7.6'));
children.push(EQ_NUM('P(A=−, B=−) = (1/4)(1 − cos(α−β))', '7.7'));
children.push(EQ_NUM('E(α,β) = ⟨AB⟩ = −cos(α − β)', '7.9'));

children.push(PR(
  'Special cases: α = β → E = −1 (perfect anti-correlation). α−β = 90° → E = 0. α−β = 180° → E = +1. These match decades of Bell tests. The correlation function is ',
  { text: 'E = −cos(α−β)', bold: true },
  ', ',
  { text: 'not', bold: true },
  ' cos²(α−β) — Malus cos²(θ/2) enters at each ',
  { text: 'single-particle', italics: true },
  ' polarizer step; the joint correlation is cosine. Kernel is ',
  { text: 'ANCHORED', bold: true },
  '. Generalization: E = −φ_{AB} cos(α−β) with φ_{AB} = 1 for ideal fill (C5, C7). Reject sgn(cos θ) half-plane sketches as proof.'
));

children.push(H2('7.4  CHSH Violation'));

children.push(EQ_NUM('|S| ≤ 2  [classical]', '7.10'));
children.push(EQ_NUM('|S| = 2√2  [quantum / lattice target]', '7.12'));

children.push(PR(
  'Optimal angles a=0°, a′=90°, b=45°, b′=135° give S = −2√2, |S| ≈ 2.828 — factor √2 above the classical bound. Bell assumed classical hidden variables fixed at creation. Here the "hidden" orientation is the inner photon in ',
  { text: 'superposition', bold: true },
  ' until a polarizer aligns it — quantum variables, not classical tags. Compatible with Aspect, Hensen, and later loophole-free tests.'
));

children.push(H2('7.5  Dynamic Formation and Break'));

children.push(EQ_NUM('dC/dt = Γ_form(1−C) − Γ_break C', '7.13'));
children.push(EQ_NUM('Γ_break ≈ Φ_env σ_obs + Γ_other', '7.14'));

children.push(PR(
  'Photons cause continuous observation → entanglement is always forming and breaking (P6-2, P6-3). In deep gravity wells the shared dilation factor can lengthen τ_E before Γ_break wins — **MODEL** (Ch 19); laboratory pressure still dominates most Bell tests.'
));

children.push(H2('7.6  No-Signaling'));

children.push(EQ_NUM('P(A=+) = 1/2  independent of β', '7.15'));

children.push(PR(
  'Alice always sees P(A=+) = 1/2 regardless of Bob\'s angle β. She can choose her axis α but not her outcome. Bob likewise sees 50/50 marginals. The ocean carries correlation instantly, but ',
  { text: 'information', bold: true },
  ' about the correlation only appears when they compare notes at speed c. Correlation without controllable message — microcausality (Ch 1).'
));

children.push(H2('7.7  Shared Tick on the π Lattice'));

children.push(PR(
  'Entangled electrons are not only correlated in spin — they share one ',
  { text: 'address tick', bold: true },
  ' on the π tree (Chapter 2). At creation both inner photons start at the same anchor; at each bisection step they take opposite ±B branches at the same depth k:'
));

children.push(EQ_NUM('A: (1−A_k, +B_k)     B: (1−A_k, −B_k)     same k', '7.16'));

children.push(PR(
  'One complete lattice step is one pump tick (T_bounce). They occupy mirrored coordinates at the ',
  { text: 'same', italics: true },
  ' iteration depth — geometric entanglement, not a label pasted on afterward. **MODEL** (Ch 12); Bell kernel remains **ANCHORED**.'
));

children.push(H2('7.8  Breaking Entanglement — Tick Separation'));

children.push(PR(
  'Measurement or environment (Γ_break) breaks the shared trajectory. Each inner photon keeps a private bounce history; iteration depths k₁ and k₂ need not match. Each electron may lock a ',
  { text: 'new', italics: true },
  ' partner from the sea (Γ_form) — but that partner arrived from a different event with its own k_env. The new pair must reconcile phases; imperfect alignment is what we call decoherence time τ_E ~ 1/(Γ_form + Γ_break). Partners in the same deep gravity well share a common dilation factor (Ch 19) — **MODEL**: relative Δk drift may slow before Γ_break wins.'
));

children.push(EQ_NUM('τ_{reconcile} ∼ |Δk| × t_{cell},   Δk = k_1 − k_{env}', '7.17'));

children.push(PR(
  'We measure the electron\'s present polarizer state, not its full bounce ledger. Unknown Δk between partners contributes to outcome spread — **MODEL** reading of quantum noise (not a return to classical hidden variables; depths can be in superposition until a polarizer samples a branch). Full counting proof — **OPEN** (O12-1). See Chapter 18.'
));

children.push(H2('7.9  Entanglement Breaking (State Picture)'));

children.push(PR(
  'One measurement collapses the singlet to a product state |α⟩_A ⊗ |α+π⟩_B — entanglement entropy drops from log 2 to 0. **ANCHORED** QM; lattice tick story above is the **MODEL** mechanism underneath.'
));

children.push(H2('7.10  Open Questions'));

children.push(OPEN_Q(
  'Iteration-depth uncertainty Δk·Δt ≥ ℏ/(2m_e c²)?',
  'Conjecture (OPEN, book 12.14 / Ch 18.4). Standard ΔxΔp Heisenberg remains the anchored baseline.'
));

children.push(PREDICTION(
  'Lower Φ_env (vacuum) lengthens entanglement lifetime τ_E ∼ 1/(Γ_form+Γ_break). Testable in pressure-controlled Bell setups.'
));

children.push(PAGEBREAK());

// =====================================================================
// CHAPTER 8: TUNNELING
// =====================================================================
children.push(H1('Chapter 13: Tunneling'));

children.push(BOX([
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { line: 360, after: 0 },
    children: [new TextRun({
      text: 'Tunneling is not a quantum miracle. The inner photon temporarily enters the sea as vapor, crosses the gap between atoms at c-limited scales, and recondenses on the far side. The anchored law T ~ e^{−2κL} comes from WKB; the lattice adds shredding, polarizer-catching, and recoherence χ.',
      size: 22, italics: true, font: 'Georgia',
    })],
  }),
]));

children.push(H2('8.1  The Vapor-State Picture'));

children.push(PR(
  'Classically, a particle with kinetic energy E_k cannot cross a barrier of height V > E_k. In the lattice picture the electron does not climb the wall — the ',
  { text: 'inner photon shreds', bold: true },
  ' out of the coin into the cosmic ocean, propagates as vapor through the empty space between barrier atoms, and ',
  { text: 'recondenses', bold: true },
  ' on the other side. The spring must partially release its hold; that costs borrowed energy ΔE = V − E_k from the sea, allowed only for time Δt ≤ ℏ/(2ΔE) by uncertainty. **MODEL** narrative (P7); WKB exponent below is **ANCHORED**.'
));

children.push(...DIAGRAM([
  '',
  '  COIN          VAPOR in sea           COIN',
  '  ╔═══╗    →   ~~~~ wake ~~~~   →    ╔═══╗',
  '  │ ● │        (between atoms)       │ ● │',
  '  ╚═══╝         empty gap            ╚═══╝',
  '  shred          L barrier            recapture',
  '',
], 'Figure 8.1: Shred → sea wake → recapture (soft regime, P7-2).'));

children.push(H2('8.2  Where e^{−2κL} Comes From'));

children.push(EQ_NUM('ΔE = V − E_k', '8.1'));
children.push(EQ_NUM('ΔE · Δt ≥ ℏ/2   →   Δt ≤ ℏ/(2ΔE)', '8.2'));
children.push(EQ_NUM('κ = √(2m ΔE)/ℏ = √(2m(V−E))/ℏ', '8.3'));
children.push(EQ_NUM('L_max = ℏ/(2√(2mΔE)) = 1/(2κ)', '8.4'));
children.push(EQ_NUM('T_{WKB} ∝ e^{−2κL}', '8.5'));

children.push(PR(
  'κ is inverse shred-range: how fast vapor amplitude decays per unit length. Small ΔE → small κ → long L_max → easier tunneling. Large m → large κ → exponentially harder tunneling. Steps (8.1)–(8.5): standard WKB — ',
  { text: 'ANCHORED', italics: true },
  '. Physical shred story is ',
  { text: 'MODEL', italics: true },
  '.'
));

children.push(H2('8.3  Polarizers Along the Barrier'));

children.push(PR(
  'Each atom in the barrier has electron springs — the same polarizers from Chapter 11. Vapor passing a layer risks being '
  { text: 'caught', bold: true },
  ' by a barrier polarizer; if caught, the coin recondenses inside the barrier and tunneling fails. For N atomic layers of spacing a, pass probability per layer p_pass ≈ e^{−nσ_p a} with vapor cross-section σ_p ~ πℏ²/(2mΔE). Over thickness L = Na this stacks to e^{−2κL} when coefficients match — the exponential is ',
  { text: 'cumulative polarizer encounters', bold: true },
  ', not a separate law. **MODEL** (O7-1); coefficient match to WKB is **PARTIAL**.'
));

children.push(H2('8.4  Shredding Fraction and Effective Mass'));

children.push(EQ_NUM('ξ_{shred} = |E_{bar}|/(|E_{bar}| + E_{ref})', '8.6'));
children.push(EQ_NUM('m_{eff} = m(1 + λ_m ξ_{shred})', '8.7'));
children.push(EQ_NUM('κ ∝ √m_{eff}  →  heavier → harder shred', '8.8'));

children.push(PR(
  'Protons and alphas are tightly fused coins — springs locked, inner photons deeply held. For the same barrier, κ_p ≈ √1836 · κ_e ≈ 43 κ_e, so T_e/T_p ~ e^{83 κ_e L} can exceed 10³⁶ for κ_e L ~ 1. That is why the Sun burns slowly: pp fusion needs proton tunneling through the Coulomb barrier at P ~ 10^{−25} — **ANCHORED** rate physics with **MODEL** shred interpretation.'
));

children.push(H2('8.5  Recoherence Inside the Barrier'));

children.push(EQ_NUM('χ̇ = Γ_{rec}(1−χ) − Γ_{sh} χ', '8.9'));
children.push(EQ_NUM('χ_{ss} = Γ_{rec}/(Γ_{rec} + Γ_{sh})', '8.10'));
children.push(EQ_NUM('T_{eff} = T_{WKB} · χ_{ss}', '8.11'));

children.push(PR(
  'χ ∈ [0,1] tracks compactness: χ = 1 fully recondensed coin, χ = 0 shredded vapor. Inside the barrier, shredding and recoherence compete. Soft barriers (P7-2: Π_pin ≪ 1) allow partial shred and recapture; hard compression (Π_pin → 1) is escape/collapse — not ordinary tunneling. T_eff in (8.11) is in code as t_eff_soft() — **PARTIAL** (Sec 7.3.1).'
));

children.push(H2('8.6  Tunneling Time — Fast but Not Superluminal'));

children.push(EQ_NUM('τ_{traverse} ~ L/c', '8.12'));

children.push(PR(
  'Vapor disturbance propagates at sea speed c. For L = 1 nm, τ ~ 3.3 attoseconds — below most clocks, so tunneling ',
  { text: 'seems', italics: true },
  ' instantaneous. Hartman-type dwell-time saturation: traversal time scales ~ L/c while ',
  { text: 'amplitude', bold: true },
  ' falls as e^{−2κL} — few particles succeed, those that do traverse quickly. **ANCHORED** phenomenology; no controllable FTL signaling (Sec 7.8).'
));

children.push(H2('8.7  Entangled Partners Share the Cost'));

children.push(PR(
  'When an entangled partner shares the ocean state (Ch 12), shredding on A perturbs B. Shared borrowing can lower effective κ per particle — qualitative explanation for Cooper-pair Josephson tunneling (**MODEL**). Do not quote a fixed enhancement factor without calibration; Josephson enhancement is **ANCHORED**, lattice cost-sharing is **PARTIAL** (O7-2).'
));

children.push(EQ_NUM('ΔE_{total} ≈ const  →  κ_{eff} ∝ √(ΔE/2)  [shared pair, MODEL]', '8.13'));

children.push(H2('8.8  Real Applications'));

children.push(BULLET('STM: I ∝ e^{−2κL}, κ ~ 10¹⁰ m⁻¹ — ~10× current change per 0.1 nm (**ANCHORED**).'));
children.push(BULLET('Alpha decay: Gamow P ~ e^{−2πη} → U-238 τ ~ 4.5×10⁹ y (**ANCHORED**).'));
children.push(BULLET('Solar pp fusion: P ~ 10^{−25} at T_core ~ 1.5×10⁷ K → L_☉ ~ 10²⁶ W (**ANCHORED**).'));
children.push(BULLET('Enzyme proton transfer: sub-nm barriers, κ_p large, rates ~ 10⁵/s with vibrational attempts (**E** phenomenology).'));
children.push(BULLET('Flash memory / tunnel diodes: same exponent, different κ, L (**ANCHORED**).'));

children.push(H2('8.9  Dark Matter and the Barrier'));

children.push(PR(
  'DM springs (no inner photon, Ch 17) form a connective mesh along some paths (P11-3, φ_path). Filled DM channel can ease shred transit: ξ_{shred} → ξ_{shred}(1 − η_{DM} φ_path) — **MODEL**; η_{DM} is a fit constant, not derived here.'
));

children.push(H2('8.10  Master Equations'));

children.push(BOX([
  new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: 'T_{WKB} = e^{−2κL}     κ = √(2m(V−E))/ℏ     T_{eff} = T_{WKB} χ_{ss}', font: 'Consolas', size: 18 })] }),
  new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: 'Δt ≤ ℏ/(2ΔE)     L_max = 1/(2κ)     τ ~ L/c', font: 'Consolas', size: 18 })] }),
  new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: 'ξ_{shred} = |E_{bar}|/(|E_{bar}|+E_{ref})     χ̇ = Γ_{rec}(1−χ)−Γ_{sh}χ', font: 'Consolas', size: 18 })] }),
  new Paragraph({ spacing: { after: 0 }, children: [new TextRun({ text: 'Gamow: P = e^{−2πη}     η = Z₁Z₂e²/(4πε₀ℏv)', font: 'Consolas', size: 18 })] }),
], { shaded: true }));

children.push(OPEN_Q(
  'Does polarizer-layer counting fully derive κ from first principles?',
  'PARTIAL (O7-1): WKB is anchored; polarizer cross-section match and Ĥ_x barrier coupling still open.'
));

children.push(PREDICTION(
  'Zeptosecond spectroscopy may resolve discrete shred/recapture steps and χ_{ss} recovery in ultrathin barriers.'
));

children.push(PAGEBREAK());

// =====================================================================
// CHAPTER 9: DOUBLE SLIT
// =====================================================================
children.push(H1('Chapter 14: Double Slit and Interference'));

children.push(BOX([
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { line: 360, after: 0 },
    children: [new TextRun({
      text: 'No paradox: the pattern lives in the cosmic ocean, not inside one electron. A signal coin pairs with a sea partner — one body per slit. Opposite-phase pumps make coherent wakes A_L + A_R; the detector clicks where sea disturbance is large. Not one electron through both slits.',
      size: 22, italics: true, font: 'Georgia',
    })],
  }),
]));

children.push(H2('9.1  Standard Fringe Geometry (Anchored)'));

children.push(PR(
  'Two slits separated by d, screen at distance L ≫ d. Path difference to height y on the screen (small-angle): Δr = yd/L = d sin θ. Phase difference Δφ = kΔr = 2πyd/(λL). Two equal amplitudes add:'
));

children.push(EQ_NUM('ψ = A_L + A_R = A(e^{iφ_L} + e^{iφ_R})', '9.1'));
children.push(EQ_NUM('I(y) = |ψ|² = 4|A|² cos²(Δφ/2) = 4|A|² cos²(πyd/(λL))', '9.2'));
children.push(EQ_NUM('Δy = λL/d  (bright fringe spacing)', '9.3'));
children.push(EQ_NUM('λ = h/p = h/√(2m_e E)  [de Broglie]', '9.4'));

children.push(PR(
  'This is classical two-path interference — ',
  { text: 'ANCHORED', bold: true },
  '. The lattice adds ',
  { text: 'what', italics: true },
  ' creates A_L, A_R (sea wakes from coin pumps), not a different fringe formula.'
));

children.push(H2('9.2  Partner Formation — One Slit Each'));

children.push(PR(
  'Ontology (section_08, mandate C5): approaching the barrier the signal electron locks a partner from the sea (Γ_form). Electron A → left slit; electron B → right slit. Inner photons share a joint state; pumps run opposite phase φ_R = φ_L + π. Each bounce drives a wake into the ocean. **MODEL** mechanism; fringe math above remains **ANCHORED**.'
));

children.push(...DIAGRAM([
  '',
  '     A (signal)              B (partner)',
  '         │                        │',
  '         ↓                        ↓',
  '     LEFT SLIT              RIGHT SLIT',
  '         \\    wake A_L    wake B_R /',
  '          \\        \\    /        /',
  '           \\    sea superposition',
  '            ▓ ░ ▓ ░ ▓  detector (clicks ∝ |wake|²)',
  '',
], 'Figure 9.1: Two entangled electrons, one slit each; wake interference in the sea.'));

children.push(H2('9.3  Wake Interference in the Sea'));

children.push(EQ_NUM('A_L(y) = (A₀/r_L) e^{ikr_L},   A_R(y) = (A₀/r_R) e^{ikr_R}', '9.5'));
children.push(EQ_NUM('I(y) = |A_L + A_R|² = I_L + I_R + 2√(I_L I_R) cos(Δφ)', '9.6'));

children.push(PR(
  'The cosmic ocean obeys wave propagation (∂²φ/∂t² = c²∇²φ in the continuum limit). Slits act as wake sources. ',
  { text: 'ψ is the wake amplitude', bold: true },
  ' at the screen; |ψ|² is local disturbance intensity — **MODEL** identification (O8-2). Detection: when local sea shift exceeds threshold, a spring polarizer fires and releases a packet (Ch 11) — the "click."'
));

children.push(H2('9.4  Which-Path Destroys Fringes'));

children.push(PR(
  'Coherent (entanglement intact): I = |A_L + A_R|² — cos(Δφ) term makes fringes. Incoherent (path marked): I = |A_L|² + |A_R|² — no cross term, two smooth humps, no fringes.'
));

children.push(PR(
  'Slit detector sends a probe → aligns A\'s spring polarizer (Ch 11) → A\'s inner photon definite → singlet breaks to product state → B\'s wake '
  { text: 'decouples', bold: true },
  ' from A\'s phase lock. Γ_break rises; visibility V → 0. Quantum eraser restores coherence when path information is erased — **ANCHORED** phenomenology, polarizer narrative **MODEL**.'
));

children.push(EQ_NUM('V = (I_{max} − I_{min})/(I_{max} + I_{min})', '9.7'));

children.push(H2('9.5  Gas Pressure and Visibility'));

children.push(EQ_NUM('Γ_{partner} = Σ_i n_i σ_i v_i f_{coin,i}', '9.8'));
children.push(EQ_NUM('V(P) ≈ V₀ e^{−Λ P}', '9.9'));

children.push(PR(
  'Each gas atom is a mini-observer: partial polarizer coupling to the entangled pair. More collisions → faster decoherence. Arndt–Zeilinger C₆₀ interferometry: visibility falls exponentially with pressure — ',
  { text: 'ANCHORED', bold: true },
  '. Fitted gas constants (order of magnitude): c_He ~ 81 bar⁻¹s⁻¹, c_Ar ~ 7.1 bar⁻¹s⁻¹ in their units. Lattice rate model **MODEL**; σ_{e,i}, f_{coin,i} micro-derivation **OPEN** (O8-1).'
));

children.push(H2('9.6  Single-Electron Buildup'));

children.push(EQ_NUM('P(y) = I(y)/I_{total}', '9.10'));
children.push(EQ_NUM('N(y) ≈ N · P(y),   SNR ~ √N', '9.11'));

children.push(PR(
  'Even "one electron at a time," each shot is one entangled pair contributing one detection drawn from P(y). After ~10² events (Tonomura-class) fringes emerge. In the limit N → ∞, hit density → I(y) — Born\'s rule as statistical accumulation over polarizer outcomes — **PARTIAL** (O1-1), not a separate postulate bolted on.'
));

children.push(PR(
  'Wave–particle duality dissolves: the ',
  { text: 'particle', bold: true },
  ' is the coin structure at the detector; the ',
  { text: 'wave', bold: true },
  ' is the sea wake it carried. Interference is wake interference; detection is polarizer packet release.'
));

children.push(H2('9.7  ³He vs ⁴He — Discriminating Test'));

children.push(PR(
  'Standard EM: ³He and ⁴He should differ only slightly (mass → velocity). Lattice: partner quality depends on σ_{e,i} and f_{coin,i} — how easily a gas atom donates or disrupts a coin partner. ³He (unpaired nuclear structure) vs ⁴He (closed shell) → different f_coin. **MODEL** (O8).'
));

children.push(EQ_NUM('Γ_{partner}^{³He}/Γ_{partner}^{⁴He} ≈ (σ_{e,3} f_{3})/(σ_{e,4} f_{4})', '9.12'));
children.push(EQ_NUM('Λ_{³He}/Λ_{⁴He} ≈ 1.075  [E-check: f₃=0.405, f₄=0.5, n=80]', '9.13'));

children.push(PR(
  'Calibrated target ~7.5% faster ³He decoherence than ⁴He at matched pressure — not the ~15% velocity-only shift of naive SM. Mass ratio m₄/m₃ ≈ 1.33 alone would overshoot; structure factors f₃/f₄ ≈ 0.81 cancel to ~7.5% (Ch 18 Pattern D). Deriving f_coin from isotope microphysics — **OPEN** (O8).'
));

children.push(PREDICTION(
  'Matched-pressure interferometry (C₆₀ or electron analog): fit V(P) = V₀ e^{−ΛP} and test H₀: Λ_{³He} = Λ_{⁴He}. Lattice predicts rejection at ~5–10% level; SM often predicts smaller split.'
));

children.push(H2('9.8  Master Equations'));

children.push(BOX([
  new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: 'I(y) = 4|A|² cos²(πyd/(λL))     Δy = λL/d     λ = h/p', font: 'Consolas', size: 18 })] }),
  new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: '|W|²_coherent = |A_L+A_R|²     |W|²_incoherent = |A_L|²+|A_R|²', font: 'Consolas', size: 18 })] }),
  new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: 'V(P) = V₀ e^{−ΛP}     Γ_partner = Σ n_i σ_i v_i f_{coin,i}', font: 'Consolas', size: 18 })] }),
  new Paragraph({ spacing: { after: 0 }, children: [new TextRun({ text: 'Λ_{³He}/Λ_{⁴He} ≈ 1.075 (E)     P(y) = I(y)/I_total', font: 'Consolas', size: 18 })] }),
], { shaded: true }));

children.push(H2('9.9  Open Questions'));

children.push(OPEN_Q(
  'Derive σ_{e,i} and f_{coin,i} from first principles?',
  'OPEN (O8-1). E-check lands at 1.075; microscopic isotope coupling not closed.'
));

children.push(OPEN_Q(
  'Maximum mass for interference?',
  'Practical limit from Γ_break and thermal emission; no proven fundamental lattice ceiling.'
));

children.push(PAGEBREAK());

(async () => {
  const doc = buildDoc('IV', 'Part IV — Quantum Mechanics (Ch 11–14)', children);
  await saveDoc(doc, 'book/output/04_QM_Measurement_Entanglement_Tunneling_DoubleSlit.docx');
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
