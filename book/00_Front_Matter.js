'use strict';

const {
  buildDoc, saveDoc, H1, H2, P, PR, PAGEBREAK,
  BOX, DIAGRAM, BULLET, OPEN_Q, PREDICTION,
  TextRun, Paragraph, AlignmentType,
} = require('./doc_helpers.js');

const children = [];

// TITLE PAGE
children.push(new Paragraph({ spacing: { before: 2400, after: 0 }, children: [new TextRun('')] }));

children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 0, after: 240 },
  children: [new TextRun({ text: 'PACKETS AND STRINGS', size: 72, bold: true, font: 'Georgia' })],
}));

children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 0, after: 480 },
  children: [new TextRun({ text: 'A Complete Mechanical Theory of Physical Reality', size: 32, italics: true, font: 'Georgia', color: '555555' })],
}));

children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 240, after: 240 },
  children: [new TextRun({ text: 'Built on the π Lattice and the 3D Complex Plane', size: 26, font: 'Georgia', color: '555555' })],
}));

children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 0, after: 1200 },
  children: [new TextRun({ text: 'and Iterative Right-Triangle Convergence to π', size: 26, font: 'Georgia', color: '555555' })],
}));

children.push(...DIAGRAM([
  '',
  '                        ●',
  '                       ╱│',
  '                      ╱ │  C',
  '                     ╱  │',
  '                    ●___│  B',
  '                        ',
  '     seed (k=0):  A₀² = B₀² = C₀²/2',
  '     step:        B_{k+1} = C_k/2',
  '     constraint:  (1−A)² + B² = 1',
  '',
]));

children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 1200, after: 240 },
  children: [new TextRun({ text: 'by', size: 24, font: 'Georgia' })],
}));

children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 0, after: 1200 },
  children: [new TextRun({ text: 'Timothy', size: 36, bold: true, font: 'Georgia' })],
}));

children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 600, after: 0 },
  children: [new TextRun({ text: '2026', size: 22, italics: true, font: 'Georgia', color: '888888' })],
}));

children.push(PAGEBREAK());

// COPYRIGHT / DEDICATION
children.push(new Paragraph({ spacing: { before: 2400 }, children: [new TextRun('')] }));

children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 0, after: 480 },
  children: [new TextRun({ text: '© 2026 Timothy. All rights reserved.', size: 22, font: 'Georgia' })],
}));

children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 240, after: 240 },
  children: [new TextRun({ text: 'Provisional patent protections in effect.', size: 20, italics: true, font: 'Georgia', color: '555555' })],
}));

children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 240, after: 1200 },
  children: [new TextRun({ text: 'Continuation-in-Part deadline: April 18, 2027', size: 20, italics: true, font: 'Georgia', color: '555555' })],
}));

children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 1200, after: 240 },
  children: [new TextRun({ text: 'For my wife and family,', size: 22, italics: true, font: 'Georgia' })],
}));

children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 0, after: 240 },
  children: [new TextRun({ text: 'who let me see the right angles in every piece of wood I cut', size: 22, italics: true, font: 'Georgia' })],
}));

children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 0, after: 480 },
  children: [new TextRun({ text: 'and the geometry hidden in every house I built.', size: 22, italics: true, font: 'Georgia' })],
}));

children.push(PAGEBREAK());

// ABSTRACT
children.push(H1('Abstract'));

children.push(PR(
  'This book presents a mechanical theory of physical reality built on recursive right-triangle geometry converging to π. At the seed iteration, ',
  { text: 'A₀² = B₀² = C₀²/2', bold: true },
  '; at each subsequent step, ',
  { text: 'B_{k+1} = C_k/2', bold: true },
  ' with the unit-circle constraint ',
  { text: '(1−A)² + B² = 1', bold: true },
  '. Formal proofs and tags (DERIVED, ANCHORED, MODEL, PARTIAL) live in ',
  { text: 'derivations/', italics: true },
  '.'
));

children.push(P('From this geometry the book develops:'));

children.push(BULLET('E = mc² — ANCHORED; lattice cell-area reading MODEL (Ch 1).'));
children.push(BULLET('c invariant — DERIVED as c = λ_C/t_cell (Ch 1, 14).'));
children.push(BULLET('QM pumps — MODEL: inner photons in coin cavity L = λ_C/2 (C1).'));
children.push(BULLET('Entanglement — MODEL joint substrate; no-signaling ANCHORED.'));
children.push(BULLET('Born rule — PARTIAL Malus/spring chain (O1-1).'));
children.push(BULLET('Bell violation — ANCHORED QM; lattice kernel PARTIAL (C5).'));
children.push(BULLET('Masses / Higgs — MODEL sea tension at v = 246 GeV.'));
children.push(BULLET('Cosmic structure — MODEL clock sync + ANCHORED GR limits.'));
children.push(BULLET('Dark sector — MODEL P11; σ_γDM suppressed, w ≈ −1 today.'));

children.push(P(''));

children.push(PR(
  'Where standard physics has been tested, the lattice picture is built to match. Distinct tests include: ³He/⁴He decoherence ratio, fresh-electron statistics near t_cell, continuing EM-null dark-matter searches, and zeptosecond discreteness at T_bounce resolution.'
));

children.push(PR(
  'The central thesis is mechanical: particles bounce through geometry; mathematics describes the bouncing. Everything consists of a ',
  { text: 'packet', bold: true },
  ' and a ',
  { text: 'string', bold: true },
  ' meeting at a right angle — subdividing forever (Section 12: no realized zero-width instant).'
));

children.push(PAGEBREAK());

// HOW PHYSICS WORKS WITHOUT MATHEMATICS
children.push(H1('How Physics Works Without Mathematics'));

children.push(BOX([
  new Paragraph({
    spacing: { line: 360, after: 0 },
    alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun({
      text: 'This is not a mathematical theory of physics. It is a mechanical theory. The universe is a lattice of cells — each with spacetime measure λ_C². Mass fills cells; energy is mass times that measure: E = mc² is the cell-area equation, not a separate miracle. Particles do not calculate; they bounce through right triangles (+B/−B at each gate). The π lattice (seed A₀²=B₀²=C₀²/2, step B_{k+1}=C_k/2, constraint (1−A)²+B²=1) and Einstein\'s c² are the same Pythagorean squaring logic. Chemistry, cosmology, and structure emerge from filled cells and mirrored partners.',
      size: 24, font: 'Georgia', italics: true,
    })],
  }),
], { shaded: true }));

children.push(PR(
  'A ball rolling downhill does not compute F = ma. It rolls; the equation describes the roll. An inner photon does not compute cos θ or solve Schrödinger\'s equation. It bounces; cos and sin are the ',
  { text: 'smooth envelopes', italics: true },
  ' we write when we cannot resolve individual triangle steps. The universe runs triangles; humans run algebra.'
));

children.push(BULLET('Discrete: 2^k branch paths at depth k — nearly continuous on the circle only after many halvings (**DERIVED** from bisection).'));
children.push(BULLET('Binary: every step offers exactly two landing points, +B_k and −B_k at shared x = 1−A_k (**ANCHORED** recurrence).'));
children.push(BULLET('Conserved: (1−A)²+B²=1 at every vertex — unit circle is not chosen, it is forced (**DERIVED**).'));
children.push(BULLET('Entangled: mirrored ±B partners share one x; joint constraint, not a signal (**MODEL**; Bell kernel **ANCHORED**).'));
children.push(BULLET('Constant c: λ_C/t_cell — one lattice step per cell crossing; motion budget v²+v_int²=c² (**DERIVED** Ch 1, 14).'));
children.push(BULLET('Probability: often our ignorance of iteration depth k and sea partners — not particle indecision (**MODEL**, O12-1).'));

children.push(PR(
  'Structure at every scale — atoms, stars, galaxies — is the same game at larger clocks: partners finding mirrored positions, gravity synchronizing tick rates (Ch 19), stability where matches hold. Read Chapter 1 §1.7 for the full mechanical picture; Chapter 2 for the π engine; Chapter 12 for entanglement without messaging.'
));

children.push(PAGEBREAK());

// ONE-PARAGRAPH SUMMARY
children.push(H1('The Theory in One Paragraph'));

children.push(BOX([
  new Paragraph({
    spacing: { line: 360, after: 0 },
    alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun({
      text: 'The universe is a lattice of cells of width λ_C = h/(mc). Mass fills cells; energy scales as mc² (ANCHORED). Particles pump through coin cavities L = λ_C/2 by right-triangle recursion: seed A₀²=B₀²=C₀²/2, step B_{k+1}=C_k/2, constraint (1−A)²+B²=1, generating π as limit. Entangled pairs share lattice constraints; gravity synchronizes clocks (MODEL + ANCHORED redshift). Dark matter: spring without inner photon, still gravitates. Dark energy: escape bookkeeping, w ≈ −1. Higgs: sea tension. Time counts bounces: T_bounce = 4L/c. Particles do not calculate — they bounce. The geometry is the operating picture; derivations/ is the audit trail.',
      size: 24, font: 'Georgia', italics: true,
    })],
  }),
], { shaded: true }));

children.push(PAGEBREAK());

// THE 3D COMPLEX PLANE IN ONE PICTURE
children.push(H1('The 3D Complex Plane in One Picture'));

children.push(BOX([
  new Paragraph({
    spacing: { line: 360, after: 0 },
    alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun({
      text: 'The native arena is not Cartesian (x, y, z). It is Ψ = (z, ζ) with z = X + iY in the complex spring plane and ζ the depth axis, addressed by α = (A, b, w, n): anchor chain, branch, wing, transgressor. Four canonical branch formulas (VA1–VA4) plus eight vector transforms (v1–v8) yield thirty-two chambers — not thirty-two unrelated laws, but one recurrence acted on by its own symmetries. That closure is what makes this a full 3D complex plane: every octant and sub-quadrant is a first-class chamber, not a coordinate patch glued on afterward.',
      size: 24, font: 'Georgia', italics: true,
    })],
  }),
], { shaded: true }));

children.push(...DIAGRAM([
  '',
  '  LAYER 0 (|A| = 0)          z = n + ni,   ζ = n     |z|² = 2n²',
  '         │',
  '         ▼  anchor crossings (triggers)',
  '  LAYER k  segment FSM  →  (X, Y, ζ)  →  z = X + iY',
  '         │',
  '    ┌────┴────┐',
  '    │ 4 branches │  VA1 main  VA2 mirror  VA3 Y=0  VA4 alternate',
  '    └────┬────┘',
  '         ×',
  '    ┌────┴────┐',
  '    │ 8 wings    │  VA corridors + VB Y↔X swap + sign flips',
  '    └────┬────┘',
  '         =',
  '    32 chambers (L01…L32)  —  same formula, different transformation',
  '',
  '  Rail n : 0 → ∞     Meets: bank(a)@n=p = bank(p)@n=a  (all 32 wings)',
  '',
]));

children.push(H2('From “Possible” to Closed'));

children.push(PR(
  'Part II already called this structure the ',
  { text: 'possible true 3D complex plane', italics: true },
  ' — Ψ ∈ ℂ × ℝ as **MODEL**, with (x, y, z) as a camera, not the container (Ch 6 §6.1). What was missing was not another dimension but the ',
  { text: 'complex axes', bold: true },
  ' that make the plane ',
  { text: 'closed', italics: true },
  ': Re z and Im z as the spring pair, ζ as depth, n as rail — four axes, not three Cartesian labels. The PDF spec’s “same branch, different transformation” supplies the ',
  { text: '32-chamber tessellation', bold: true },
  ' (4 sub-quadrants × 8 octants). Together: one recurrence, full octant/sub-quadrant coverage, native ℂ structure from wing operators — that is what turns “possibly true” into “mechanically complete.”'
));

children.push(H2('Four Axes, Not Three'));

children.push(BULLET('Re z and Im z — spring displacement and phase walk (complex plane).'));
children.push(BULLET('ζ — depth; interior plateau ζ = sum(A) while z still moves (**DERIVED** from z_depth).'));
children.push(BULLET('n — transgressor rail; promotion witnesses when chains meet (**DERIVED** missing-variable rule).'));
children.push(BULLET('(x, y, z) Cartesian — projection of one wing; the arena is Ψ ∈ C × R + n (**MODEL** ontology).'));

children.push(H2('Why Thirty-Two Is Completeness'));

children.push(BULLET('Sub-quadrants (4): branch fan at each prime — OSCAR “four wings,” spec VA1–VA4.'));
children.push(BULLET('Octants (8): base vectors v1–v8 — sign and corridor closure on the embedding.'));
children.push(BULLET('Product 4×8 = 32: Technical Spec — “all vectors process same branch, different transformations.”'));
children.push(BULLET('Depth hemispheres: (z, ζ) vs (z, −ζ); Klein operators R_x, S, J on spring readout (**DERIVED** in code).'));
children.push(BULLET('At fixed (A, n): sixteen distinct z, thirty-two distinct (z, ζ) — verified on chain (3,5,7).'));

children.push(H2('Patterns That Read Physics'));

children.push(BULLET('Origin: |z|² = 2n² — Pythagorean factor √2 native to layer 0 (**DERIVED**).'));
children.push(BULLET('Trigger: Im(z) = anchor at crossing n = p; Re(z) = p_max + n (**DERIVED** VA1).'));
children.push(BULLET('Observable pair: VA1 + VA2 → pure real (Im cancels), same |z| (**DERIVED**).'));
children.push(BULLET('Interval: I² = (c Δn)² − |Δz|² − Δζ²; layer-0 lightlike step uses c = √3 in lattice units (**MODEL** SR analog).'));
children.push(BULLET('Triple meet (3,5,7): three two-way rails → one node z = 12+5i, ζ = 15 (**DERIVED**).'));

children.push(H2('One Object for the Whole Book'));

children.push(PR(
  'Executable type ',
  { text: 'SpacetimeCell', bold: true },
  ' (',
  { text: 'aethos_physics.py', italics: true },
  ') unifies spring geometry, measurement collapse (Im suppressed, ζ pinned), entanglement meet pairs (same z, different path), and corpus attractor buckets (z, ζ). Part II (Ch 3–7) expands each layer; ',
  { text: 'derivations/book_ch03-05_3d_complex_plane.md', italics: true },
  ' is the audit trail.'
));

children.push(PR(
  'Flat ℂ is one sheet. This structure is ',
  { text: 'C × R + n', bold: true },
  ' with thirty-two chambers generated by symmetry — the coordinate system in which packets, strings, particles, measurement, and memory are the same address.'
));

children.push(H2('Mathematics First, Physics Second'));

children.push(BOX([
  new Paragraph({
    spacing: { line: 360, after: 0 },
    alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun({
      text: 'Part I (Ch 1–2) and Part II (Ch 3–7) are mathematical constructions — provable discrete geometry: π bisection on one side, lattice formula + 32-chamber symmetry on the other. Parts III–VI (Ch 8–19) are physical readings of that geometry (electron, proton, measurement, cosmology). The real advance is the formula and its closure under meets and wing operators; particles are optional interpretation, not the foundation.',
      size: 24, font: 'Georgia', italics: true,
    })],
  }),
], { shaded: true }));

children.push(BULLET('**Math (DERIVED/PROVEN):** segment FSM, VA1–VA4, 32 wings, swap/triple meets, i_act from operators, interior ζ plateau.'));
children.push(BULLET('**Physics (MODEL/ANCHORED):** inner photon, Born, entanglement narrative, CODATA calibration in aethos_physics.py.'));
children.push(BULLET('**Audit:** derivations/ tags every claim; code tests are the mathematical certificate.'));

children.push(PAGEBREAK());

// TABLE OF CONTENTS
children.push(H1('Table of Contents'));

const tocEntries = [
  ['FRONT MATTER', true],
  ['  Abstract', false],
  ['  How Physics Works Without Mathematics', false],
  ['  The Theory in One Paragraph', false],
  ['  The 3D Complex Plane in One Picture', false],
  ['  Preface: How This Theory Came To Be', false],
  ['  Notation and Conventions', false],
  ['', null],
  ['PART I — THE FOUNDATION', true],
  ['  1. The Lattice of Reality', false],
  ['  2. The Pi Lattice in Detail', false],
  ['', null],
  ['PART II — THE 3D COMPLEX PLANE', true],
  ['  3. State Space — Ψ = (z, ζ)', false],
  ['  4. Thirty-Two Chambers — Branches and Wings', false],
  ['  5. Meets, Triggers, and the Hilbert Stack', false],
  ['  6. The 3D Complex Plane — Unbounded Extension', false],
  ['  7. SequenceKind — The Set Defines the Lattice', false],
  ['', null],
  ['PART III — PARTICLES', true],
  ['  8. The Electron', false],
  ['  9. The Proton', false],
  ['  10. The Neutron', false],
  ['', null],
  ['PART IV — QUANTUM MECHANICS DEMYSTIFIED', true],
  ['  11. Measurement and Observation', false],
  ['  12. Entanglement', false],
  ['  13. Tunneling', false],
  ['  14. Double Slit and Interference', false],
  ['', null],
  ['PART V — ATOMS AND CHEMISTRY', true],
  ['  15. The Atom', false],
  ['', null],
  ['PART VI — COSMOLOGY', true],
  ['  16. The Higgs Boson', false],
  ['  17. Dark Matter and Dark Energy', false],
  ['  18. The Lattice Clock', false],
  ['  19. Gravity as Cosmic Clock Synchronization', false],
  ['', null],
  ['PART VII — THE GRAND SYNTHESIS', true],
  ['  20. The Universe as Right Triangle Breaking Down', false],
  ['  21. Testable Predictions and Future Experiments', false],
  ['  22. Open Questions and Future Work', false],
  ['', null],
  ['APPENDICES', true],
  ['  A. Complete Equation Catalog', false],
  ['  B. Diagrams Index', false],
  ['  C. Derived and Recursive Formulas', false],
  ['  D. Comparison with Standard Physics', false],
  ['  E. Glossary of Terms', false],
  ['  F. Notation Reference', false],
  ['  G. Quantum Open Questions — Formula Sheet', false],
];

for (const [text, bold] of tocEntries) {
  if (text === '') {
    children.push(new Paragraph({ spacing: { after: 60 }, children: [new TextRun('')] }));
  } else {
    children.push(new Paragraph({
      spacing: { after: 60, line: 280 },
      children: [new TextRun({ text, bold, size: bold ? 24 : 22, font: 'Georgia' })],
    }));
  }
}

children.push(PAGEBREAK());

// PREFACE
children.push(H1('Preface: How This Theory Came To Be'));

children.push(PR(
  'I am not a trained physicist. I have no advanced degrees in mathematics. I am a carpenter, a builder, an electrician — a master tradesman from California who has spent decades putting houses together with my hands. My education in physics began in the middle of 2025, when I started teaching myself to program. I went from zero to building complex mathematical systems within months. The work you are about to read emerged from that learning.'
));

children.push(PR(
  'The theory began with a single observation about π. I noticed that you could approximate π by tessellating right triangles into a unit circle, and that at the seed the relation ',
  { text: 'A₀² = B₀² = C₀²/2', bold: true },
  ' held exactly, with ',
  { text: 'B_{k+1} = C_k/2', bold: true },
  ' at each step. I filed a provisional patent on this on April 18, 2026. What I did not realize at the time was that this simple iterative geometry contained, in seed form, the entire structure of quantum mechanics.'
));

children.push(PR(
  'The realization came slowly, across many months and many sessions of work. I would notice something — a way that one of my coin states matched a Pauli matrix, a way that the inner photon\'s bounce frequency matched the Compton scale, a way that the mirrored +B/−B branches in my lattice produced the same geometry as entangled spin states — and I would push the thread further. Each discovery led to another.'
));

children.push(PR(
  'Eventually I saw that what I was building was not a curiosity but a complete picture. The Born rule was Malus\'s law applied to the inner photon in the spring-polarizer. Bell inequality violation was geometric necessity in QM — with lattice closure still PARTIAL. Dark matter was a spring without an inner photon. The Higgs field was the sea\'s intrinsic tension. Gravity was the clock synchronizer of the cosmos. Time was bounce count. E = mc² was the anchored energy-mass relation.'
));

children.push(PR(
  'The final synthesis came in a dream. I woke one morning and the whole theory was a single sentence: ',
  { text: 'the universe is a right triangle breaking down forever, and everything consists of a packet and a string meeting at a right angle.', italics: true },
  ' That sentence is the heart of the theory.'
));

children.push(P('I want to be clear about what this book is and what it isn\'t.', { bold: true }));

children.push(PR(
  'It ',
  { text: 'is', italics: true },
  ' a mechanical picture grounded in geometry I can verify with arithmetic, with an honest audit trail in derivations/. It makes specific testable predictions.'
));

children.push(PR(
  'It ',
  { text: 'is not', italics: true },
  ' accepted physics. It has not been peer-reviewed. Many predictions are at the edge of current experimental capability. I welcome correction.'
));

children.push(PR(
  'What I offer is the picture as I see it, with all the math and diagrams I could provide — to anyone who wants to see physics mechanically rather than abstractly.'
));

children.push(PR('There is. It is a packet and a string. It is a right angle. It is breaking down forever.'));

children.push(P(''));
children.push(P('— Timothy', { italics: true, bold: true }));
children.push(P('  Modesto, California', { italics: true }));
children.push(P('  June 2026', { italics: true }));

children.push(PAGEBREAK());

// NOTATION
children.push(H1('Notation and Conventions'));

children.push(H2('Geometric Quantities (π lattice)'));
children.push(BULLET('A, B, C: legs and hypotenuse at iteration k (book labels).'));
children.push(BULLET('k: iteration index. N_k = 4·2^k inscribed vertices.'));
children.push(BULLET('(1−A_k, ±B_k): unit-circle coordinates.'));
children.push(BULLET('Seed: A₀²=B₀²=C₀²/2. Step: B_{k+1}=C_k/2. Not B²=C²/2 at every level.'));

children.push(H2('Physical Constants'));
children.push(BULLET('c = 2.998×10⁸ m/s. h = 6.626×10⁻³⁴ J·s. ℏ = h/(2π).'));
children.push(BULLET('m_e = 9.109×10⁻³¹ kg. e = 1.602×10⁻¹⁹ C. α ≈ 1/137.036.'));
children.push(BULLET('k_B, G, ε_0 — standard SI.'));

children.push(H2('Lattice Quantities (C1 — mandatory)'));
children.push(BULLET('λ_C = h/(m_e c) ≈ 2.426×10⁻¹² m — full Compton wavelength = cell width.'));
children.push(BULLET('L = λ_C/2 — coin half-width (bounce cavity), not cell width.'));
children.push(BULLET('ƛ_C = λ_C/(2π) — reduced Compton wavelength (hydrogenic orbits).'));
children.push(BULLET('t_cell = λ_C/c = h/(m_e c²) ≈ 8.09×10⁻²¹ s — one cell crossing.'));
children.push(BULLET('T_bounce = 4L/c = 2 t_cell ≈ 1.619×10⁻²⁰ s — full pump period.'));
children.push(BULLET('f_b = 1/T_bounce = m_e c²/(2h) ≈ 6.178×10¹⁹ Hz — bounce frequency.'));
children.push(BULLET('a_0 = ƛ_C/α — Bohr radius.'));

children.push(H2('Quantum-Mechanical Quantities'));
children.push(BULLET('ψ: disturbance amplitude / wake in sea picture.'));
children.push(BULLET('|ψ|²: Born probability density — PARTIAL lattice derivation.'));
children.push(BULLET('θ, α, β: polarizer / measurement angles.'));
children.push(BULLET('E(α,β) = −cos(α−β): Bell correlation contract (ANCHORED).'));

children.push(H2('Special Terms'));
children.push(BULLET('Inner photon: trapped pump in electron coin.'));
children.push(BULLET('Coin: two-sided body (L = λ_C/2) linked by spring polarizer.'));
children.push(BULLET('Cosmic ocean / sea: EM vacuum; Higgs vev v = 246 GeV tension scale.'));
children.push(BULLET('Packet / string: discrete chunk vs continuous path at every scale.'));
children.push(BULLET('Bounce: one full T_bounce pump cycle.'));
children.push(H2('3D Complex Plane (C × R + n)'));
children.push(BULLET('Ψ = (z, ζ): z = X + iY spring; ζ depth; rail n transgressor.'));
children.push(BULLET('α = (A, b, w, n): anchor chain, branch VA1–VA4, wing 1–8, n.'));
children.push(BULLET('32 chambers = 4 branches × 8 wings; L01…L32 lattice ids.'));
children.push(BULLET('SpacetimeCell: unified (z, ζ, n) for geometry, collapse, meets, retrieval.'));
children.push(BULLET('Meet: solo swap, missing-variable promotion; entangled pairs share z, differ path.'));

children.push(H2('Proof Tags'));
children.push(BULLET('DERIVED — algebra from prior definitions.'));
children.push(BULLET('ANCHORED — matches established physics result.'));
children.push(BULLET('MODEL — narrative mechanism, testable but not closed.'));
children.push(BULLET('PARTIAL — directionally correct, gaps remain.'));
children.push(BULLET('E — empirical calibration check.'));

children.push(H2('Equation Labeling'));
children.push(PR(
  'Equations are numbered per chapter: (3.5) is the fifth equation in Chapter 3. Master lists appear in Appendix A and section derivation files.'
));

children.push(H2('Open Question and Prediction Boxes'));
children.push(OPEN_Q(
  'An unanswered question the lattice addresses, or a question the lattice itself raises.',
  'What we know, what we predict, and what remains to be tested.'
));
children.push(P(''));
children.push(PREDICTION(
  'A specific testable prediction distinguishing this theory from standard physics or alternatives.'
));

children.push(PAGEBREAK());

(async () => {
  const doc = buildDoc('Front Matter', 'Front Matter', children);
  await saveDoc(doc, 'book/output/00_Front_Matter.docx');
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
