'use strict';

const {
  buildDoc, saveDoc, H1, H2, H3, P, PR, EQ, EQ_NUM, CAPTION, PAGEBREAK,
  BOX, DIAGRAM, BULLET, NUMBERED, OPEN_Q,
  TextRun, Paragraph, AlignmentType,
} = require('./doc_helpers.js');

const children = [];

children.push(H1('Chapter 2: The π Lattice in Detail'));

children.push(BOX([
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { line: 360, after: 0 },
    children: [new TextRun({
      text: 'π lattice only — unit-circle bisection, not the 3D complex plane (Part II, Ch 3–7). Seed, recurrence, ±B partners, S_k→π, Zeno. The lattice formula on anchor chains generates Ψ=(z,ζ) elsewhere. Two constructions — ONTOLOGY.md. Code: pi/constructive_pi.py.',
      size: 22, italics: true, font: 'Georgia',
    })],
  }),
]));

children.push(H2('2.0  Book ↔ Code Notation'));

children.push(PR(
  'This chapter uses ',
  { text: 'book labels', bold: true },
  ' (B = halving leg, A = sagitta complement). The companion code uses A for the halving leg and B for the sagitta. Equations are consistent within each column of the following table:'
));

children.push(BOX([
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 60 },
    children: [new TextRun({ text: 'Book B_{k+1} = C_k/2  ↔  code A_{k+1} = C_k/2', font: 'Consolas', size: 20 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 0 },
    children: [new TextRun({ text: 'Book A_{k+1} = 1−√(1−B_{k+1}²)  ↔  code B_{k+1} = 1−√(1−A_{k+1}²)', font: 'Consolas', size: 20 })],
  }),
], { shaded: true }));

// ----- 2.1 -----
children.push(H2('2.1  The Iteration Rule'));

children.push(PR(
  'Begin with the unit circle. At iteration k = 0, start from the inscribed-square seed triangle in the first quadrant: legs A_0 = B_0 = 1, hypotenuse C_0 = √2, and N_0 = 4 vertices on the square. At each subsequent step, halve the previous chord:'
));

children.push(EQ_NUM('B_{k+1} = C_k / 2', '2.1'));
children.push(EQ_NUM('B_{k+1}² = C_k² / 4', '2.2'));

children.push(PR(
  'Equation (2.2) is the squared halving step. The seed identity A_0² = B_0² = C_0²/2 holds only for the initial 45° triangle (Chapter 1, eq. 1.1). After that, the step rule is (2.1), not B² = C²/2 at every level.'
));

// ----- 2.2 -----
children.push(H2('2.2  Pythagorean Constraint'));

children.push(PR(
  'Lattice points use coordinates (1 − A_k, ±B_k) on the unit circle:'
));

children.push(EQ_NUM('(1 − A_k)² + B_k² = 1', '2.3'));
children.push(EQ_NUM('A_k = 1 − √(1 − B_k²)', '2.4'));
children.push(EQ_NUM('C_k = √(A_k² + B_k²)', '2.5'));
children.push(EQ_NUM('x_k = 1 − A_k = √(1 − B_k²)', '2.5a'));
children.push(EQ_NUM('y_k = ± B_k', '2.5b'));

children.push(PR('Equations (2.1)–(2.5b) are complete: every finite-k point is computable from nested radicals in Q(√2).'));

// ----- 2.3 -----
children.push(H2('2.3  Coordinate Generation: Points Are Addresses'));

children.push(PR(
  'This is the heart of the π lattice — not a trick for computing π, but a recursive address engine. Each bisection step ',
  { text: 'generates new (x, y) points on the unit circle', bold: true },
  '. Those points become the vertices the next iteration\'s triangles use. The geometry builds itself: halve a chord, place a new point on the arc, mirror it across the axis, repeat around the circle. No sine tables. Only ±B and (1−A) from the recurrence.'
));

children.push(PR(
  'At every new point there are ',
  { text: 'two valid coordinates', bold: true },
  ' at the same x: +B_k and −B_k. They are mirror images — same arc distance from the anchor, opposite sides of the chord. This is what we mean by an entangled position on the lattice (',
  { text: 'MODEL', italics: true },
  ' preview for Chapter 12): one address, two geometric expressions. Measurement later picks a branch; the partner branch is fixed by symmetry.'
));

children.push(PR(
  'You can walk the recurrence ',
  { text: 'outward from the center', italics: true },
  ' or ',
  { text: 'inward from the arc', italics: true },
  ' — forward B_{k+1} = C_k/2, backward C_k = 2B_{k+1}. Same lattice points either way. Time-reversal at the geometric level (macroscopic arrows of time come from pumping statistics, Chapter 14).'
));

children.push(H3('Iteration 0 (seed)'));

children.push(P('A_0 = B_0 = 1, C_0 = √2 ≈ 1.4142. Anchor (1, 0). Inscribed square has N_0 = 4 corners on the circle.'));

children.push(H3('Iteration 1'));

children.push(EQ('B_1 = C_0/2 = √2/2 ≈ 0.7071'));
children.push(EQ('A_1 = 1 − √(1 − 1/2) = 1 − √2/2 ≈ 0.2929'));
children.push(EQ('C_1 = √(A_1² + B_1²) = √(2 − √2) ≈ 0.7654'));

children.push(P('New mirror pair on the circle:'));

children.push(EQ('P₁⁺ = (1 − A_1, +B_1) ≈ (0.7071, +0.7071)'));
children.push(EQ('P₁⁻ = (1 − A_1, −B_1) ≈ (0.7071, −0.7071)'));

children.push(PR(
  'Along the quadrant chain from (1, 0), three distinguished points appear (anchor plus ± pair). Full inscribed polygon at k = 1 has N_1 = 8 vertices.'
));

children.push(H3('Iteration 2'));

children.push(EQ('B_2 = C_1/2 = √(2 − √2)/2 ≈ 0.3827'));
children.push(EQ('A_2 = 1 − √(1 − B_2²) ≈ 0.0761'));
children.push(EQ('C_2 ≈ 0.3902'));
children.push(P('Quadrantal symmetry applies; N_2 = 16 inscribed vertices.'));

children.push(H3('Iteration 3'));

children.push(EQ('B_3 = C_2/2 ≈ 0.1951'));
children.push(EQ('A_3 ≈ 0.0192'));
children.push(EQ('C_3 ≈ 0.1960'));
children.push(P('N_3 = 32. Precision doubles in the π estimate at each bisection step.'));

// ----- 2.4 -----
children.push(H2('2.4  The Nested Radical Structure'));

children.push(PR('The halving legs are nested radicals built from 2:'));

children.push(EQ('B_1 = √2 / 2'));
children.push(EQ('B_2 = √(2 − √2) / 2'));
children.push(EQ('B_3 = √(2 − √(2 + √2)) / 2'));
children.push(EQ('⋮'));

children.push(PR(
  'At finite k, all values lie in ',
  { text: 'Q(√2)', bold: true },
  ' — rational arithmetic and square roots only. The limit π is transcendental, but every finite stage is algebraic and constructible.'
));

// ----- 2.5 -----
children.push(H2('2.5  The Bifurcation Tree'));

children.push(PR(
  'Each refinement splits a chord into two children at +B and −B. Binary depth k carries 2^k branch paths; the full inscribed polygon has N_k = 4·2^k vertices.'
));

children.push(...DIAGRAM([
  '',
  '                         (1, 0)              iteration 0',
  '                           │',
  '              ┌────────────┴────────────┐',
  '              │                         │',
  '         (·, +B_1)                 (·, −B_1)     iteration 1',
  '              │                         │',
  '      ┌───────┴───────┐         ┌───────┴───────┐',
  '      │               │         │               │',
  '  (·,+B_2)      (·,−B_2)   (·,+B_2)      (·,−B_2)  iteration 2',
  '      ⋮               ⋮         ⋮               ⋮',
  '',
  '  Leaf paths at depth k: 2^k        Inscribed vertices: N_k = 4·2^k',
  '  Mirror sibling pairs: 2^(k−1) per split level',
  '',
], 'Figure 2.1: Bifurcation tree and inscribed-vertex count.'));

children.push(BULLET('Binary branch paths at depth k: 2^k.'));
children.push(BULLET('Inscribed polygon vertices after k bisections: N_k = 4·2^k.'));
children.push(BULLET('A branch word of k symbols (+/−) identifies a leaf path.'));
children.push(BULLET('Depth is unbounded in mathematics; physical address interpretation uses cell scale λ_C (Chapter 1).'));

// ----- 2.6 -----
children.push(H2('2.6  Mirrored Positions and Entanglement'));

children.push(PR(
  'Mirrored partners share x_k = 1 − A_k but take y_k = +B_k and y_k = −B_k. Chapters 5 and 10 model entangled pairs as sharing one lattice address at mirrored coordinates — a ',
  { text: 'global geometric constraint', bold: true },
  ', not a message sent between distant locations.'
));

children.push(PR(
  { text: 'Microcausality:', bold: true },
  ' the unit-circle rule is a single constraint on joint state; observables at spacelike separation still commute. Bell correlations can violate classical bounds without any controllable faster-than-light signaling (architecture mandate C5; full treatment Chapter 12).'
));

children.push(OPEN_Q(
  'How can entangled particles correlate without faster-than-light communication?',
  'The lattice answer: no signal is sent. The pair shares one address on one constraint surface. Correlation is kinematic consistency of ±B branches, tested jointly — not a packet traveling between Alice and Bob.'
));

children.push(H2('2.6a  Halving, Doubling, and the Particle That Does Not Calculate'));

children.push(PR(
  'The recurrence does not ask permission to halve. B_{k+1} = C_k/2 is how a chord subdivides; mirrored partners at y = ±B_{k+1} share x = 1−A_{k+1}. Vertical separation 2B_{k+1} equals the previous chord C_k — the pair\'s lattice gap tracks the halving engine step by step. The inner photon does not evaluate the formula; it is already on the only curve the rule can draw.'
));

children.push(PR(
  'Branch paths double each depth (2^k leaves); inscribed vertices scale as N_k = 4·2^k. High k looks continuous on the circle — that smoothness is what human trigonometry names, not what the particle executes. At each gate it sees two neighbors: +B and −B. Entanglement locks one partner on each — seesaw geometry (Ch 1 §1.7d), not a phone call between distant coins.'
));

children.push(...DIAGRAM([
  '',
  '   depth k:   A at (1−A_k, +B_k)     B at (1−A_k, −B_k)',
  '              |y_A − y_B| = 2B_k',
  '',
  '   depth k+1: B_{k+1} = C_k/2  →  new mirrored pair, still on x²+y²=1',
  '',
  '   Particle: one bounce.  Physicist: write E = −cos(α−β).',
  '',
], 'Figure 2.4: Halving engine and mirror constraint.'));

// ----- 2.7 -----
children.push(H2('2.7  Reversibility'));

children.push(PR('Forward: B_{k+1} = C_k/2. Backward: C_k = 2 B_{k+1}. Same geometry read in opposite refinement direction.'));
children.push(PR(
  'The recurrence is time-reversal symmetric at the geometric level. Macroscopic arrows of time come from statistically irreversible forward pumping of trapped photons (Chapter 14), not from a built-in direction in the π rules.'
));

// ----- 2.8 -----
children.push(H2('2.8  Convergence to π'));

children.push(PR(
  'Let N_k be the inscribed vertex count and C_k the edge chord after k bisections. Perimeter P_k = N_k C_k approaches 2π:'
));

children.push(EQ_NUM('P_k = N_k · C_k', '2.6'));
children.push(EQ_NUM('2π = lim_{k→∞} N_k C_k', '2.7'));
children.push(EQ_NUM('π = lim_{k→∞} (N_k C_k)/2 = lim_{k→∞} 2^{k+1} C_k', '2.8'));

children.push(P('(Since N_k = 4·2^k, half the perimeter is 2^{k+1} C_k.)'));

children.push(BOX([
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 60 },
    children: [new TextRun({ text: 'k    N_k      C_k         N_k·C_k/2       |π − est|', bold: true, font: 'Consolas', size: 20 })],
  }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: '0      4    1.4142       2.8284         3.1×10⁻¹', font: 'Consolas', size: 20 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: '1      8    0.7654       3.0615         8.0×10⁻²', font: 'Consolas', size: 20 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: '2     16    0.3902       3.1214         2.0×10⁻²', font: 'Consolas', size: 20 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: '3     32    0.1960       3.1365         5.0×10⁻³', font: 'Consolas', size: 20 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: '5    128    0.0491       3.1413         3.1×10⁻⁴', font: 'Consolas', size: 20 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: '10  4096    0.001534     3.14159235     3.1×10⁻⁷', font: 'Consolas', size: 20 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [new TextRun({ text: '20  4,194,304  0.00000150  3.1415926536  2.9×10⁻¹³', font: 'Consolas', size: 20 })] }),
], { shaded: true }));

children.push(CAPTION('Table 2.1: Half-perimeter N_k C_k / 2 converging to π (from pi_recurrence).'));

children.push(PR(
  'Cumulative sliver area S_k from the same recurrence also converges to π. At finite k every estimate is algebraic; π is the smooth limit of discrete bisection.'
));

// ----- 2.9 -----
children.push(H2('2.9  Zeno\'s Paradox'));

children.push(PR(
  'Zeno: infinitely many positive half-steps should take infinite time. The lattice answers on two levels:'
));

children.push(NUMBERED('Mathematical: Σ 2^{−n} = 1 — infinitely many halvings sum to a finite span. The π recurrence is this sum made geometric.'));
children.push(NUMBERED('Physical (Chapter 16 / Sec 12): no finite process realizes a zero-width instant. Each finite k has C_k > 0. The Compton cell λ_C is the operational address floor for particles — not a halt to mathematical refinement.'));

children.push(PR(
  'Zeno is not a paradox in this framework: it is the statement that refinement converges while no terminal “rest instant” is ever reached.'
));

// ----- 2.10 -----
children.push(H2('2.10  Recursive Formulas'));

children.push(H3('2.10.1  Vertex coordinates'));

children.push(EQ_NUM('x_k = √(1 − B_k²)', '2.9'));
children.push(EQ_NUM('y_k = ± B_k', '2.10'));

children.push(H3('2.10.2  Half-angle interpretation (continuous limit)'));

children.push(PR(
  'In the continuum limit, θ_k halves each step and (B_k, 1 − A_k) approach (sin θ_k, cos θ_k). In the discrete lattice, sin and cos are ',
  { text: 'emergent', bold: true },
  ' — the iteration defines them; they do not drive it.'
));

children.push(EQ_NUM('θ_k = θ_{k−1}/2   [continuous limit]', '2.11'));

children.push(H3('2.10.3  Iteration depth and pump ticks'));

children.push(PR(
  'Do not equate iteration index k with accumulated rest energy. Depth k is address refinement. One electron pump cycle takes T_bounce = 2 t_cell (Chapter 1). Linking k to physical time requires a separate clock map — open until Chapters 3 and 13.'
));

children.push(H3('2.10.4  Area accumulation'));

children.push(EQ_NUM('S_{k+1} = S_k + N_k · B_{k+1} · A_{k+1}', '2.12'));
children.push(PR('with S_0 = 2 (inscribed unit square). Then S_k → π, parallel to perimeter convergence.'));

children.push(H3('2.10.5  Constructive mechanics vs circular functions'));

children.push(PR(
  'cos θ and sin θ ',
  { text: 'assume', bold: true },
  ' a circle already exists — they describe motion on a given curve. The π iteration ',
  { text: 'builds', bold: true },
  ' that curve from discrete right-triangle steps: only rationals and nested √2 radicals in `constructive_pi.py`, no trig library. Analogy: temperature describes heat; molecular motion explains it. cos(θ) describes circular position; vertex (1−A_k, ±B_k) explains it.'
));

children.push(PR(
  'At finite k the motion is ',
  { text: 'discrete', bold: true },
  ' — a bounce or address step, not an infinitesimal arc. At k → ∞ the inscribed polygon smooths and sin/cos appear as the continuous envelope. π is not postulated; it is the limit of perimeter N_k C_k. Time at the electron scale ticks at t_cell and T_bounce (Chapter 1), not Planck time — the lattice lives at Compton scale λ_C.'
));

children.push(PR(
  'Do ',
  { text: 'not', bold: true },
  ' replace every cos/sin in the book with iteration overnight. Standard formulas (Malus cos²(θ/2), Bell E = −cos(α−β), fringe cos²(πyd/λL)) remain ',
  { text: 'ANCHORED', italics: true },
  ' as limits. The lattice supplies the mechanism underneath — **MODEL / PARTIAL** until discrete counting proofs close (O1-1, Oπ-3).'
));

// ----- 2.11 -----
children.push(H2('2.11  Open Questions'));

children.push(OPEN_Q(
  'Is there a maximum physical refinement k_max?',
  'Order-of-magnitude estimate: k_max ≈ log₂(λ_C/ℓ_Planck) ≈ 80, comparing electron cell width to Planck length. Whether nature realizes all depths or selects an active anchor set (C6) remains model-dependent.'
));

children.push(OPEN_Q(
  'Why is one ± branch realized and not the other?',
  'Geometry permits both; measurement orientation and pump state select the branch (Chapters 6–7). Deepest “choice” layer: open.'
));

children.push(OPEN_Q(
  'Is the π lattice the simplest structure that yields tested physics?',
  'Conjecture: two rules (seed + unit circle + halving step) may be minimal. Formal proof: open.'
));

// ----- 2.12 -----
children.push(H2('2.12  Summary'));

children.push(PR(
  'The π lattice is seeded by the 1–1–√2 triangle (C_0 = √2), refined by B_{k+1} = C_k/2 with unit-circle constraint (1 − A)² + B² = 1. Coordinates are nested radicals in Q(√2). Binary ±B branching supplies entanglement geometry (Chapter 12). Perimeter N_k C_k and area S_k converge exponentially to 2π and π. The recurrence is reversible; macroscopic time arrows come from pump statistics, not from the bisection rule alone.'
));

children.push(PR('Chapter 3 applies this structure to the electron — coin, cavity L = λ_C/2, and trapped inner photon.'));

children.push(PAGEBREAK());

(async () => {
  const doc = buildDoc('2', 'Chapter 2 — The π Lattice in Detail', children);
  await saveDoc(doc, 'book/output/02_The_Pi_Lattice.docx');
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
