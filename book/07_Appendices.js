'use strict';

const {
  buildDoc, saveDoc, H1, H2, H3, P, PR, EQ_NUM, CAPTION, PAGEBREAK,
  BOX, BULLET,
  TextRun, Paragraph, AlignmentType,
} = require('./doc_helpers.js');

const children = [];

children.push(new Paragraph({ spacing: { before: 2400 }, children: [new TextRun('')] }));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 1200 },
  children: [new TextRun({ text: 'APPENDICES', size: 56, bold: true, font: 'Georgia' })],
}));
children.push(PAGEBREAK());

// APPENDIX A
children.push(H1('Appendix A: Complete Equation Catalog'));

children.push(PR('Curated master sheet. Full proofs: derivations/section_XX_derivations.md. Tags: D=derived, A=anchored, M=model, P=partial.'));

children.push(H2('A.1  Foundations (Ch 1–2)'));
children.push(BULLET('c = λ_C/t_cell  [D]'));
children.push(BULLET('E = mc²  [A]; cell-area reading [M]'));
children.push(BULLET('E = hf, p = h/λ, E = pc (photon)  [A]'));
children.push(BULLET('Seed: A₀²=B₀²=C₀²/2  [I]; step B_{k+1}=C_k/2  [D]'));
children.push(BULLET('(1−A)² + B² = 1  [D]'));
children.push(BULLET('π = lim 2^{k+1} C_k  [D] (constructive_pi.py)'));

children.push(H2('A.2  Electron / coin (Ch 3, C1)'));
children.push(BULLET('λ_C = h/(m_e c), L = λ_C/2  [G]'));
children.push(BULLET('t_cell = λ_C/c, T_bounce = 4L/c = 2 t_cell  [D]'));
children.push(BULLET('f_b = m_e c²/(2h) = 1/T_bounce  [D]'));
children.push(BULLET('H_coin: Δ, Ω, E_obs drive  [P]'));

children.push(H2('A.3  Proton / fusion (Ch 4, C2)'));
children.push(BULLET('K_f from geometry — not FIT to 1836  [G]'));
children.push(BULLET('R_pe = m_p/m_e — compare to 1836.152…  [E check]'));
children.push(BULLET('L_p = L_0(1 − α K_f)  [M]'));

children.push(H2('A.4  Neutron (Ch 5, C3)'));
children.push(BULLET('Pressure escape P → P_c primary  [M/G]'));
children.push(BULLET('τ_n — SM weak rate as comparison  [E]'));

children.push(H2('A.5  Measurement / entanglement (Ch 6–9)'));
children.push(BULLET('P ∝ cos²(θ/2) Malus / spring tension²  [P]'));
children.push(BULLET('E(α,β) = −cos(α−β)  [A]'));
children.push(BULLET('Λ_n = 2∫Γ_n dt  [P]'));
children.push(BULLET('Partner slit: signal + entangled partner, one slit each  [M]'));

children.push(H2('A.6  Atoms / chemistry (Ch 10)'));
children.push(BULLET('E_bond from ℏω_b, η_AB  [P]'));
children.push(BULLET('Hydrogenic orbitals — ANCHORED QM'));

children.push(H2('A.7  Cosmology (Ch 11–14)'));
children.push(BULLET('v_space² + v_time² = c²  [A/D]'));
children.push(BULLET('v_time = c√(1−2GM/rc²)  [A]'));
children.push(BULLET('σ_γDM ≈ σ_geom K_sup  [P]'));
children.push(BULLET('w ≈ −1 today; CPL drift allowed  [M]'));
children.push(BULLET('ρ_Λ ~ Π_vac u_S/c²  [P]'));

children.push(H2('A.8  Synthesis / time (Ch 15–17, Sec 12)'));
children.push(BULLET('Δt = N_bounce × T_bounce  [D]'));
children.push(BULLET('S ∼ k_B k ln 2  [M sketch]'));
children.push(BULLET('No realized w=0 instant — finite prime descent  [D]'));

children.push(PAGEBREAK());

// APPENDIX B
children.push(H1('Appendix B: Diagrams Index'));

children.push(PR('Key figures by chapter. Full diagrams appear in chapter .docx files.'));

const diagramIndex = [
  ['Ch 1', 'Lattice cell; E=mc² area; sea ontology'],
  ['Ch 2', 'π convergence; unit circle; halving tree'],
  ['Ch 3', 'Electron coin; inner photon bounce path'],
  ['Ch 4', 'Proton fusion; drain geometry'],
  ['Ch 5', 'Neutron layers; pressure escape'],
  ['Ch 6', 'Spring polarizer; measurement axis'],
  ['Ch 7', 'Entanglement mirror; Bell setup'],
  ['Ch 8', 'Tunneling barrier; wake shred'],
  ['Ch 9', 'Double slit + partner; decoherence'],
  ['Ch 10', 'Atomic orbitals; bond geometry'],
  ['Ch 11', 'Higgs / sea tension'],
  ['Ch 12', 'DM spring; DE escape channel'],
  ['Ch 13', 'Lattice clock; dilation'],
  ['Ch 14', 'Gravity well; CMB sync sketch'],
  ['Ch 15', 'Fractal right-angle descent (Fig 15.1)'],
  ['Ch 16', 'Prediction summary table'],
  ['Front', 'Title-page right triangle'],
];

for (const [ch, desc] of diagramIndex) {
  children.push(BULLET(`${ch}: ${desc}`));
}

children.push(PAGEBREAK());

// APPENDIX C
children.push(H1('Appendix C: Derived and Recursive Formulas'));

children.push(H2('C.1  π-lattice recurrence'));
children.push(EQ_NUM('B_{k+1} = C_k / 2', 'C.1'));
children.push(EQ_NUM('A_{k+1} = 1 - \\sqrt{1 - B_{k+1}^2}', 'C.2'));
children.push(EQ_NUM('C_{k+1} = \\sqrt{A_{k+1}^2 + B_{k+1}^2}', 'C.3'));
children.push(EQ_NUM('N_k = 4 \\cdot 2^k', 'C.4'));
children.push(EQ_NUM('\\pi = \\lim_{k\\to\\infty} 2^{k+1} C_k', 'C.5'));

children.push(PR('Code map (constructive_pi.py): book B ↔ code A (halving leg); verify before mixing labels.'));

children.push(H2('C.2  Motion budget'));
children.push(EQ_NUM('v_{space}^2 + v_{time}^2 = c^2', 'C.6'));
children.push(EQ_NUM('v_{time} = c/\\gamma = c\\sqrt{1 - v^2/c^2}', 'C.7'));
children.push(EQ_NUM('v_{flow} = \\sqrt{2GM/r}', 'C.8'));

children.push(H2('C.3  Pump / clock'));
children.push(EQ_NUM('T_{bounce} = 4L/c = 2 t_{cell}', 'C.9'));
children.push(EQ_NUM('f_b = 1/T_{bounce} = m_e c^2/(2h)', 'C.10'));
children.push(EQ_NUM('f_b = f_{b0}/\\gamma  (moving electron)', 'C.11'));

children.push(H2('C.4  Zeno width descent (Sec 12)'));
children.push(EQ_NUM('w_{k+1} = w_k / p_k', 'C.12'));
children.push(PR('Prime p_k (or general active-set radix) subdivides frame width; no finite step reaches w=0.'));

children.push(PAGEBREAK());

// APPENDIX D
children.push(H1('Appendix D: Comparison with Standard Physics'));

children.push(BOX([
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: 'Topic              Standard              Lattice              Match', bold: true, font: 'Consolas', size: 18 })] }),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: 'E=mc²              exact                 cell-area read       A + M', font: 'Consolas', size: 18 })] }),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: 'SR time dilation   gamma                 bounce slowdown      A', font: 'Consolas', size: 18 })] }),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: 'Born rule          |psi|²                Malus/spring         P', font: 'Consolas', size: 18 })] }),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: 'Bell               Tsirelson 2√2         -cos(α-β) kernel     A + P', font: 'Consolas', size: 18 })] }),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: 'Hydrogen spectrum  QM                    orbitals anchored    A', font: 'Consolas', size: 18 })] }),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: 'DM direct detect   WIMP/axion            σ_γDM suppressed     test', font: 'Consolas', size: 18 })] }),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: 'DE equation        w≈-1                  escape bookkeeping   P', font: 'Consolas', size: 18 })] }),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: '³He/⁴He Λ          mass only <1%         5-10% split          discriminator', font: 'Consolas', size: 18 })] }),
  new Paragraph({ spacing: { after: 0 }, children: [new TextRun({ text: 'GR field eqs       Einstein              budget partial       OPEN', font: 'Consolas', size: 18 })] }),
]));
children.push(CAPTION('Table D.1: Lattice vs standard physics. A=anchored, M=model, P=partial.'));

children.push(PR('When lattice and standard disagree in a tested regime, standard wins until experiment says otherwise.'));

children.push(PAGEBREAK());

// APPENDIX E
children.push(H1('Appendix E: Glossary of Terms'));

const glossary = [
  ['Active set', 'Countable anchor addresses on the recursive lattice (C6); not limited to primes.'],
  ['Bounce', 'One full electron pump period T_bounce.'],
  ['Coin', 'Electron geometric body; half-width L = λ_C/2.'],
  ['Compton cell', 'Lattice cell width λ_C = h/(mc).'],
  ['Cosmic ocean', 'Photon sea / EM vacuum with Higgs-scale tension.'],
  ['Dark energy', 'Escape / sea bookkeeping channel; w ≈ −1 today.'],
  ['Dark matter', 'Spring lattice structure without inner photon; gravitates.'],
  ['Decoherence', 'Loss of phase coherence via observation / environment.'],
  ['Inner photon', 'Trapped pump quantizing electron mass.'],
  ['Packet', 'Discrete energy/information chunk (short leg).'],
  ['Prime descent', 'Zeno frame subdivision w_{k+1}=w_k/p_k; no terminal instant.'],
  ['Spring polarizer', 'Coin link aligning to measurement compression.'],
  ['String', 'Continuous propagation path (long leg).'],
  ['Wake', 'Sea disturbance from tunneling / slit geometry.'],
];

for (const [term, def] of glossary) {
  children.push(H3(term));
  children.push(P(def));
}

children.push(PAGEBREAK());

// APPENDIX F
children.push(H1('Appendix F: Notation Reference'));

children.push(PR('Subset of derivations/symbol_registry.md. Full registry is authoritative for code and proofs.'));

children.push(H2('F.1  Mandated geometry (C1)'));
children.push(BOX([
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: 'λ_C   cell width = h/(m_e c)', font: 'Consolas', size: 20 })] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: 'L     coin half-width = λ_C/2', font: 'Consolas', size: 20 })] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: 't_cell   = λ_C/c', font: 'Consolas', size: 20 })] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: 'T_bounce = 4L/c = 2 t_cell', font: 'Consolas', size: 20 })] }),
  new Paragraph({ spacing: { after: 0 }, children: [new TextRun({ text: 'f_b      = 1/T_bounce = m_e c²/(2h)', font: 'Consolas', size: 20 })] }),
]));

children.push(H2('F.2  π-lattice (book labels)'));
children.push(BULLET('A_k — sagitta complement leg'));
children.push(BULLET('B_k — halving leg; B_{k+1}=C_k/2'));
children.push(BULLET('C_k — hypotenuse / chord'));
children.push(BULLET('N_k = 4·2^k — vertex count'));

children.push(H2('F.3  Measurement / entanglement'));
children.push(BULLET('Λ_n — integrated measurement strength'));
children.push(BULLET('Γ_form, Γ_break — formation / break rates'));
children.push(BULLET('φ_AB — DM ripple fill on entanglement string'));

children.push(H2('F.4  Dark sector'));
children.push(BULLET('ρ_DM, ρ_DE — matter / energy densities'));
children.push(BULLET('w — equation-of-state p/(ρc²)'));
children.push(BULLET('σ_γDM — EM cross-section (suppressed)'));

children.push(H2('F.5  Zeno / motion (Sec 12)'));
children.push(BULLET('w_n — frame width at depth n'));
children.push(BULLET('v_space, v_time — motion-budget components'));
children.push(BULLET('v_flow — gravitational flow speed'));

children.push(PR('Updates: sync with symbol_registry.md whenever section derivations change (cross_reference_matrix checklist).'));

children.push(PAGEBREAK());

// APPENDIX G
children.push(H1('Appendix G: Quantum Open Questions — Formula Sheet'));

children.push(PR('Companion to Chapter 18.2 and derivations/physics_questions_map.md Part B.'));

const qmSheet = [
  ['Measurement / collapse', 'Λ_n = 2∫Γ_n dt; Kraus M_{s,n}', 'PARTIAL'],
  ['Born rule origin', 'P_S ∝ T_S², T_S = k_S Re(ψ_S)', 'PARTIAL'],
  ['Coherence ODE', 'Ċ = Γ_form(1−C) − Γ_break C', 'PARTIAL'],
  ['EPR / Bell', 'E(α,β) = −cos(α−β); |E| ≤ 2√2', 'A + P'],
  ['No signaling', 'Retarded Green\'s function; T3', 'PARTIAL'],
  ['Double-slit interference', 'I = |A_L + A_R|²; φ_R = φ_L + π', 'A + M'],
  ['Partner decoherence', 'Γ_partner = Σ n_i σ_i v̄_i f_i', 'PARTIAL'],
  ['Visibility vs pressure', 'V(P) = V_0 e^{−ΛP}', 'PARTIAL'],
  ['Quantum Zeno', 'w_{k+1} = w_k/p_k; high Γ_obs', 'A + P'],
  ['Uncertainty', 'Δx Δp ≥ ℏ/2', 'ANCHORED; mech OPEN'],
  ['Tunneling WKB', 'T ~ e^{−2κL}; T_eff = T_WKB χ_ss', 'PARTIAL'],
  ['Charge sign', 'q = q_0 χ', 'PARTIAL'],
  ['Mass ratio', 'R_pe=(π²/8)×M_lat; n=80→1847', 'PARTIAL+E-check'],
  ['Neutrino map', 'β decay ↔ γ_obs escape', 'PARTIAL'],
  ['Fresh electron test', 'P(θ) discrete until N_bounce ≫ 1', 'TEST'],
  ['³He/⁴He', 'Λ ratio ~1.075 (f3=0.405,f4=0.5)', 'TEST+E-check'],
  ['Recoherence scale', 'τ_re^mol/τ_re^e ~ (M/m_e)^{1/2}…', 'PARTIAL'],
  ['Complex amplitudes', 'ψ = |ψ|e^{iφ}', 'GAP'],
  ['Pauli / shells', 'N_max(n) = 2n²', 'A count; FD OPEN'],
  ['Horizon limit', 'dτ/dt → 0 at r_s', 'ANCHORED'],
];

children.push(BOX([
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: 'Question area           Formula / relation        Status', bold: true, font: 'Consolas', size: 17 })] }),
  ...qmSheet.map(([area, formula, status], i) => new Paragraph({
    spacing: { after: i === qmSheet.length - 1 ? 0 : 50 },
    children: [new TextRun({ text: `${area.padEnd(22)} ${formula.padEnd(28)} ${status}`, font: 'Consolas', size: 16 })],
  })),
]));
children.push(CAPTION('Table G.1: Quantum physics open questions with backing lattice equations.'));

children.push(PAGEBREAK());

(async () => {
  const doc = buildDoc('Appendices', 'Appendices A–G', children);
  await saveDoc(doc, 'book/output/07_Appendices.docx');
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
