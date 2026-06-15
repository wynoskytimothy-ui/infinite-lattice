# Book Chapters 6–9 — Quantum Mechanics: Formal Derivations

Maps Part III to `section_05`–`08` derivations and mandates **C5**.

---

## Chapter 6 — Measurement

### 6.1 Core map

| Book claim | Repo status |
|------------|-------------|
| Observation = physical compression | **P5-1** |
| Kraus/projectors from `H_coin` | **PARTIAL** (Sec 5.3) |
| `P(up|θ) = cos²(θ/2)` | **E/ANCHORED** |
| Malus = Born mechanism | **MODEL / PARTIAL** (O1-1, O5-2) |
| Spring = active polarizer | **MODEL** |

### 6.2 Time scales

Polarizer reorientation: **`t_cell`** or **`T_bounce`** (Ch 3), not `1/(m_e c²/h)` alone.

### 6.3 Consecutive axes

Noncommuting sequential measurements: **ANCHORED** QM.

Malus chain (unpolarized → 45° → 90°): **25%** transmission — **PROVEN** optics.

**Forbidden:** wrong product `(cos²22.5°)²(sin²22.5°)²`.

### 6.4 Energy partition (6.5–6.6)

Packet release fractions — **MODEL** narrative; full material closure **OPEN**.

---

## Chapter 7 — Entanglement

### 7.1 Correlation kernel

\[
E(\alpha,\beta) = -\cos(\alpha-\beta)
\qquad
|S|_{\max} = 2\sqrt{2}
\]

**Status:** **E/ANCHORED** (QM); lattice derivation **PARTIAL**.

### 7.2 C5 mandate

| Allowed | Forbidden |
|---------|-----------|
| Joint mechanical ripple on shared substrate | `sgn(cos θ)` half-plane sketch (**REJECT** — fails numeric check) |

Mirror `(1−A, ±B)` pairs = **geometric preview** (Ch 2), not complete Bell kernel.

Full kernel contract: `E = −φ_{AB} cos(a−b)` with DM path fill **φ_{AB}** (P11-3) — **OPEN**.

### 7.3 Dynamic entanglement (P6)

\[
\frac{dC}{dt} = \Gamma_{form}(1-C) - \Gamma_{break} C
\qquad
\Gamma_{break} \approx \Phi_{env}\sigma_{obs} + \Gamma_{other}
\]

Photons continuously break locks — **MODEL**, testable.

### 7.4 No-signaling

\[
P(A|\alpha,\beta) = P(A|\alpha)
\]

**ANCHORED**; does not contradict P1a.

### 7.5 Δk × Δt relation (book 7.14)

**Status:** **OPEN** conjecture — not proven as fundamental uncertainty in repo.

---

## Chapter 8 — Tunneling

### 8.1 WKB anchor

\[
T \sim e^{-2\kappa L},\quad \kappa = \sqrt{2m(V_0-E)}/\hbar
\]

**E/ANCHORED**.

### 8.2 Shred / vapor (P7)

Inner photon sheds to sea vapor; recapture sets **χ** — **MODEL** (Sec 7).

### 8.3 Cooper pairs

Josephson enhancement — **MODEL**; specific `T_pair ≈ 4 T_single` **not anchored** — state qualitatively only.

### 8.4 Hartman / τ_tunnel

Wake propagation at **c** — **MODEL** interpretation; attosecond scales **ANCHORED** (experiments).

---

## Chapter 9 — Double slit

### 9.1 Ontology (critical)

**Primary model (section_08):** signal electron + **entangled partner** — one per slit.

**Not primary:** single electron physically through both slits.

\[
\Psi = A_L + A_R,\quad I = |A_L|^2 + |A_R|^2 + 2\mathrm{Re}(A_L A_R^*)
\]

Opposite-phase pump lock: `φ_R = φ_L + π` (Sec 6/8).

### 9.2 Which-path

Detector at slit → `M_n` pins path → **Γ_break** rises → visibility **V → 0** — **ANCHORED** math + **MODEL** language.

### 9.3 ³He vs ⁴He

\[
\Lambda_{^3He} \neq \Lambda_{^4He}
\quad\text{via}\quad
\sigma_{e,i} f_{coin,i}
\]

**MODEL — testable** (O8 discriminator); ratio **~1.05–1.10** is illustrative until calibrated.

### 9.4 Gas decoherence

\[
\Gamma_{partner} \propto n_{gas}\sigma v_{rel}\times(\text{species factors})
\]

**MODEL**; Arndt envelope **ANCHORED** phenomenology.

---

## Review checklist

- [x] Born cos² law **ANCHORED**; Malus derivation **PARTIAL**
- [x] C5: reject sign sketch; add dynamic C(t)
- [x] Double slit = partner model restored
- [x] WKB **ANCHORED**; vapor **MODEL**
- [x] ³He/⁴He tagged testable **MODEL**

**Book Ch 6–9 derivations: COMPLETE (v1.0).**
