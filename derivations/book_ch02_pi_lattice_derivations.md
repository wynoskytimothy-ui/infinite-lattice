# Book Chapter 2 — π Lattice: Formal Derivations

**Not** the 3D complex plane (Ch 3–7). See [`ONTOLOGY.md`](../ONTOLOGY.md).

Maps *Packets and Strings* Ch 2 to `pi/constructive_pi.py` and Sec 1.10 (v1.2).

---

## 2.1 Notation (book ↔ code)

| Book label | `constructive_pi` | Meaning |
|------------|-------------------|---------|
| `B` (halving leg) | `A` | `B_{k+1} = C_k/2` |
| `A` (sagitta complement) | `1-\sqrt{1-A_{\mathrm{code}}^2}` chain | `A_{k+1}=1-\sqrt{1-B_{k+1}^2}` |
| `C` | `C` | hypotenuse / chord |
| `(1-A,\,\pm B)` | unit-circle coords | satisfies (1.2) |

**Forbidden:** mixing book and code labels in one equation block without the table above.

---

## 2.2 Seed (iteration k = 0)

Inscribed square on unit circle; quadrant seed triangle:

\[
A_0 = B_0 = 1,\quad C_0 = \sqrt{2},\quad N_0 = 4
\]

**Seed identity (eq. 1.1 — 45° triangle only):**

\[
A_0^2 = B_0^2 = \frac{C_0^2}{2}
\]

**Status:** **PROVEN** (arithmetic).

**Not valid:** `C_0 = 0` or `C_0 \to 2 \Rightarrow B_1 = 1` (removed from book Ch 2).

---

## 2.3 Recurrence (iteration k → k+1)

\[
B_{k+1} = \frac{C_k}{2}
\]
\[
A_{k+1} = 1 - \sqrt{1 - B_{k+1}^2}
\]
\[
C_{k+1} = \sqrt{A_{k+1}^2 + B_{k+1}^2}
\]
\[
N_{k+1} = 2 N_k,\quad N_0 = 4
\]

Equivalent squared step (book eq. 2.2):

\[
B_{k+1}^2 = \frac{C_k^2}{4}
\]

**Not** `B_{k+1}^2 = C_k^2/2` except at the seed triangle.

**Status:** **PROVEN** in `pi_recurrence`; implementation **ANCHORED**.

---

## 2.4 Unit-circle constraint

\[
(1 - A_k)^2 + B_k^2 = 1
\qquad\Leftrightarrow\qquad
x_k = 1 - A_k = \sqrt{1 - B_k^2},\quad y_k = \pm B_k
\]

**Status:** **PROVEN** (algebra from recurrence construction).

---

## 2.5 First iterations (book numbers)

| k | `B_k` | `A_k` | `x_k=1-A_k` | `C_k` | `N_k` |
|---|-------|-------|-------------|-------|-------|
| 0 | 1 | 1 | 0 | √2 | 4 |
| 1 | √2/2 | 1−√2/2 | √2/2 | √(2−√2) | 8 |
| 2 | √(2−√2)/2 | … | 0.9239 | 0.3902 | 16 |
| 3 | 0.1951 | … | 0.9808 | 0.1960 | 32 |

Mirror partners `(x_k,\,+B_k)` and `(x_k,\,-B_k)` share the same `x_k`.

---

## 2.6 π convergence

Inscribed `N_k`-gon edge chord `C_k`:

\[
P_k = N_k C_k \to 2\pi
\qquad\Rightarrow\qquad
\pi = \lim_{k\to\infty} \frac{N_k C_k}{2}
= \lim_{k\to\infty} 2^{k+1} C_k
\]

(since `N_k = 4\cdot 2^k`).

**Cumulative area** from sliver triangles (`pi_recurrence` field `area`):

\[
S_k \to \pi
\]

**Forbidden book form:** `π = lim 2^{k-1} C_k` or `P_k = 2^k C_k` without `N_k`.

Numerical half-perimeter estimates (verified):

| k | `N_k` | `N_k C_k / 2` | `|error|` |
|---|-------|---------------|-----------|
| 1 | 8 | 3.06147 | 8.0×10⁻² |
| 5 | 128 | 3.14128 | 3.1×10⁻⁴ |
| 10 | 4096 | 3.14159235 | 3.1×10⁻⁷ |
| 20 | 4194304 | 3.1415926536 | 2.9×10⁻¹³ |

---

## 2.7 Nested radicals

All `B_k` at finite `k` lie in `\mathbb{Q}(\sqrt{2})` (nested `\sqrt{2\pm\sqrt{2\pm\cdots}}`).

**Status:** **PROVEN** (field closure). Limit `π` is **not** in this field.

---

## 2.8 Bifurcation tree

Binary choices `+B` / `−B` at each depth: `2^k` leaf paths from a quadrant anchor.

**Full circle vertex count** after `k` bisections: `N_k = 4\cdot 2^k` (inscribed polygon corners).

---

## 2.9 Mirrored positions and entanglement (preview)

**MODEL (Ch 7):** entangled pair occupies mirrored `y=\pm B_k` at shared `x_k`.

**C5 mandate:** joint mechanical ripple on shared substrate — **not** `sgn(cos θ)` half-plane sketch.

**P1a / microcausality:** global constraint does **not** imply FTL signaling; `[O_A,O_B]=0` at spacelike separation.

**T:** Bell violation + no-signaling tests.

---

## 2.10 Zeno (preview — full treatment Ch 15 / Sec 12)

| Layer | Statement | Tag |
|-------|-----------|-----|
| Mathematical | `\sum_{n=1}^\infty 2^{-n} = 1` — infinite halving sums to finite span | **PROVEN** |
| Lattice | finite `k` ⇒ finite chord `C_k > 0`; no zero-width realized frame | **PROVEN** |
| Physical cell | `λ_C` sets address scale (Sec 1.10); not a stop to mathematical descent | **MODEL** |

**Not claimed:** physical space cannot be subdivided mathematically below `λ_C`; claimed: **no realized instant** + cell is operational floor for particles.

---

## 2.11 Energy / bounce links (honest status)

| Book eq. | Verdict |
|----------|---------|
| `E_{\mathrm{total}}(k) = k\, m c^2` | **REJECT** as physics — dimensionally a count×energy, not derived |
| `dE/dt = m^2 c^4 / h` | **WRONG** — correct pump power scale `\sim m c^2 f_b = m^2 c^4/(2h)` |

**Allowed:** interpret iteration depth `k` as **address refinement**, not accumulated rest energy.

---

## 2.12 Open problems

| ID | Claim | Status |
|----|-------|--------|
| Oπ-1 | `k_{\max} \approx \log_2(\lambda_C/\ell_P) \sim 80` | **MODEL** estimate |
| Oπ-2 | Simplest structure generating QM+SR | **OPEN** conjecture |
| Oπ-3 | Branch choice (+B vs −B) = spin / measurement | **MODEL** → Ch 6–7 |
| Oπ-4 | Lattice substrate ontology | **OPEN** |

---

## 2.13 Review checklist

- [x] Recurrence synced with `pi/constructive_pi.py`
- [x] π limit uses `N_k C_k/2` not `2^{k-1} C_k`
- [x] Seed `C_0=\sqrt2` documented; `C_0=0` rejected
- [x] Book/code notation table fixed
- [x] Entanglement preview respects C5 + P1a
- [x] Zeno aligned with Sec 12 (no realized instant)
- [x] Bad energy recursions flagged

**Book Ch 2 derivations: COMPLETE (v1.0).**
