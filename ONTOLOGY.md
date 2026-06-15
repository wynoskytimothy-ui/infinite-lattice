# AETHOS ontology — two constructions

Canonical names. **Do not** use "prime lattice" or "infinity lattice."

---

## A. π lattice (Part I — Ch 1–2, `pi/`)

| | |
|---|---|
| **Seed** | `A₀² = B₀² = C₀²/2` |
| **Step** | `B_{k+1} = C_k/2` |
| **Constraint** | `(1 − A)² + B² = 1` |
| **Addresses** | `(1 − A_k, ±B_k)` |
| **Output** | `S_k → π`; triangles stack to area |
| **Role** | Compton cells, Zeno, ±B entanglement seed |
| **Code** | `pi/constructive_pi.py` |

**Not** the 3D complex plane.

---

## B. 3D complex plane (Part II — Ch 3–7)

### B.1 Construction reading (primary)

| Layer | Content | Tag |
|-------|---------|-----|
| **PRIMITIVE** | Rail `n`; chain `A=(a₁,…,a_k)`; segment `s`; branches VA1–VA4; wings 1–8; meets | **DERIVED** (code) |
| **READOUT** | `(X,Y,Z)=canon_on_chain(b,A,n)` → wing → `z=X+iY`, `ζ=Z` | **DERIVED** |
| **OBJECT** | Trajectory + meet space with readout `(z,ζ)` = the plane | **MODEL** (physics) |

**Not** "assume ℂ×ℝ, then define a map." **Yes** "rules first, coordinates second."

### B.2 Formula pipeline (concrete)

```
1. A = make_chain(kind, k)
2. s = segment_index(A, n)
3. ζ = Z(A,n,s)     — interior: ζ = ΣA when 0 < s < k
4. (X,Y,Z) = canon_on_chain(b, A, n)
5. (X_w,Y_w,Z_w) = apply_vector(..., wing w)
6. Ψ = (z, ζ),  z = X_w + iY_w
```

**VA1** (segments; VA2–VA4 same breaks, different algebra):

| s | (X, Y, ζ) |
|---|-----------|
| 0 | (a₁+a_k, a₁, ζ) |
| 0<s<k | (a_k+n, n, ζ) |
| k | (a_k+n, a_k, ζ) |

**Layer 0** (`|A|=0`): `(n,n,n)` → `z₀=n+ni`, `ζ₀=n`.

**Example:** PRIMES k=5, n=10 → A=(3,5,7,11,13), s=3, ζ=39, VA1 → z=23+10i.

```python
from aethos_sequences import make_chain, SequenceKind
from aethos_complex_plane import wing_transform
from aethos_lattice import BranchKind

psi = wing_transform(BranchKind.VA1, make_chain(SequenceKind.PRIMES, 5), n=10, wing=1)
# psi.z == 23+10j, psi.zeta == 39
```

### B.3 ℂ from wing operators (resolved)

**PROVEN** (`aethos_spring_complex.py`, `tests/test_spring_complex.py`):

Spring pairs `(X,Y)` only — no imported `i` in definitions.

```
S(X,Y)   = (Y, X)           swap corridor
R_x(X,Y) = (-X, Y)          reflect real
i_act    = R_x ∘ S          (X,Y) ↦ (-Y, X)
conj_act = (X, -Y)          equals -R_x on readout z = X + iY
```

Readout: `z = X + iY` ⇒ `i_act` is multiplication by `i`; `i_act ∘ i_act` is negation.
Matches `reflect_real(swap_corridor(z))` in `aethos_complex_rotation.py`.

**PARTIAL:** full field axioms on all trajectories (sampled checks pass; not a closed-form proof on VA1 canon for all n).

### B.4 π lattice ↔ plane (partial functor)

**PROVEN** (`aethos_pi_bridge.py`, `tests/test_pi_bridge.py`):

| Bridge | Statement |
|--------|-----------|
| Layer 0 | `z₀ = n(1+i)` has angle π/4 = `point_on_circle_complex(1,1)` direction |
| Unit i | `point_on_circle_complex(0,1) = (0,1) = i_act(1,0)` |
| Dyadic vertices | `point_on_circle_complex(k,j)` from `pi/constructive_pi.py` |

**PARTIAL** maps (demo scale, not full label equivalence):

| Map | Rule |
|-----|------|
| depth `k` → rail `n` | `n = 2^k` |
| ±B bit path → wing | last 3 bits → wing mask 0..7 |
| vertex `(k,j)` → layer 0 | scale on `n(1+i)` ray by angle-matched projection |

**OPEN:** `canon_on_chain(prime A, n)` equals π walker position at a shared address for all `(k, chain)`.

### B.5 Imaginary complex number (ICN) — MODEL term

**Not** standard \(z = X + iY\) (that is **complex readout** on the spring plane).

| | |
|---|---|
| **ICN** | Prime composite \(C = \prod p_i\) (FTA unique factorization) |
| **Role** | Promoted meet address on the **imaginary encoding axis** (pool primes, L4/L5 products) |
| **Normalize down** | \(C \to\) sorted prime chain \(\to\) `canon_on_chain` \(\to\) \(\Psi=(z,\zeta)\) |
| **Non-primes** | Words, phrases, integers that are not prime **are** composites of primes in this layer |

Promotion ladder (retrieval + ingest): L1 letter primes \(\to\) L2/L3 pool primes \(\to\) ICN products \(\to\) readout \(\Psi\).

```python
# Normalize: composite -> chain -> plane readout
from aethos_intersection_nodes import chain_from_composite
from aethos_complex_plane import wing_transform
from aethos_lattice import BranchKind

chain = chain_from_composite(667)  # e.g. product of two pool primes
psi = wing_transform(BranchKind.VA1, chain, n=7, wing=1)
```

**32 chambers** = one spring vector, rotated/labeled (4 branches × 8 wings) — not 32 independent ICNs.

**Do not confuse:** ICN (composite address) vs complex readout \(z\) vs full 3-way meet \(\Psi=(z,\zeta)\).

Code: `aethos_intersection_nodes.py`, `aethos_phrase_composite.py`, `aethos_promotion.py`, `aethos_hub_signature.py`.

### B.6 Vector naming hierarchy — MODEL terms (patent / pitch glossary)

**Prefer vector over axis** — vectors have direction and magnitude; the engine rotates and routes; axes are static chart lines.

| AETHOS term | Replaces (informal) | Meaning | Code anchor | Tag |
|-------------|---------------------|---------|-------------|-----|
| **Imaginary vector** | imaginary axis | Phase engine: spring \((X,Y)\), operators \(R_x,S,i_{\mathrm{act}}\), **branch** \(b\) (main stick, 4 formula fans) | `aethos_spring_complex`, `canon_on_chain`, `BranchKind` | **MODEL** |
| **Complex vector** | 2-way meet “path” | Active trajectory when two rails interlock: solo swap `bank(a)@n=p ↔ bank(p)@n=a`; readout \(z=X+iY\) along the witness | `swap_meet`, `trigger_history` | **DERIVED** |
| **Complex number** | 3-way equalization node | **Final locked coordinate** \(\Psi=(z,\zeta)\) after triple meet (not textbook \(a+bi\) alone) | `triple_equalization`, `ComplexPlane3D`, `wing_transform` | **MODEL** |

**Chain of command (mechanics):**

```text
  Imaginary vector  →  turns the gear (rotation / branch fan)
  Complex vector    →  drives the path (2-way meet witness)
  Complex number    →  locks the room (3-way node Ψ)
```

**Joystick transmission (pitch metaphor):** main stick = branch \(b\) (4 gears); thumb stick = wing \(w\) (8 octants); chain \(A\) + rail \(n\) = which road and mile marker. **32 chambers** = one physical vector, two selectors — not 32 levers.

**Disambiguation (required in patents):**

| Standard math | AETHOS name |
|---------------|-------------|
| \(z = a + bi\) | **complex readout** (spring 2-way meet) |
| \(a + bi\) as final address | **complex number** = \(\Psi=(z,\zeta)\) at 3-way meet only |
| Prime composite \(C\) | **ICN** (§B.5) on encoding layer |

### B.7 Still OPEN

| Item | Status |
|------|--------|
| Full functor π-address ↔ `(A,b,w,n)` at all chains | **OPEN** |
| Hilbert inner product uniqueness | **OPEN** |

### B.8 Deprecated names

| Old | Use |
|-----|-----|
| prime / infinity / φ-prime lattice | **3D complex plane** |

Primes = `SequenceKind.PRIMES` for chain `A`. Infinity = unbounded `n`, `k`, depth.

---

## C. Relation π ↔ plane

| | π lattice | 3D complex plane |
|---|-----------|------------------|
| Parameter | Bisection depth k | Transgressor n + chain k |
| Branching | ±B | 4×8 chambers |
| Link | **PARTIAL** functor (`aethos_pi_bridge.py`); full label map **OPEN** |

---

## D. One line

> **π lattice** tiles the circle. **3D complex plane** is the trajectory space whose readout is **Ψ=(z,ζ)**, built by the lattice formula.

Code: `aethos_complex_plane.py`. Book: Ch 3 §3.1–3.3. Audit: `derivations/book_ch03-05_3d_complex_plane.md`.
