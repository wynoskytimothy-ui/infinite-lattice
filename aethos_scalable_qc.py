"""aethos_scalable_qc — the SCALABLE glass-box quantum computer (dual backend, auto-switching).

The universality result + the stabilizer speedup, combined into one engine:
  * Clifford circuits (no T)  -> the POLYNOMIAL stabilizer tableau (CHP). Scales to THOUSANDS of qubits.
  * the moment a T gate (magic, from the pi ladder) appears -> fall back to the exact statevector.
It auto-switches and REPORTS THE MAGIC BUDGET (the T-count) -- the exact, countable resource that sets the
classical<->quantum boundary. Gates are native: H,S from the wing operators; T = pi-ladder e^{i pi/4}; CNOT.

This makes the glass-box QC genuinely scalable AND honest: you see how much magic each algorithm spends.
"""
from __future__ import annotations
import numpy as np
from aethos_glassbox_qc import GlassBoxQC   # the statevector backend (with the native gates)


# ============================================================================
#  CHP stabilizer backend (Aaronson-Gottesman): polynomial, with measurement
# ============================================================================
def _g(x1, z1, x2, z2):
    """Phase contribution of multiplying Pauli (x1,z1) into (x2,z2), per column (values in {-1,0,1})."""
    out = np.zeros(x1.shape, np.int64)
    m11 = (x1 == 1) & (z1 == 1); out[m11] = z2[m11].astype(np.int64) - x2[m11].astype(np.int64)
    m10 = (x1 == 1) & (z1 == 0); out[m10] = z2[m10].astype(np.int64) * (2 * x2[m10].astype(np.int64) - 1)
    m01 = (x1 == 0) & (z1 == 1); out[m01] = x2[m01].astype(np.int64) * (1 - 2 * z2[m01].astype(np.int64))
    return out


class CHP:
    """Polynomial stabilizer tableau: 2n+1 rows (n destabilizers, n stabilizers, 1 scratch)."""
    def __init__(self, n, seed=0):
        self.n = n; self.rng = np.random.default_rng(seed)
        m = 2 * n + 1
        self.X = np.zeros((m, n), np.uint8); self.Z = np.zeros((m, n), np.uint8); self.R = np.zeros(m, np.uint8)
        for i in range(n):
            self.X[i, i] = 1            # destabilizers = X_i
            self.Z[n + i, i] = 1        # stabilizers   = Z_i

    def h(self, a):
        self.R ^= (self.X[:, a] & self.Z[:, a])
        self.X[:, a], self.Z[:, a] = self.Z[:, a].copy(), self.X[:, a].copy()

    def s(self, a):
        self.R ^= (self.X[:, a] & self.Z[:, a]); self.Z[:, a] ^= self.X[:, a]

    def cnot(self, a, b):
        self.R ^= (self.X[:, a] & self.Z[:, b] & (self.X[:, b] ^ self.Z[:, a] ^ 1))
        self.X[:, b] ^= self.X[:, a]; self.Z[:, a] ^= self.Z[:, b]

    def x(self, a): self.h(a); self.z(a); self.h(a)
    def z(self, a): self.s(a); self.s(a)

    def _rowsum(self, h, i):
        s = (2 * int(self.R[h]) + 2 * int(self.R[i]) + int(_g(self.X[i], self.Z[i], self.X[h], self.Z[h]).sum())) % 4
        self.R[h] = 1 if s == 2 else 0
        self.X[h] ^= self.X[i]; self.Z[h] ^= self.Z[i]

    def measure(self, a):
        n = self.n
        p = next((i for i in range(n, 2 * n) if self.X[i, a]), None)
        if p is not None:                                  # random outcome
            for i in range(2 * n):
                if i != p and self.X[i, a]: self._rowsum(i, p)
            self.X[p - n] = self.X[p].copy(); self.Z[p - n] = self.Z[p].copy(); self.R[p - n] = self.R[p]
            self.X[p] = 0; self.Z[p] = 0; self.Z[p, a] = 1; self.R[p] = self.rng.integers(0, 2)
            return int(self.R[p])
        self.X[2 * n] = 0; self.Z[2 * n] = 0; self.R[2 * n] = 0      # determined outcome
        for i in range(n):
            if self.X[i, a]: self._rowsum(2 * n, n + i)
        return int(self.R[2 * n])


# ============================================================================
#  The dual-backend dispatcher
# ============================================================================
class ScalableQC:
    def __init__(self, n):
        self.n = n; self.gates = []

    def h(self, q): self.gates.append(("h", q))
    def s(self, q): self.gates.append(("s", q))
    def t(self, q): self.gates.append(("t", q))
    def x(self, q): self.gates.append(("x", q))
    def z(self, q): self.gates.append(("z", q))
    def cnot(self, c, t): self.gates.append(("cnot", (c, t)))

    def magic_budget(self): return sum(1 for g, _ in self.gates if g == "t")

    def run(self, measure=None, verbose=True):
        t_count = self.magic_budget()
        if t_count == 0:
            backend = "STABILIZER (polynomial)"
            chp = CHP(self.n)
            for g, q in self.gates:
                getattr(chp, g)(*q) if isinstance(q, tuple) else getattr(chp, g)(q)
            out = {q: chp.measure(q) for q in (measure or [])}
            mem = f"{(2*self.n+1)*2*self.n/1e6:.2f} MB"
        else:
            backend = "STATEVECTOR (exact, magic present)"
            qc = GlassBoxQC(self.n)
            for g, q in self.gates:
                getattr(qc, g)(*q) if isinstance(q, tuple) else getattr(qc, g)(q)
            out = {q: float(np.sum(qc.probs().reshape([2]*self.n).take(1, axis=q))) for q in (measure or [])}
            mem = f"2^{self.n} amplitudes ({2**self.n*16/1e6:.0f} MB)" if self.n <= 26 else f"2^{self.n} amplitudes (INTRACTABLE)"
        if verbose:
            print(f"    backend: {backend} | n={self.n} | gates={len(self.gates)} | MAGIC BUDGET (T-count)={t_count} | mem={mem}")
        return out, {"backend": backend, "magic": t_count, "mem": mem}


def _demo():
    print("=" * 80)
    print("SCALABLE GLASS-BOX QC — auto-switching stabilizer <-> statevector, magic-budget report")
    print("=" * 80)
    print("\n  (A) huge CLIFFORD circuit (0 magic) -> stabilizer backend, thousands of qubits:")
    import time
    qc = ScalableQC(3000); rng = np.random.default_rng(1)
    for _ in range(30000):
        g = rng.integers(0, 3)
        if g == 0: qc.h(int(rng.integers(0, 3000)))
        elif g == 1: qc.s(int(rng.integers(0, 3000)))
        else:
            a, b = rng.integers(0, 3000, 2)
            if a != b: qc.cnot(int(a), int(b))
    t0 = time.perf_counter(); qc.run(measure=[0, 1, 2]); print(f"    ran in {(time.perf_counter()-t0)*1000:.0f} ms")

    print("\n  (B) Bell pair (Clifford) -> stabilizer backend, verify measurement correlation:")
    same = 0
    for _ in range(400):
        b = ScalableQC(2); b.h(0); b.cnot(0, 1)
        out, _ = b.run(measure=[0, 1], verbose=False)
        same += (out[0] == out[1])
    print(f"    stabilizer Bell measurements correlated: {same}/400 (must be 400 -> |00>+|11>)")

    print("\n  (C) circuit WITH magic (T gates) -> statevector fallback + magic budget:")
    m = ScalableQC(3); m.h(0); m.t(0); m.h(1); m.t(1); m.cnot(0, 2); m.t(2)
    m.run(measure=[0])

    print("\n  VERDICT: Clifford runs POLYNOMIAL (3000 qubits in ms); magic auto-falls-back to exact statevector")
    print("  and the T-count (magic budget) is reported -- the countable classical<->quantum resource, from the pi ladder.")


if __name__ == "__main__":
    _demo()
