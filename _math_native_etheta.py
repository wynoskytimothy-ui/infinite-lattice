#!/usr/bin/env python3
"""THE TEST THAT COULD OVERTURN THE VERDICT (fast) — derive E(theta) from the lattice's OWN spring phases.
Read the native phase phi(n)=arg(z) from wing_transform (the real 4-way fan), use the transgressor n as the
shared hidden variable, measure the correlation E the lattice actually produces. SHAPE is the referee:
COSINE -> emergent quantum interference (2sqrt2 earned); TRIANGLE/linear -> local model (CHSH<=2, classical)."""
import numpy as np
from aethos_complex_plane import wing_transform
from aethos_lattice import BranchKind


def native_phase(a0, n, wing=1):
    return np.angle(wing_transform(BranchKind.VA1, (a0,), float(n), wing).z)


def main():
    print("=" * 78)
    print("NATIVE E(theta) — from the lattice's own spring phases (no hardcoded cos)")
    print("=" * 78)
    a0 = 5
    NMAX = 4000
    phi_table = np.array([native_phase(a0, n) for n in range(1, NMAX + 1)])   # 4000 real lattice calls
    print(f"\n  native phase phi(n)=arg(z), a0={a0}, n=1..{NMAX}:")
    print(f"    distinct values: {len(np.unique(np.round(phi_table,6)))}; range "
          f"[{phi_table.min():.3f},{phi_table.max():.3f}] rad; std {phi_table.std():.3f}")
    # is the phase ~uniform on the circle, or pinned?
    hist, _ = np.histogram(phi_table % (2*np.pi), bins=12)
    print(f"    phase histogram (12 bins): {hist.tolist()}")

    rng = np.random.default_rng(0)
    phi = phi_table[rng.integers(0, NMAX, 300000)]                            # the source samples

    G = 24
    angles = np.linspace(0, np.pi, G)
    # A(a)=sign(cos(a-phi)); B(b) anti-correlated = -sign(cos(b-phi)). Egrid[i,j]=<A_i B_j>.
    Asign = np.sign(np.cos(angles[:, None] - phi[None, :]))   # (G, S)
    Asign[Asign == 0] = 1
    Egrid = -(Asign @ Asign.T) / len(phi)                     # (G,G); E[i,j] = -<A_i A_j>

    # E(delta) shape: average Egrid over constant (i-j)
    deltas = []; Edelta = []
    for d in range(G):
        vals = [Egrid[i, i - d] for i in range(d, G)]
        deltas.append(angles[d] - angles[0]); Edelta.append(np.mean(vals))
    deltas = np.array(deltas); Edelta = np.array(Edelta)
    cosm = -np.cos(deltas); trim = -(1 - 2*deltas/np.pi)
    s = Edelta[0] / cosm[0] if cosm[0] != 0 else 1.0
    def r2(m):
        m = m * s; ssr = np.sum((Edelta - m)**2); sst = np.sum((Edelta - Edelta.mean())**2)
        return 1 - ssr/sst if sst > 0 else 0.0
    r2_cos, r2_tri = r2(cosm), r2(trim)

    # CHSH: max over all (i,ip,j,jp) using the precomputed grid (vectorized)
    E = Egrid
    S = (E[:, None, :, None] - E[:, None, None, :] + E[None, :, :, None] + E[None, :, None, :])
    chsh = float(np.max(np.abs(S)))

    print(f"\n  E(delta) shape:   R^2(cosine) = {r2_cos:.3f}   R^2(triangle) = {r2_tri:.3f}")
    print(f"  native CHSH (grid-optimized): |S| = {chsh:.4f}   (classical 2.0, Tsirelson 2sqrt2={2*np.sqrt(2):.4f})")
    print("\n" + "=" * 78)
    print("VERDICT (from the native operators):")
    shape = "COSINE (emergent interference!)" if r2_cos > r2_tri + 0.05 else \
            ("TRIANGLE/linear (LOCAL model)" if r2_tri > r2_cos + 0.05 else "neither cleanly")
    print(f"  * E(theta) shape = {shape}")
    if chsh > 2.83:
        print(f"  * |S|={chsh:.3f} > 2sqrt2 -> normalization bug (no physics allows this) — investigate")
    elif chsh > 2.02:
        print(f"  * |S|={chsh:.3f} EXCEEDS classical 2.0 from NATIVE operators -> emergent quantum signal!")
    else:
        print(f"  * |S|={chsh:.3f} <= 2.0 -> the native readout is CLASSICAL (local hidden-variable).")
        print(f"    CONFIRMS the verdict: the lattice geometry is classical; QM's 2sqrt2 must be INSERTED")
        print(f"    by hand via complex amplitudes (the 4 branches are collinear C^1, cannot supply a qubit).")


if __name__ == "__main__":
    main()
