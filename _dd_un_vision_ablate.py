"""Feature-group ablation: which parts of the 543-dim aethos feature vector
carry the signal? Distinguishes the genuinely-lattice parts (8-vector momentum,
connectivity, curvature, hole topology) from generic image stats (density grids,
projections, downsampled intensity).

Builds index ranges by re-deriving the feature layout from vision.py order.
"""
import sys, time, io, contextlib
sys.path.insert(0, r"C:/Users/wynos/aethos_master/src")
import numpy as np
from aethos_master.mnist import PrimeLatticeVision, load_mnist
from sklearn.linear_model import LogisticRegression
buf = io.StringIO()

# Feature layout (from vision.py _extract_features, in order):
# 0-2   : ink stats (3)
# 3-5   : hole topology (3)             <- LATTICE-ish (topology)
# 6-8   : endpoint/edge/junction frac (3) <- LATTICE (8-vec connectivity)
# 9-17  : connectivity_counts/na (9)    <- LATTICE (8-vec)
# 18-41 : gradient_hist 8x3 (24)        <- LATTICE (8-vec gradient)
# 42-73 : momentum_hist 8x4 (32)        <- LATTICE (Prime-2 4-state)
# 74-77 : curvature summary (4)         <- LATTICE (curvature)
# 78-85 : per-quadrant convex/concave 4x2 (8) <- LATTICE (curvature)
# 86-101: regional density 4x4 (16)     <- GENERIC (density grid)
# 102-150: density 7x7 (49)             <- GENERIC
# 151-155: geometric moments (5)        <- GENERIC
# 156-161: symmetry (6)                 <- GENERIC
# 162-189: H/V projections 14+14 (28)   <- GENERIC
# 190-217: crossings 14+14 (28)         <- semi-lattice (run transitions)
# 218-..: profile features              <- GENERIC
# then quadrant struct, multi-res intensity 49+16+196 (261) <- GENERIC pixels

def main(n_train=8000, n_test=8000):
    X_tr,y_tr,X_te,y_te = load_mnist(max_train=n_train, max_test=n_test)
    m = PrimeLatticeVision()
    t0=time.time()
    Ftr = np.array([m._extract_features(x) for x in X_tr])
    Fte = np.array([m._extract_features(x) for x in X_te])
    print(f"feat dim={Ftr.shape[1]} ({time.time()-t0:.0f}s)", flush=True)

    def evalset(idx, name):
        a = Ftr[:,idx]; b = Fte[:,idx]
        mu=a.mean(0); sd=a.std(0)+1e-9
        lr = LogisticRegression(max_iter=400, C=1.0)
        with contextlib.redirect_stdout(buf): lr.fit((a-mu)/sd, y_tr)
        acc=(lr.predict((b-mu)/sd)==y_te).mean()
        print(f"  {name:42s} dims={len(idx):4d}  acc={acc:.4f}", flush=True)
        return acc

    D = Ftr.shape[1]
    lattice_8vec = list(range(3,86))          # topology+connectivity+gradient+momentum+curvature
    generic_density = list(range(86,156))     # density grids + moments
    projections = list(range(156,218))        # symmetry+projections+crossings
    # everything from 218 to end = profiles + quadrant + multi-res intensity (raw pixel)
    raw_intensity = list(range(D-261, D))     # last 261 = multi-res downsampled intensity
    print("--- single-group logreg ---", flush=True)
    evalset(list(range(D)), "ALL features")
    evalset(lattice_8vec, "LATTICE only (8vec/momentum/curv/topo)")
    evalset(generic_density, "generic density grids + moments")
    evalset(projections, "projections/symmetry/crossings")
    evalset(raw_intensity, "raw multi-res downsampled intensity")
    print("--- leave-one-out ---", flush=True)
    allidx=set(range(D))
    evalset(sorted(allidx-set(lattice_8vec)), "ALL minus LATTICE")
    evalset(sorted(allidx-set(raw_intensity)), "ALL minus raw-intensity")

if __name__ == "__main__":
    main()
