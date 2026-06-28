"""Full standard MNIST: 60000 train / 10000 test.
AETHOS PrimeLatticeVision vs canonical baselines (kNN k=3, MLP-100, logreg).
Caches extracted features to .npy so re-runs are fast.
"""
import sys, time, io, contextlib, os
sys.path.insert(0, r"C:/Users/wynos/aethos_master/src")
import numpy as np
from aethos_master.mnist import PrimeLatticeVision, load_mnist

SCR = r"C:/Users/wynos/New folder (3)"
buf = io.StringIO()

def main():
    X_tr, y_tr, X_te, y_te = load_mnist()  # full 60k/10k
    print(f"full load: {len(X_tr)}/{len(X_te)}", flush=True)

    model = PrimeLatticeVision()
    # Extract (cache)
    ftr_p = os.path.join(SCR, "_mnist_ftr_full.npy")
    fte_p = os.path.join(SCR, "_mnist_fte_full.npy")
    if os.path.exists(ftr_p) and os.path.exists(fte_p):
        Ftr = np.load(ftr_p); Fte = np.load(fte_p)
        model.n_features = Ftr.shape[1]
        print(f"loaded cached feats {Ftr.shape} {Fte.shape}", flush=True)
    else:
        t0 = time.time()
        f0 = model._extract_features(X_tr[0]); model.n_features = len(f0)
        Ftr = np.empty((len(X_tr), len(f0)), dtype=np.float64); Ftr[0] = f0
        for i in range(1, len(X_tr)):
            Ftr[i] = model._extract_features(X_tr[i])
            if (i+1) % 10000 == 0: print(f"  train feat {i+1} ({time.time()-t0:.0f}s)", flush=True)
        Fte = np.array([model._extract_features(x) for x in X_te])
        print(f"extracted in {time.time()-t0:.0f}s", flush=True)
        np.save(ftr_p, Ftr); np.save(fte_p, Fte)

    # AETHOS native classifier: ridge OLS on manifold-expanded features
    def expand(F):
        return np.hstack([F, F**2, np.sqrt(np.abs(F)), np.log1p(np.abs(F))])
    Atr = np.hstack([expand(Ftr), np.ones((len(Ftr),1))])
    Ate = np.hstack([expand(Fte), np.ones((len(Fte),1))])
    Y = np.zeros((len(y_tr),10)); Y[np.arange(len(y_tr)), y_tr] = 1.0
    lam = 0.5
    t0 = time.time()
    AtA = Atr.T@Atr + lam*np.eye(Atr.shape[1]); AtY = Atr.T@Y
    beta = np.linalg.solve(AtA, AtY)
    t_fit = time.time()-t0
    pred = (Ate@beta).argmax(1)
    acc = (pred==y_te).mean()
    print(f"[AETHOS native ridge] acc={acc:.4f} (fit {t_fit:.1f}s, {Atr.shape[1]} feat)", flush=True)

    # logreg on aethos features
    from sklearn.linear_model import LogisticRegression
    mu=Ftr.mean(0); sd=Ftr.std(0)+1e-9
    lrf = LogisticRegression(max_iter=400, C=1.0)
    with contextlib.redirect_stdout(buf): lrf.fit((Ftr-mu)/sd, y_tr)
    af = (lrf.predict((Fte-mu)/sd)==y_te).mean()
    print(f"[logreg on aethos feats] acc={af:.4f}", flush=True)

    # Baselines on raw pixels (full data)
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3); knn.fit(X_tr, y_tr)
    ak = (knn.predict(X_te)==y_te).mean()
    print(f"[kNN3 pixels FULL] acc={ak:.4f}", flush=True)

    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, random_state=0)
    with contextlib.redirect_stdout(buf): mlp.fit(X_tr/255.0, y_tr)
    am = (mlp.predict(X_te/255.0)==y_te).mean()
    print(f"[MLP-100 pixels FULL] acc={am:.4f}", flush=True)

    lr = LogisticRegression(max_iter=200)
    with contextlib.redirect_stdout(buf): lr.fit(X_tr/255.0, y_tr)
    al = (lr.predict(X_te/255.0)==y_te).mean()
    print(f"[logreg pixels FULL] acc={al:.4f}", flush=True)

    print("\nFULL-MNIST SUMMARY (60k/10k):", flush=True)
    print(f"  AETHOS native ridge          {acc:.4f}", flush=True)
    print(f"  logreg on AETHOS feats       {af:.4f}", flush=True)
    print(f"  kNN3 raw pixels              {ak:.4f}", flush=True)
    print(f"  MLP-100 raw pixels           {am:.4f}", flush=True)
    print(f"  logreg raw pixels            {al:.4f}", flush=True)

if __name__ == "__main__":
    main()
