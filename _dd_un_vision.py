"""
Deep-dive probe: aethos_master MNIST PrimeLatticeVision vs fair baselines.

Runs the ACTUAL packaged engine (no reimplementation) on identical train/test
splits, against:
  - kNN (k=3) on raw pixels
  - logistic regression on raw pixels
  - tiny MLP (sklearn MLPClassifier) on raw pixels
  - logistic regression on the SAME aethos structural features (ablation:
    does the lattice geometry help, or is it the ridge-OLS + manifold expand?)

PYTHONPATH must include C:/Users/wynos/aethos_master/src
"""
import sys, time, os
sys.path.insert(0, r"C:/Users/wynos/aethos_master/src")
import numpy as np

from aethos_master.mnist import PrimeLatticeVision, load_mnist

def timed(fn, *a, **k):
    t0 = time.time(); r = fn(*a, **k); return r, time.time() - t0

def main(n_train, n_test, do_full_feat_baseline=True):
    print(f"=== MNIST probe: n_train={n_train} n_test={n_test} ===", flush=True)
    (data, t_load) = timed(load_mnist, max_train=n_train, max_test=n_test)
    X_tr, y_tr, X_te, y_te = data
    print(f"loaded {len(X_tr)} train / {len(X_te)} test in {t_load:.1f}s", flush=True)

    results = {}

    # --- AETHOS PrimeLatticeVision (the real engine) ---
    model = PrimeLatticeVision()
    t0 = time.time()
    # silence its verbose prints
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model.fit(X_tr, y_tr)
    t_fit = time.time() - t0
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        preds = model.predict(X_te)
    t_pred = time.time() - t0
    acc = (preds == y_te).mean()
    nfeat = model.n_features
    nfeat_exp = getattr(model, "n_features_expanded", None)
    results["aethos_prime_lattice"] = dict(acc=acc, fit_s=t_fit, pred_s=t_pred,
                                           nfeat=nfeat, nfeat_exp=nfeat_exp)
    print(f"[AETHOS]  acc={acc:.4f}  fit={t_fit:.1f}s pred={t_pred:.1f}s  "
          f"feat={nfeat}->{nfeat_exp}", flush=True)

    # --- Baseline 1: kNN k=3 on raw pixels ---
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_tr, y_tr)
    t0 = time.time(); pk = knn.predict(X_te); tk = time.time() - t0
    ak = (pk == y_te).mean()
    results["knn_k3_pixels"] = dict(acc=ak, pred_s=tk)
    print(f"[kNN3 px] acc={ak:.4f}  pred={tk:.1f}s", flush=True)

    # --- Baseline 2: Logistic regression on raw pixels ---
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=200, C=1.0, n_jobs=-1)
    with contextlib.redirect_stdout(buf):
        lr.fit(X_tr / 255.0, y_tr)
    al = (lr.predict(X_te / 255.0) == y_te).mean()
    results["logreg_pixels"] = dict(acc=al)
    print(f"[LR px]   acc={al:.4f}", flush=True)

    # --- Baseline 3: tiny MLP on raw pixels ---
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=60, random_state=0)
    with contextlib.redirect_stdout(buf):
        mlp.fit(X_tr / 255.0, y_tr)
    am = (mlp.predict(X_te / 255.0) == y_te).mean()
    results["mlp_100_pixels"] = dict(acc=am)
    print(f"[MLP px]  acc={am:.4f}", flush=True)

    # --- ABLATION: same aethos structural features, plain logreg classifier ---
    # This isolates: do the LATTICE FEATURES carry the signal, or does the
    # ridge-OLS + manifold expansion do the work?
    if do_full_feat_baseline:
        t0 = time.time()
        Ftr = np.array([model._extract_features(x) for x in X_tr])
        Fte = np.array([model._extract_features(x) for x in X_te])
        t_feat = time.time() - t0
        # standardize
        mu = Ftr.mean(0); sd = Ftr.std(0) + 1e-9
        Ftr_n = (Ftr - mu) / sd
        Fte_n = (Fte - mu) / sd
        lrf = LogisticRegression(max_iter=300, C=1.0, n_jobs=-1)
        with contextlib.redirect_stdout(buf):
            lrf.fit(Ftr_n, y_tr)
        af = (lrf.predict(Fte_n) == y_te).mean()
        results["logreg_on_aethos_features"] = dict(acc=af, feat_s=t_feat)
        print(f"[LR aeth] acc={af:.4f}  (logreg on aethos {Ftr.shape[1]} feats, "
              f"feat={t_feat:.1f}s)", flush=True)

        # And kNN on aethos features (does the geometry cluster well?)
        knnf = KNeighborsClassifier(n_neighbors=3)
        knnf.fit(Ftr_n, y_tr)
        akf = (knnf.predict(Fte_n) == y_te).mean()
        results["knn_on_aethos_features"] = dict(acc=akf)
        print(f"[kNN aeth]acc={akf:.4f}  (kNN3 on aethos feats)", flush=True)

    print("\nSUMMARY", flush=True)
    for k, v in results.items():
        print(f"  {k:32s} acc={v['acc']:.4f}", flush=True)
    return results

if __name__ == "__main__":
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    n_test = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    main(n_train, n_test)
