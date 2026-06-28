"""Full-data baselines using cached aethos features + raw pixels. 60k/10k."""
import sys, time, io, contextlib, os
sys.path.insert(0, r"C:/Users/wynos/aethos_master/src")
import numpy as np
from aethos_master.mnist import load_mnist
buf = io.StringIO()
SCR = r"C:/Users/wynos/New folder (3)"

X_tr,y_tr,X_te,y_te = load_mnist()
Ftr = np.load(os.path.join(SCR,"_mnist_ftr_full.npy"))
Fte = np.load(os.path.join(SCR,"_mnist_fte_full.npy"))
print(f"feats {Ftr.shape} {Fte.shape}", flush=True)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

mu=Ftr.mean(0); sd=Ftr.std(0)+1e-9
lrf=LogisticRegression(max_iter=400)
with contextlib.redirect_stdout(buf): lrf.fit((Ftr-mu)/sd,y_tr)
print(f"[logreg on AETHOS feats]  {(lrf.predict((Fte-mu)/sd)==y_te).mean():.4f}", flush=True)

t0=time.time()
knn=KNeighborsClassifier(n_neighbors=3); knn.fit(X_tr,y_tr)
print(f"[kNN3 raw pixels FULL]    {(knn.predict(X_te)==y_te).mean():.4f}  ({time.time()-t0:.0f}s)", flush=True)

mlp=MLPClassifier(hidden_layer_sizes=(100,),max_iter=100,random_state=0)
with contextlib.redirect_stdout(buf): mlp.fit(X_tr/255.,y_tr)
print(f"[MLP-100 raw pixels FULL] {(mlp.predict(X_te/255.)==y_te).mean():.4f}", flush=True)

lr=LogisticRegression(max_iter=200)
with contextlib.redirect_stdout(buf): lr.fit(X_tr/255.,y_tr)
print(f"[logreg raw pixels FULL]  {(lr.predict(X_te/255.)==y_te).mean():.4f}", flush=True)

# kNN on aethos feats
knnf=KNeighborsClassifier(n_neighbors=3); knnf.fit((Ftr-mu)/sd,y_tr)
print(f"[kNN3 on AETHOS feats]    {(knnf.predict((Fte-mu)/sd)==y_te).mean():.4f}", flush=True)
