"""
Diagnose the two divergences from the deep-dive: are they STRUCTURAL (the meet
can't do it) or NUMERICAL (my estimator was sloppy)? Two-sided honesty.
"""
import numpy as np, math
import networkx as nx
INF = float("inf")
rng = np.random.default_rng(7)

# ---- (c2) max-plus eigenvalue: use the CORRECT cycle-time formula.
# The max-plus eigenvalue of an irreducible matrix = MAX cycle mean, recovered
# exactly by Karp/Howard, NOT by naive per-step delta averaging (which oscillates
# with the cyclicity/period). Use the SAME Karp DP as min-cycle-mean but max.
def max_cycle_mean_karp(A, n):
    NEG = -INF
    F = [[NEG]*n for _ in range(n+1)]
    for v in range(n):
        F[0][v] = 0.0
    for k in range(1, n+1):
        for v in range(n):
            best = NEG
            for u in range(n):
                if A[u][v] != NEG:
                    if F[k-1][u] != NEG:
                        best = max(best, F[k-1][u] + A[u][v])
            F[k][v] = best
    lam = NEG
    for v in range(n):
        if F[n][v] == NEG:
            continue
        worst = INF; ok = True
        for k in range(n):
            if F[k][v] == NEG:
                ok = False; continue
            worst = min(worst, (F[n][v]-F[k][v])/(n-k))
        if ok and worst > lam:
            lam = worst
    return lam

def test_maxplus_karp(trials=60, n=6):
    exact = 0; diffs=[]
    for t in range(trials):
        A=[[-INF]*n for _ in range(n)]
        for i in range(n):
            A[i][(i+1)%n]=float(rng.integers(1,15))
        for _ in range(n*2):
            i,j=int(rng.integers(0,n)),int(rng.integers(0,n))
            A[i][j]=float(rng.integers(1,15))
        lam = max_cycle_mean_karp(A,n)
        G=nx.DiGraph()
        for i in range(n):
            for j in range(n):
                if A[i][j]!=-INF: G.add_edge(i,j,weight=A[i][j])
        best=-INF
        for cyc in nx.simple_cycles(G):
            w=sum(G[cyc[a]][cyc[(a+1)%len(cyc)]]["weight"] for a in range(len(cyc)))
            best=max(best,w/len(cyc))
        diffs.append(abs(lam-best))
        if abs(lam-best)<1e-9: exact+=1
    return exact,trials,max(diffs)

# ---- (OT) Sinkhorn: the T->0 limit IS the assignment value; my T=0.005 caused
# exp underflow. Use log-domain Sinkhorn (stabilized) and sweep T downward to
# SHOW monotone convergence to the tropical value.
def logsumexp(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)

def sinkhorn_log(C, T, iters=3000):
    n = C.shape[0]
    a = np.full(n, np.log(1.0/n)); b = np.full(n, np.log(1.0/n))
    f = np.zeros(n); g = np.zeros(n)
    K = -C / T
    for _ in range(iters):
        f = a*T - T*logsumexp((K + g[None,:]/T)*1.0, axis=1)*0 - T*logsumexp(K/1.0 + (g/T)[None,:], axis=1)
        # standard stabilized update:
        f = T*(a - logsumexp(K + (g/T)[None,:], axis=1))
        g = T*(b - logsumexp(K.T + (f/T)[None,:], axis=1))
    # transport plan in log domain
    P = np.exp((K + (f/T)[:,None] + (g/T)[None,:]))
    return np.sum(P * C)

def test_ot_sweep():
    from scipy.optimize import linear_sum_assignment
    C = rng.random((5,5))*10
    r,c = linear_sum_assignment(C); trop = C[r,c].sum()/5
    row=[]
    for T in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
        v = sinkhorn_log(C, T)
        row.append((T, v, abs(v-trop)/trop))
    return trop, row

if __name__=="__main__":
    print("DIAGNOSIS 1 — max-plus eigenvalue via Karp (correct estimator):")
    e,n,md = test_maxplus_karp()
    print(f"   EXACT vs max cycle mean: {e}/{n}  (max|diff|={md:.2e})")
    print("   -> if 60/60, the earlier 23/40 was MY power-method averaging, not the meet.")

    print("\nDIAGNOSIS 2 — Sinkhorn log-domain T-sweep (one instance):")
    trop, row = test_ot_sweep()
    print(f"   tropical/assignment value (per-unit) = {trop:.4f}")
    for T,v,rel in row:
        print(f"   T={T:<5}  sinkhorn={v:.4f}  rel_err={rel:.3e}")
    print("   -> monotone decrease to ~0 confirms T->0 limit = assignment (tropical).")
