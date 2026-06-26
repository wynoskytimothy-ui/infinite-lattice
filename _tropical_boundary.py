"""
BOUNDARY PROBE: separate what iterated meet solves in POLY time (genuine
min-plus matrix algebra) from what only the n! tropical-permanent brute force
solves. Two-sided honesty on the assignment claim.
"""
import numpy as np, itertools
from scipy.optimize import linear_sum_assignment

rng = np.random.default_rng(1)
INF = float("inf")

def trop_matmul(A, B):
    n, m, p = len(A), len(B[0]), len(B)
    C = [[INF]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            best = INF
            for k in range(p):
                if A[i][k] != INF and B[k][j] != INF:
                    best = min(best, A[i][k] + B[k][j])
            C[i][j] = best
    return C

def trop_permanent(C):
    n=len(C); best=INF
    for perm in itertools.permutations(range(n)):
        best=min(best, sum(C[i][perm[i]] for i in range(n)))
    return best

# Q1: does min-plus matrix POWER (the poly-time iterated meet) equal the
# assignment value? It should NOT — matrix power = cheapest WALK, can reuse
# columns (not a permutation). Demonstrate divergence.
print("Q1: poly-time min-plus matrix power vs assignment (should DIVERGE)")
div=0; n=5
for t in range(100):
    C = rng.integers(1,30,size=(n,n)).astype(float).tolist()
    # min-plus power A^(n-1): cheapest length-(n-1) walk i->j
    P=C
    for _ in range(n-1): P=trop_matmul(P,C)
    walk_diag_min = min(P[i][i] for i in range(n))  # cheapest closed walk-ish proxy
    r,c=linear_sum_assignment(np.array(C)); val=np.array(C)[r,c].sum()
    perm=trop_permanent(C)
    assert abs(perm-val)<1e-9   # n! tropical permanent == assignment, always
    if abs(walk_diag_min - val) > 1e-9:
        div+=1
print(f"   tropical PERMANENT (n! brute) == scipy assignment: 100/100 (exact value)")
print(f"   poly-time min-plus matrix POWER != assignment value: {div}/100 (diverges, as expected)")

# Q2: Is the Hungarian algorithm itself expressible as iterated meet? It needs
# dual potentials + augmenting paths (Bellman-Ford style = min-plus relaxation),
# so the INNER shortest-path is iterated meet, but the OUTER matching is not a
# single meet closure. Confirm by counting ops scaling.
print()
print("Q2: assignment via min-plus is BRUTE (n!), not a meet closure:")
import time
for n in (6,8,9):
    C = rng.integers(1,30,size=(n,n)).astype(float).tolist()
    t0=time.perf_counter(); perm=trop_permanent(C); t1=time.perf_counter()
    r,c=linear_sum_assignment(np.array(C)); t2=time.perf_counter()
    print(f"   n={n}: trop-permanent {t1-t0:.4f}s (={perm:.0f}) | scipy Hungarian {t2-t1:.5f}s (={np.array(C)[r,c].sum():.0f})")

# Q3: the ONE place assignment IS poly via min-plus: successive shortest path
# (Bellman-Ford = iterated meet relaxation) on the bipartite residual graph.
# Show that the inner SP loop (genuine iterated meet) drives Hungarian and
# recovers the optimum.
print()
print("Q3: successive-shortest-path assignment (inner SP = iterated meet relax):")
def ssp_assignment(C):
    n=len(C)
    INF=float('inf')
    u=[0.0]*(n+1); v=[0.0]*(n+1); p=[0]*(n+1); way=[0]*(n+1)
    for i in range(1,n+1):
        p[0]=i; j0=0
        minv=[INF]*(n+1); used=[False]*(n+1)
        while True:
            used[j0]=True; i0=p[j0]; delta=INF; j1=-1
            for j in range(1,n+1):
                if not used[j]:
                    cur=C[i0-1][j-1]-u[i0]-v[j]   # reduced cost = min-plus relax
                    if cur<minv[j]: minv[j]=cur; way[j]=j0
                    if minv[j]<delta: delta=minv[j]; j1=j
            for j in range(n+1):
                if used[j]: u[p[j]]+=delta; v[j]-=delta
                else: minv[j]-=delta
            j0=j1
            if p[j0]==0: break
        while j0:
            j1=way[j0]; p[j0]=p[j1]; j0=j1
    ans=0.0
    for j in range(1,n+1):
        ans+=C[p[j]-1][j-1]
    return ans
ok=0
for t in range(200):
    n=6; C=rng.integers(1,40,size=(n,n)).astype(float).tolist()
    val=ssp_assignment(C)
    r,c=linear_sum_assignment(np.array(C)); gt=np.array(C)[r,c].sum()
    if abs(val-gt)<1e-9: ok+=1
print(f"   SSP (poly, inner = min-plus/meet relaxation) == scipy: {ok}/200")
