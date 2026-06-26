"""Final consolidated check: D4 factor count fix + tropical-meet consistency."""
import numpy as np
from collections import Counter
VECTORS=[("v1","VA",0,0,0),("v2","VA",0,0,1),("v3","VA",1,0,0),("v4","VA",1,0,1),
         ("v5","VB",0,0,0),("v6","VB",0,0,1),("v7","VB",1,0,0),("v8","VB",1,0,1)]
def wm(v):
    _,fam,fx,fy,fz=v; M=np.eye(3,dtype=int)
    if fam=="VB": M=np.array([[0,1,0],[1,0,0],[0,0,1]])@M
    return np.diag([-1 if fx else 1,-1 if fy else 1,-1 if fz else 1])@M
mats=[wm(v) for v in VECTORS]
def mk(M):return tuple(M.flatten().tolist())
I=np.eye(3,dtype=int); G={mk(I):I}
for M in mats:G[mk(M)]=M
ch=True
while ch:
    ch=False
    for A in list(G.values()):
        for B in list(G.values()):
            P=A@B
            if mk(P) not in G:G[mk(P)]=P;ch=True
G=list(G.values())
# D4 factor = elements with z-block +1
zflip=np.diag([1,1,-1])
d4=[M for M in G if M[2,2]==1]
print("D4 factor (z-block=+1) order =",len(d4))
def eo(M):
    P=np.eye(3,dtype=int);o=0
    while True:
        P=P@M;o+=1
        if (P==np.eye(3,dtype=int)).all():return o
        if o>16:return -1
print("D4 factor order-profile =",dict(sorted(Counter(eo(M) for M in d4).items())),"(D4=1,5,2)")
print("z-sign central in G:",all((zflip@M==M@zflip).all() for M in G))
print("G order:",len(G),"=> G = D4 x Z2 = SmallGroup(16,11)")

# Tropical-meet consistency: meet=(sum,min). sorted prefix-sum vs min-plus
# The 'meet' (a+p,a,a+p+n) with min=a is the (sum,min) tropical pair; iterate => Floyd-Warshall.
# Confirm the min-plus semiring closure equals all-pairs shortest path (already in memory, re-verify tiny).
W=np.array([[0,3,np.inf,7],[8,0,2,np.inf],[5,np.inf,0,1],[2,np.inf,np.inf,0]])
D=W.copy()
n=4
for k in range(n):
    for i in range(n):
        for j in range(n):
            D[i,j]=min(D[i,j],D[i,k]+D[k,j])
# min-plus matrix power
def mp(A,B):
    n=len(A);C=np.full((n,n),np.inf)
    for i in range(n):
        for j in range(n):
            C[i,j]=min(A[i,k]+B[k,j] for k in range(n))
    return C
P=W.copy()
for _ in range(n-1):P=mp(P,W)
print("min-plus (sum,min) closure == Floyd-Warshall:",np.allclose(np.minimum(P,D),D) and np.array_equal(P,D))
