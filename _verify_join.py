import numpy as np, random, time, networkx as nx

# Build a random weighted graph; store each edge as a meet address (sum,min,total)
# meet3(u,v,w) with w=weight: address = (top2sum, median, total) of the 3-multiset {u,v,w}? 
# Per the dossier the edge is stored as meet3 of (u, v, weight). Test the red-team claim:
# every address inverts EXACTLY to (u,v,weight), and dict+FW gives identical shortest paths.

def meet3(t):
    a,b,c = sorted(t)
    return (b+c, b, a+b+c)   # (top2sum, median, total)
def inv3(X,Y,Z):
    a=Z-X; b=Y; c=X-Y
    return tuple(sorted((a,b,c)))

random.seed(7)
n=60
G=nx.DiGraph()
edges={}
addr_index={}
for u in range(n):
    for v in range(n):
        if u!=v and random.random()<0.08:
            w=random.randint(1,20)
            G.add_edge(u,v,weight=w)
            edges[(u,v)]=w
            A=meet3((u,v,w))
            addr_index[(u,v)]=A

# CLAIM: invert every address back to {u,v,w}
bad=0
for (u,v),A in addr_index.items():
    rec=inv3(*A)
    if rec!=tuple(sorted((u,v,edges[(u,v)]))): bad+=1
print(f"addresses that invert EXACTLY to (u,v,weight): {len(addr_index)-bad}/{len(addr_index)}  (bad={bad})")
print("--> 'edge-free' index is a relabel of the edge list:", bad==0)

# Shortest paths: dict-served FW vs networkx
t0=time.perf_counter()
nx_sp = dict(nx.all_pairs_dijkstra_path_length(G))
t_nx=time.perf_counter()-t0

# tropical closure on weights recovered from addresses (no edge dict)
W = np.full((n,n), np.inf)
for i in range(n): W[i,i]=0
for (u,v),A in addr_index.items():
    a,b,c = inv3(*A)
    # recover weight: weight was one of the 3; but which? red-team's honest point:
    # you must KNOW which member is the weight. Here u,v known as the key -> weight = the leftover.
    members=[a,b,c]
    for m in [u,v]:
        if m in members: members.remove(m)
    w = members[0] if members else None
    W[u,v]=w
t0=time.perf_counter()
D=W.copy()
for k in range(n):
    D=np.minimum(D, D[:,k:k+1]+D[k:k+1,:])
t_trop=time.perf_counter()-t0

# compare
dis=0; cmp=0
for i in range(n):
    for j in range(n):
        if i==j: continue
        nxv = nx_sp[i].get(j, np.inf)
        tv = D[i,j]
        cmp+=1
        if not (np.isinf(nxv) and np.isinf(tv)) and abs(nxv-tv)>1e-9: dis+=1
print(f"tropical-vs-nx disagreements: {dis}/{cmp}")
print(f"timing: networkx={t_nx*1000:.1f}ms  tropical-closure={t_trop*1000:.1f}ms")
print("NOTE: weight recovery required knowing (u,v) to subtract -> address alone does NOT name which member is the weight")
