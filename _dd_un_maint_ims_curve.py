import sys
sys.path.insert(0, r"C:/Users/wynos/aethos_master/src")
import numpy as np
from pathlib import Path
from aethos_master.ims.bearings import load_ims_test
snaps = load_ims_test(Path(r"C:/Users/wynos/OneDrive/New folder/2nd_test"))
n=len(snaps)
rms0=np.array([s['channels'][0]['rms'] for s in snaps])
nh=max(10,int(n*0.2))
base=np.median(rms0[:nh]); mu=rms0[:nh].mean(); sd=rms0[:nh].std()
print(f"n={n}  healthy RMS median={base:.4f} mean={mu:.4f} std={sd:.5f}")
# RMS ratio to healthy at key snapshots
for idx in [196,400,500,530,532,540,545,600,700,800,900,983]:
    print(f"  snap {idx:>3}: RMS={rms0[idx]:.4f}  ratio={rms0[idx]/base:.2f}x  z={(rms0[idx]-mu)/sd:.1f}")
# When does RMS first PERMANENTLY (>=10 consec) exceed +3sigma after baseline?
z=(rms0-mu)/sd
# how 'monotone' is degradation: snapshot of global RMS min after baseline and the climb
print(f"max RMS reached: {rms0.max():.3f} ({rms0.max()/base:.0f}x healthy) at snap {rms0.argmax()}")
