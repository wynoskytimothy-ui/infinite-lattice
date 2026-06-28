import sys, time
sys.path.insert(0, r"C:/Users/wynos/aethos_master/src")
import numpy as np
from pathlib import Path
from aethos_master.cmapss.rul import (load_cmapss, select_informative_sensors,
    SensorNorm, extract_per_sensor_features)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

DATA = Path(r"C:/Users/wynos/OneDrive/New folder/cmapss_data")
RUL_CAP=125; WINDOWS=(30,50); LAMBDA=0.05
train, test, rul_gt = load_cmapss(DATA,"FD001")
sids = select_informative_sensors(train,1e-6)
norm=SensorNorm(); norm.fit(train,sids)
rul_gt=np.array(rul_gt[:len(test)],float)

def feats(traj,end):
    f=[]
    for w in WINDOWS:
        sub=traj[max(0,end-w+1):end+1]
        wd=[norm.transform(s,sids) for _,s,_ in sub]
        f.extend(extract_per_sensor_features(wd,LAMBDA))
    return f
Xtr,ytr=[],[]
for u in sorted(train):
    tr=train[u]; mc=max(c for c,_,_ in tr); step=max(1,(len(tr)-max(WINDOWS))//40)
    for t in range(max(WINDOWS),len(tr),step):
        Xtr.append(feats(tr,t)); ytr.append(float(min(mc-tr[t][0],RUL_CAP)))
Xte=[feats(test[u],len(test[u])-1) for u in sorted(test)]
Xtr,ytr,Xte=np.array(Xtr),np.array(ytr),np.array(Xte)

maes=[];rmses=[]
for seed in range(3):
    m=RandomForestRegressor(n_estimators=200,random_state=seed,n_jobs=-1).fit(Xtr,ytr)
    p=np.clip(m.predict(Xte),0,None)
    maes.append(np.mean(np.abs(p-rul_gt))); rmses.append(np.sqrt(np.mean((p-rul_gt)**2)))
print(f"RandomForest 5 seeds: MAE {np.mean(maes):.3f}+-{np.std(maes):.3f}  RMSE {np.mean(rmses):.3f}+-{np.std(rmses):.3f}")

m=GradientBoostingRegressor(n_estimators=200,max_depth=3,random_state=0).fit(Xtr,ytr)
p=np.clip(m.predict(Xte),0,None)
print(f"GradientBoosting:     MAE {np.mean(np.abs(p-rul_gt)):.3f}  RMSE {np.sqrt(np.mean((p-rul_gt)**2)):.3f}")
