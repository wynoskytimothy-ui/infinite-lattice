"""
Fair-baseline audit for aethos_master CMAPSS RUL (FD001).

AETHOS report: ridge_ensemble MAE 12.596 / RMSE 16.545 (just reproduced).

Question: is the lattice/geometric feature pipeline + SensorBrain adding value
over PLAIN baselines on the SAME train/test split, same RUL cap (125)?

Baselines (all sklearn, fair = same data, same target):
  B0  last-cycle raw 14 informative sensors -> Ridge
  B1  last-cycle raw 14 sensors -> RandomForest
  B2  AETHOS's own per-sensor window features (mean/std/slope/last/range/ewmean
        over windows 30,50) -> Ridge  (== reimplement aethos features, sklearn ridge)
  B3  same features -> RandomForest
  B4  same features -> sklearn Ridge w/ standardize (mirror of aethos pipeline)

Also run aethos run_cmapss and run_cmapss_trained for head-to-head.
"""
import sys, math, time
sys.path.insert(0, r"C:/Users/wynos/aethos_master/src")
import numpy as np
from pathlib import Path

from aethos_master.cmapss.rul import (
    load_cmapss, select_informative_sensors, SensorNorm,
    extract_per_sensor_features, _cross_features,
    run_cmapss, run_cmapss_trained,
)
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DATA = Path(r"C:/Users/wynos/OneDrive/New folder/cmapss_data")
RUL_CAP = 125
WINDOWS = (30, 50)
LAMBDA = 0.05

def metrics(pred, gt):
    pred = np.asarray(pred, float); gt = np.asarray(gt, float)
    mae = float(np.mean(np.abs(pred - gt)))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    return mae, rmse

def main():
    train, test, rul_gt = load_cmapss(DATA, "FD001")
    sensor_ids = select_informative_sensors(train, 1e-6)
    norm = SensorNorm(); norm.fit(train, sensor_ids)
    print(f"FD001: {len(train)} train engines, {len(test)} test, {len(sensor_ids)} informative sensors")
    rul_gt = np.array(rul_gt[:len(test)], float)

    # ---- training-sample builder (mirrors aethos run_cmapss sampling) ----
    def build_xy(traj_dict, with_cross):
        X, y = [], []
        for u in sorted(traj_dict):
            traj = traj_dict[u]
            max_cyc = max(c for c, _, _ in traj)
            step = max(1, (len(traj) - max(WINDOWS)) // 40)
            for t in range(max(WINDOWS), len(traj), step):
                rul = min(max_cyc - traj[t][0], RUL_CAP)
                feats = []
                for w in WINDOWS:
                    start = max(0, t - w + 1)
                    sub = traj[start:t + 1]
                    wd = [norm.transform(s, sensor_ids) for _, s, _ in sub]
                    f = extract_per_sensor_features(wd, LAMBDA)
                    if with_cross:
                        f = f + _cross_features(f, 8)
                    feats.extend(f)
                X.append(feats); y.append(float(rul))
        return np.array(X), np.array(y)

    def test_feats(with_cross):
        X = []
        for u in sorted(test):
            traj = test[u]; end = len(traj) - 1
            feats = []
            for w in WINDOWS:
                start = max(0, end - w + 1)
                sub = traj[start:end + 1]
                wd = [norm.transform(s, sensor_ids) for _, s, _ in sub]
                f = extract_per_sensor_features(wd, LAMBDA)
                if with_cross:
                    f = f + _cross_features(f, 8)
                feats.extend(f)
            X.append(feats)
        return np.array(X)

    # ---- last-cycle raw sensors (no windows) ----
    def last_raw_xy():
        X, y = [], []
        for u in sorted(train):
            traj = train[u]
            max_cyc = max(c for c, _, _ in traj)
            step = max(1, (len(traj) - max(WINDOWS)) // 40)
            for t in range(max(WINDOWS), len(traj), step):
                rul = min(max_cyc - traj[t][0], RUL_CAP)
                X.append([traj[t][1][s] for s in sensor_ids]); y.append(float(rul))
        Xt = []
        for u in sorted(test):
            traj = test[u]; end = len(traj) - 1
            Xt.append([traj[end][1][s] for s in sensor_ids])
        return np.array(X), np.array(y), np.array(Xt)

    rows = []

    # ---- AETHOS engines ----
    t0 = time.perf_counter()
    r = run_cmapss(DATA, "FD001", RUL_CAP, WINDOWS, LAMBDA, alpha=10.0, use_cross=True)
    rows.append(("AETHOS ridge_ensemble (pure-py)", r.mae, r.rmse, time.perf_counter()-t0))

    t0 = time.perf_counter()
    rb = run_cmapss_trained(DATA, "FD001", RUL_CAP, WINDOWS, LAMBDA, alpha=10.0, use_cross=True)
    rows.append(("AETHOS brain_ridge (SensorBrain)", rb.mae, rb.rmse, time.perf_counter()-t0))

    # ---- B0/B1 last-cycle raw ----
    Xtr, ytr, Xte = last_raw_xy()
    t0=time.perf_counter()
    m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xtr, ytr)
    p = np.clip(m.predict(Xte), 0, None); mae, rmse = metrics(p, rul_gt)
    rows.append(("B0 last-cycle raw -> Ridge", mae, rmse, time.perf_counter()-t0))
    t0=time.perf_counter()
    m = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    p = np.clip(m.predict(Xte), 0, None); mae, rmse = metrics(p, rul_gt)
    rows.append(("B1 last-cycle raw -> RandomForest", mae, rmse, time.perf_counter()-t0))

    # ---- B2..B4 on aethos window features (no cross) ----
    Xtr, ytr = build_xy(train, with_cross=False); Xte = test_feats(with_cross=False)
    t0=time.perf_counter()
    m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xtr, ytr)
    p = np.clip(m.predict(Xte), 0, None); mae, rmse = metrics(p, rul_gt)
    rows.append(("B2 aethos-feats(no cross) -> sklearn Ridge", mae, rmse, time.perf_counter()-t0))
    t0=time.perf_counter()
    m = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    p = np.clip(m.predict(Xte), 0, None); mae, rmse = metrics(p, rul_gt)
    rows.append(("B3 aethos-feats(no cross) -> RandomForest", mae, rmse, time.perf_counter()-t0))

    # ---- B5 on aethos window features WITH cross (== aethos's own X) ----
    Xtr, ytr = build_xy(train, with_cross=True); Xte = test_feats(with_cross=True)
    t0=time.perf_counter()
    m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xtr, ytr)
    p = np.clip(m.predict(Xte), 0, None); mae, rmse = metrics(p, rul_gt)
    rows.append(("B5 aethos-feats(+cross) -> sklearn Ridge", mae, rmse, time.perf_counter()-t0))
    t0=time.perf_counter()
    m = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    p = np.clip(m.predict(Xte), 0, None); mae, rmse = metrics(p, rul_gt)
    rows.append(("B6 aethos-feats(+cross) -> RandomForest", mae, rmse, time.perf_counter()-t0))

    print(f"\n{'method':<45} {'MAE':>7} {'RMSE':>7} {'sec':>7}")
    print("-"*70)
    for name, mae, rmse, dt in rows:
        print(f"{name:<45} {mae:7.3f} {rmse:7.3f} {dt:7.2f}")

if __name__ == "__main__":
    main()
