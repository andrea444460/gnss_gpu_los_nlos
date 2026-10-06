import csv
from pathlib import Path
import numpy as np

p = Path(r"C:/Users/Andrea/Desktop/Gneova Map/genova_area_points_osm (3).csv")
rows = list(csv.DictReader(p.open(encoding="utf-8")))
vis = np.array([float(r["mean_n_visible"]) for r in rows])
los = np.array([float(r["mean_n_los"]) for r in rows])
nlos = np.array([float(r["mean_n_nlos"]) for r in rows])
tbv = np.array([float(r["mean_n_terrain_blocked_visible"]) for r in rows])
tb = np.array([float(r["mean_n_terrain_blocked"]) for r in rows])
n = len(rows)

# Candidates above elevation mask ~ visible + terrain_blocked_visible
cand = vis + tbv
terrain_frac = np.where(cand > 1e-9, tbv / cand, 0.0)
# Building NLOS among remaining visible
nlos_frac_vis = np.where(vis > 1e-9, nlos / vis, 0.0)

print(f"n_points={n}")
print(f"city mean tbv={tbv.mean():.3f} med={np.median(tbv):.3f} p90={np.percentile(tbv,90):.3f} p95={np.percentile(tbv,95):.3f} p99={np.percentile(tbv,99):.3f} max={tbv.max():.3f}")
print(f"city mean cand_above_mask={cand.mean():.3f} terrain_frac_of_cand={terrain_frac.mean()*100:.2f}%")
print()
print("--- absolute tbv thresholds ---")
for thr in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
    m = tbv > thr
    print(f"tbv>{thr:g}: {m.sum()} pts ({100*m.mean():.2f}%) | mean_tbv={tbv[m].mean() if m.any() else 0:.2f} mean_nlos={nlos[m].mean() if m.any() else 0:.2f} mean_los={los[m].mean() if m.any() else 0:.2f}")

print()
print("--- terrain as primary occlusion vs buildings ---")
# Among elevation-visible candidates, terrain removes more than buildings cause NLOS
# (tbv > nlos): terrain dominates count-wise
dom = tbv > nlos
print(f"tbv > nlos: {dom.sum()} ({100*dom.mean():.2f}%)")
dom1 = tbv > (nlos + 1)
print(f"tbv > nlos+1: {dom1.sum()} ({100*dom1.mean():.2f}%)")
# terrain fraction of elevation-mask candidates
for thr in [0.05, 0.10, 0.20, 0.30]:
    m = terrain_frac >= thr
    print(f"terrain_frac>={thr:.0%}: {m.sum()} ({100*m.mean():.2f}%) mean_tbv={tbv[m].mean() if m.any() else 0:.2f}")

print()
print("--- without DEM: would count extra sats as candidates ---")
# If no DEM, visible_candidates would be cand; with DEM it's vis
extra = tbv  # sats wrongly kept without DEM
# Among those, some would be LOS some NLOS under building RT - we don't have that split
# But impact on "usable sky":
print(f"mean extra sats without DEM (tbv)={extra.mean():.3f}")
print(f"points where without DEM you'd keep >=1 spurious candidate: {(tbv>=1).sum()} ({100*(tbv>=1).mean():.2f}%)")
print(f"points where without DEM you'd keep >=2: {(tbv>=2).sum()} ({100*(tbv>=2).mean():.2f}%)")
print(f"points where without DEM you'd keep >=3: {(tbv>=3).sum()} ({100*(tbv>=3).mean():.2f}%)")

print()
print("--- joint: terrain matters AND buildings do NOT dominate ---")
# high terrain, low building nlos -> open canyon / hillside with terrain walls
m = (tbv >= 1) & (nlos < 5)
print(f"tbv>=1 & nlos<5: {m.sum()} ({100*m.mean():.2f}%) mean_tbv={tbv[m].mean() if m.any() else 0:.2f} mean_los={los[m].mean() if m.any() else 0:.2f}")
m2 = (tbv >= 2) & (nlos < 5)
print(f"tbv>=2 & nlos<5: {m2.sum()} ({100*m2.mean():.2f}%)")
m3 = (terrain_frac >= 0.10) & (nlos_frac_vis < 0.30)
print(f"terrain_frac>=10% & building_nlos_frac<30%: {m3.sum()} ({100*m3.mean():.2f}%)")

print()
print("--- distribution histogram tbv ---")
bins = [0, 0.01, 0.5, 1, 2, 3, 5, 10, 100]
hist, _ = np.histogram(tbv, bins=bins)
for i in range(len(hist)):
    print(f"  [{bins[i]}, {bins[i+1]}): {hist[i]} ({100*hist[i]/n:.2f}%)")
