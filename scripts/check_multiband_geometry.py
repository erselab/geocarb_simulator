"""Phase-2 gate for the multi-band joint block (docs/MULTIBAND_PLAN.md Sec.5):
shared-eta tiling that respects BOTH bands' keystone maxima. Real polynomials only
(keystone always on). Exit status 1 on any failure.

    PYTHONPATH=.:<gert> python scripts/check_multiband_geometry.py
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from geocarb_gert.gd_polynomials import N_PX, rows_crossed  # noqa: E402
from geocarb_gert.joint_state import default_pad_for_psf  # noqa: E402
from geocarb_gert.multiband_geometry import (band_tables, build_window_tiles_multiband,  # noqa: E402
                                              joint_anchor_eta_range, window_eta_extent)
import gd_joint_block_retrieve as gjr  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


FPAS = (0, 2)
SCALE = 1.0
tiles = build_window_tiles_multiband(FPAS, min_window=gjr.MIN_WINDOW, window_scale=SCALE, overlap=0)
print(f"{len(tiles)} windows for FPAs {FPAS}; widths (eta) min/max "
      f"{min(t.width_eta for t in tiles):.4f}/{max(t.width_eta for t in tiles):.4f}")

# G1 -- row partition per band (overlap = 0): ordered, disjoint, gap-free
for f in FPAS:
    rr = [t.rows[f] for t in tiles]
    ok = all(rr[i][1] + 1 == rr[i + 1][0] for i in range(len(rr) - 1))
    check(f"FPA{f}: windows partition its rows (disjoint, contiguous)", ok,
          f"rows {rr[0][0]}..{rr[-1][1]} of 0..{N_PX - 1}")

# G2 -- width >= 2 * scale * 2.2 * (either band's largest crossing inside the window)
sp = {f: band_tables(f)["spacing"] for f in FPAS}
min_r = gjr.MIN_WINDOW * max(sp.values())
bad = []
for i, t in enumerate(tiles[:-1]):            # last window is clipped by the slit end
    need = max(2 * min_r, 2 * SCALE * 2.2 * max(
        float(np.max(rows_crossed(f, np.arange(t.rows[f][0], t.rows[f][1] + 1)))) * sp[f] for f in FPAS))
    if t.width_eta < need * 0.98:
        bad.append((i, round(t.width_eta, 4), round(need, 4)))
check("every window is at least as wide as BOTH bands' largest keystone crossing inside it",
      not bad, str(bad[:4]))

# G3 -- the max rule is never narrower than the centre-only rule; report where it is wider
tc = build_window_tiles_multiband(FPAS, min_window=gjr.MIN_WINDOW, window_scale=SCALE, crossing="center")
wmax = np.mean([t.width_eta for t in tiles])
wcen = np.mean([t.width_eta for t in tc])
check("max-crossing tiling mean window width >= centre-only", wmax >= wcen - 1e-12,
      f"mean width {wmax:.4f} vs {wcen:.4f}; windows {len(tiles)} vs {len(tc)}")

# G4 -- anchors cover every pixel of both bands (keystone included); how often would
#       nominal per-band anchors (rows +/- pad at the centre column) have missed some?
pad = max(gjr.PAD, default_pad_for_psf(1.5))
miss = {f: 0 for f in FPAS}
worst = {f: 0.0 for f in FPAS}
ok_all = True
for t in tiles:
    a_lo, a_hi = joint_anchor_eta_range(t, pad)
    for f in FPAS:
        e_lo, e_hi = window_eta_extent(f, *t.rows[f], pad=pad)
        ok_all &= (a_lo <= e_lo + 1e-12) and (a_hi >= e_hi - 1e-12)
        tb = band_tables(f)
        n_lo, n_hi = tb["eta_c"][max(0, t.rows[f][0] - pad)], tb["eta_c"][min(N_PX - 1, t.rows[f][1] + pad)]
        u_lo, u_hi = window_eta_extent(f, *t.rows[f], pad=0)      # the window's OWN rows only
        short = max(n_lo - u_lo, u_hi - n_hi, 0.0)
        if short > 1e-12:
            miss[f] += 1
            worst[f] = max(worst[f], short / sp[f])
check("joint anchor range covers every pixel (all columns) of both bands, for every window", ok_all)
for f in FPAS:
    print(f"       info: FPA{f}: nominal centre-column anchors (rows +/- {pad}) would MISS part of the "
          f"pixel eta extent of the window's OWN rows in {miss[f]}/{len(tiles)} windows (worst {worst[f]:.1f} rows short)")

# G5 -- one band, crossing="center" reproduces the existing single-band tiling
for f in (0, 2):
    old = gjr.build_window_tiles(f, 0, 1023, gjr.MIN_WINDOW, SCALE, 0)
    new = [t.rows[f] for t in build_window_tiles_multiband((f,), gjr.MIN_WINDOW, SCALE, 0, crossing="center")]
    same_n = len(old) == len(new)
    dmax = max((abs(a[0] - b[0]) for a, b in zip(old, new)), default=0) if same_n else -1
    check(f"FPA{f} alone (centre rule) reproduces the single-band tiling (count, boundaries within 2 rows)",
          same_n and dmax <= 2, f"tiles {len(old)} vs {len(new)}; max start-row diff {dmax}")

print("\nALL CHECKS PASSED" if not FAILS else f"\nFAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
