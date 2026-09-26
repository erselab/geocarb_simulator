"""Joint-state Jacobian vs finite difference for the aerosol rows, with a finite-difference step sweep (2026-09-26).

Per-anchor Jacobians (check_aerosol_jacobians.py) are exact, yet at the last iterate of the 4-band solve the JOINT height column was
51% off. This builds the joint problem for ONE band (fast) and compares analytic columns with central FD at several step sizes, at
the prior state x0 and at x0 with the aerosol rows moved toward the solve's retrieved values (height +3000 Pa, amplitude x1.5),
to separate a systematic mismatch (same at every step) from finite-difference noise / nonsmoothness (changes with the step).
    python scripts/check_joint_jacobian_steps.py --fpa 0
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gd_multiband_window as gmw  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fpa", type=int, default=0)
ap.add_argument("--rows", default="378-400")
ap.add_argument("--anchor-workers", type=int, default=16)
a = ap.parse_args()
os.environ["GEOCARB_AEROSOL_TYPE"] = "smoke"
lo, hi = [int(t) for t in a.rows.split("-")]
FREE = "co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo,amplitude_aerosol,height_aerosol".split(",")


def hook(P):
    mb, fwd, lin = P["mb"], P["forward"], P["linearize"]
    spec = mb.joint
    x0 = spec.x0()
    sl = spec.slices()
    dxs = spec.dx_scale()
    albn = [k for k in sl if k.startswith("albedo")][0]
    for tag, mod in (("prior state x0", lambda x: x),
                     ("aerosol moved (height +3000 Pa, amp x1.5)", None)):
        x = x0.copy()
        if mod is None:
            hs, as_ = sl["height_aerosol"], sl["amplitude_aerosol"]
            x[hs] = x0[hs] + 3000.0
            x[as_] = x0[as_] * 1.5
        t0 = time.time()
        y, K, _ = lin(x)
        print(f"\n== {tag}: linearization {time.time() - t0:.0f} s", flush=True)
        for nm, off in (("height_aerosol", 11), ("amplitude_aerosol", 11), ("p_surface_hpa", 11), ("albedo", 70)):
            key = nm if nm != "albedo" else albn
            k = sl[key].start + min(off, sl[key].stop - sl[key].start - 1)
            an = K[:, k]
            for rel in (1e-4, 1e-3, 1e-2, 1e-1):
                h = rel * dxs[k]
                xp, xm = x.copy(), x.copy()
                xp[k] += h
                xm[k] -= h
                fd = (fwd(xp) - fwd(xm)) / (2 * h)
                cos = float(np.dot(an, fd) / (np.linalg.norm(an) * np.linalg.norm(fd) + 1e-300))
                err = float(np.linalg.norm(an - fd) / (np.linalg.norm(fd) + 1e-300))
                print(f"   {key:18s} col {k:4d} step {rel:.0e}*sigma: cosine {cos:.6f}  rel err {err:.2e}", flush=True)
    return True


gmw.solve_window_multiband({a.fpa: (lo, hi)}, FREE, g_ratio=1.0, anchor_mode="cover", anchor_workers=a.anchor_workers, hook=hook,
                           verbose=False, aerosol=True, prior_fields="realistic", solver="xrtm")
