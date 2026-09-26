"""Why does the 4-band aerosol solve need extra damping / crawl along aerosol height?  (2026-09-26)

At the last iterate of the finished tile-12 4-band aerosol solve (results/.../check_aero4_tile12_analytic_psurf.pkl, still
moving 0.5 sigma per iteration), this measures three things:
  1. CONDITIONING: eigen-decomposition of the Gauss-Newton matrix A = K^T Sy^-1 K + Sa^-1 in units of each element's own
     prior sigma -- the weakest directions and which state rows they are made of (is there a flat height/amplitude/p/CO2 valley?);
  2. JACOBIAN CONSISTENCY: analytic K columns (height, amplitude, p_surface, CO2, an albedo, T) against central finite
     differences of the joint forward model -- a wrong column would by itself cause rejected steps;
  3. NONLINEARITY: the objective J along the damped GN step (lam = 1e-3, the trial that keeps being rejected) at several step
     fractions vs the quadratic model J0 - 2 s b.dx + s^2 dx.A.dx, and along the pure height direction.
    python scripts/diagnose_aerosol_height.py --tile 12
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gd_multiband_window as gmw  # noqa: E402
import gd_joint_block_retrieve as gjr  # noqa: E402
from geocarb_gert.multiband_geometry import build_window_tiles_multiband  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tile", type=int, default=12)
ap.add_argument("--state", default="results/realistic_prior/multiband/check_aero4_tile12_analytic_psurf.pkl")
ap.add_argument("--anchor-workers", type=int, default=16)
a = ap.parse_args()
os.environ["GEOCARB_AEROSOL_TYPE"] = "smoke"
fpas = [0, 1, 2, 3]
rows = dict(build_window_tiles_multiband(fpas, gjr.MIN_WINDOW, 1.0, 2)[a.tile].rows)
FREE = "co2_ppm,ch4_ppb,co_ppb,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo,amplitude_aerosol,height_aerosol".split(",")
x_f = np.asarray(pickle.load(open(a.state, "rb"))["x"], dtype=float)


def hook(P):
    mb, fwd, lin, y_true, Sy_inv = P["mb"], P["forward"], P["linearize"], P["y"], P["Sy_inv"]
    spec = mb.joint
    x_a = spec.x0()
    n = x_a.size
    assert x_f.size == n, (x_f.size, n)
    sl = spec.slices()
    dxs = spec.dx_scale()
    Sa_inv = spec.Sa_inv()
    W = np.asarray(Sy_inv, dtype=float)
    names = {nm: s for nm, s in sl.items()}
    print(f"\ntile {a.tile}: n_free {n}, rows: " + ", ".join(f"{k}[{v.stop - v.start}]" for k, v in names.items()), flush=True)

    def J_of(x, y):
        r = y_true - y
        return float(np.sum(W * r ** 2) + (x - x_a) @ Sa_inv @ (x - x_a))

    t0 = time.time()
    y0, K, _ = lin(x_f)
    print(f"linearization at the last iterate: {time.time() - t0:.0f} s", flush=True)
    r0 = y_true - y0
    J0 = J_of(x_f, y0)
    KtW = K.T * W[None, :]
    A = KtW @ K + Sa_inv
    b = KtW @ r0 - Sa_inv @ (x_f - x_a)
    print(f"J0 = {J0:.2f}  (data term {float(np.sum(W * r0 ** 2)):.2f}, prior term {float((x_f - x_a) @ Sa_inv @ (x_f - x_a)):.2f}); |b/sigma| = {np.linalg.norm(b * dxs):.3e}", flush=True)

    # ---- 1. conditioning in sigma units -------------------------------------------------------------
    D = np.diag(dxs)
    As = D @ A @ D                                   # A in units where each element is one prior sigma
    w, V = np.linalg.eigh(As)
    print(f"\n[1] eigenvalues of A in sigma units: min {w[0]:.3e}, max {w[-1]:.3e}, condition {w[-1] / max(w[0], 1e-300):.2e}", flush=True)
    for i in range(6):
        v = V[:, i]
        share = {nm: float(np.sum(v[s] ** 2)) for nm, s in names.items()}
        top = sorted(share.items(), key=lambda t: -t[1])[:3]
        print(f"   eig {i}: {w[i]:.3e}  made of " + ", ".join(f"{k} {100 * s:.0f}%" for k, s in top), flush=True)
    # the Newton step lives mostly along weak eigen-directions: how much of b (in sigma units) sits in each
    bs = D @ b
    proj = V.T @ bs
    print("   |projection of b| on the 6 weakest directions: " + " ".join(f"{abs(proj[i]):.2e}" for i in range(6)), flush=True)

    # ---- 2. Jacobian consistency ----------------------------------------------------------------
    print("\n[2] analytic K column vs central finite difference of the forward model (relative step 1e-3 of the row's dx scale):", flush=True)
    picks = [("height_aerosol", 11), ("amplitude_aerosol", 11), ("p_surface_hpa", 11), ("co2_ppm", 11), ("t_offset_k", 11),
             ("albedo_O2_A", 70), ("albedo_CO2_strong", 70)]
    for nm, off in picks:
        if nm not in names:
            continue
        k = names[nm].start + min(off, names[nm].stop - names[nm].start - 1)
        h = 1e-3 * dxs[k]
        xp, xm = x_f.copy(), x_f.copy()
        xp[k] += h
        xm[k] -= h
        fd = (fwd(xp) - fwd(xm)) / (2 * h)
        an = K[:, k]
        cos = float(np.dot(an, fd) / (np.linalg.norm(an) * np.linalg.norm(fd) + 1e-300))
        rel = float(np.linalg.norm(an - fd) / (np.linalg.norm(fd) + 1e-300))
        print(f"   {nm:18s} col {k:4d}: cosine {cos:.6f}  rel L2 err {rel:.2e}  |K col|*sigma {np.linalg.norm(an) * dxs[k]:.3e}", flush=True)

    # ---- 3. nonlinearity along the damped GN step and along pure height ---------------------------------
    diagA = np.clip(np.diag(A), 1e-12, None)
    print("\n[3] objective along the damped GN step (lam 1e-3 = the trial that keeps being rejected) vs the quadratic model:", flush=True)
    for lam in (1e-3, 1e-2):
        dx = np.linalg.solve(A + lam * np.diag(diagA), b)
        pred_full = 2 * b @ dx - dx @ A @ dx
        hpart = np.linalg.norm(dx[names["height_aerosol"]] / dxs[names["height_aerosol"]])
        apart = np.linalg.norm(dx[names["amplitude_aerosol"]] / dxs[names["amplitude_aerosol"]])
        print(f"   lam {lam:g}: |dx/sigma| {np.linalg.norm(dx / dxs):.3f}; height part {hpart:.3f}, "
              f"amplitude part {apart:.3f}, quad-model reduction at s=1: {pred_full:.2f}", flush=True)
        for s in (0.25, 0.5, 1.0):
            xt = spec.clip_trial(x_f + s * dx)
            Jt = J_of(xt, fwd(xt))
            pred = 2 * s * (b @ dx) - s * s * (dx @ A @ dx)
            print(f"      s={s:4.2f}: actual dJ = {J0 - Jt:9.3f}   quadratic-model dJ = {pred:9.3f}", flush=True)
    hs = names["height_aerosol"]
    print("   pure height direction (all height elements shifted together):", flush=True)
    for delta in (-4000.0, -1000.0, 1000.0, 4000.0):
        xt = x_f.copy()
        xt[hs] += delta
        Jt = J_of(xt, fwd(xt))
        v = np.zeros(n)
        v[hs] = delta
        pred = 2 * (b @ v) - v @ A @ v
        print(f"      height {delta:+7.0f} Pa: actual dJ = {J0 - Jt:9.3f}   quadratic-model dJ = {pred:9.3f}", flush=True)
    return True


gmw.solve_window_multiband(rows, FREE, g_ratio=1.0, anchor_mode="cover", anchor_workers=a.anchor_workers, hook=hook,
                           verbose=False, aerosol=True, prior_fields="realistic", solver="xrtm")
