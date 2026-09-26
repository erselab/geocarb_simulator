"""Where does one Gauss-Newton iteration's time go for a multi-band tile?  (2026-09-21)

Builds the joint problem for one tile (via gd_multiband_window's `hook`), then times ONE
forward evaluation (an LM trial costs one of these) and ONE linearization (the Jacobian; one
per iteration), splitting each into its parts with parent-side wall-clock timers:

  forward   : render_at_anchors (anchor RT in a pool + the detector operator) -- split into
              the operator (gd_render.predict_neighborhood, run in the parent) and the rest
              (anchor RT + pool overhead)
  linearize : LinearizePool.run_anchor (anchor RT + per-layer derivatives, in a pool),
              LinearizePool.run_L (the detector operator applied to EVERY state column, in a
              pool) and everything else (assembly)
Also reports the sizes that drive them (anchors, wavelengths, state columns, shared-memory bytes).

  python profile_multiband_iteration.py --fpas 0,2 --tile 12 --free ... --anchor-workers 8
"""
import argparse
import collections
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gd_multiband_window as gmw  # noqa: E402
import gd_joint_block_retrieve as gjr  # noqa: E402
from geocarb_gert import gd_render, jacobians as jac, joint_state as js  # noqa: E402
from geocarb_gert.multiband_geometry import build_window_tiles_multiband  # noqa: E402

T = collections.defaultdict(float)
N = collections.Counter()


def timed(obj, name, key):
    orig = getattr(obj, name)

    def wrapper(*a, **k):
        t0 = time.time()
        try:
            return orig(*a, **k)
        finally:
            T[key] += time.time() - t0
            N[key] += 1
    setattr(obj, name, wrapper)


def reset():
    T.clear()
    N.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpas", required=True)
    ap.add_argument("--tile", type=int, required=True)
    ap.add_argument("--free", required=True)
    ap.add_argument("--anchor-workers", type=int, default=8)
    ap.add_argument("--aerosol-type", default=None,
                    help="profile the aerosol solve (XRTM; pass amplitude_aerosol,height_aerosol in --free) with this type")
    a = ap.parse_args()
    if a.aerosol_type:
        os.environ["GEOCARB_AEROSOL_TYPE"] = a.aerosol_type
    fpas = [int(t) for t in a.fpas.split(",")]
    rows = dict(build_window_tiles_multiband(fpas, gjr.MIN_WINDOW, 1.0, 2)[a.tile].rows)

    timed(js, "render_at_anchors", "fwd.render_at_anchors")
    timed(gd_render, "predict_neighborhood", "parent.predict_neighborhood")
    timed(jac.LinearizePool, "run_anchor", "lin.run_anchor(RT+derivs)")
    timed(jac.LinearizePool, "run_L", "lin.run_L(operator x columns)")
    timed(jac, "linearize", "lin.total")
    t_start = time.time()

    def hook(P):
        t_setup = time.time() - t_start
        mb, fwd, lin = P["mb"], P["forward"], P["linearize"]
        x0 = mb.joint.x0()
        n_anchor, n_free = len(P["anchor_etas"]), x0.size
        n_hires = {f: len(P["B"][f]["wn_hires"]) for f in P["fpas"]}
        print(f"\ntile {a.tile} rows {rows}: {n_anchor} anchors, {n_free} state columns, hi-res points per band {n_hires}, "
              f"data {P['n_data_by_band']}, workers {a.anchor_workers}", flush=True)
        print(f"setup (band setup + truth forwards + noise) before the hook: {t_setup:.0f} s", flush=True)

        reset()
        t0 = time.time(); fwd(x0); t_fwd = time.time() - t0
        f = dict(T)
        print(f"\nONE FORWARD (both bands): {t_fwd:.1f} s", flush=True)
        for k, v in f.items():
            print(f"   {k:34s} {v:8.1f} s  ({100 * v / t_fwd:4.1f}%)", flush=True)
        oper = f.get("parent.predict_neighborhood", 0.0)
        print(f"   => anchor RT + pool overhead     {t_fwd - oper:8.1f} s  ({100 * (t_fwd - oper) / t_fwd:4.1f}%)   "
              f"detector operator {oper:.1f} s ({100 * oper / t_fwd:.1f}%)", flush=True)

        reset()
        t0 = time.time(); y, K, _ = lin(x0); t_lin = time.time() - t0
        l = dict(T)
        print(f"\nONE LINEARIZATION (both bands): {t_lin:.1f} s   (K {K.shape}, {K.nbytes / 1e9:.2f} GB)", flush=True)
        for k, v in l.items():
            print(f"   {k:34s} {v:8.1f} s  ({100 * v / t_lin:4.1f}%)  calls {N[k]}", flush=True)
        ra, rl = l.get("lin.run_anchor(RT+derivs)", 0.0), l.get("lin.run_L(operator x columns)", 0.0)
        print(f"   => RT+derivs {100 * ra / t_lin:.1f}%, operator over columns {100 * rl / t_lin:.1f}%, "
              f"other (assembly, forward y=L(S), field building) {100 * (t_lin - ra - rl) / t_lin:.1f}%", flush=True)
        per_iter = t_lin + 1.5 * t_fwd
        print(f"\nAn LM iteration ~= 1 linearization + ~1-3 trial forwards ~= {t_lin + t_fwd:.0f}-{t_lin + 3 * t_fwd:.0f} s here.", flush=True)
        return True

    gmw.solve_window_multiband(rows, a.free.split(","), g_ratio=1.0, anchor_mode="cover",
                               anchor_workers=a.anchor_workers, hook=hook, verbose=False,
                               aerosol=bool(a.aerosol_type), prior_fields="realistic",
                               solver="xrtm" if a.aerosol_type else "single_scatter")


if __name__ == "__main__":
    main()
