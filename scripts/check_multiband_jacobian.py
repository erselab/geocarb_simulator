"""Phase-3 gate (b): the STACKED analytic Jacobian of a real multi-band window against
finite differences of the stacked forward model, for a handful of columns covering every
kind of joint-state element (each shared row + BOTH bands' albedo rows).

    ARGS="..." sbatch --export=ALL scripts/submit_multiband_check.sbatch     (or run directly)
    python scripts/check_multiband_jacobian.py --fpas 0,2 --tile 12 --free co2_ppm,p_surface_hpa,...
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gd_multiband_window as gmw  # noqa: E402
from geocarb_gert.multiband_geometry import build_window_tiles_multiband  # noqa: E402
import gd_joint_block_retrieve as gjr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpas", required=True)
    ap.add_argument("--tile", type=int, required=True)
    ap.add_argument("--free", required=True)
    ap.add_argument("--anchor-workers", type=int, default=1)
    ap.add_argument("--g-ratio", type=float, default=1.0)
    ap.add_argument("--solver", default=None, choices=["single_scatter", "xrtm"],
                    help="default: xrtm if --aerosol, else single_scatter (2026-09-23)")
    ap.add_argument("--prior-fields", default="structural")
    ap.add_argument("--aerosol", action="store_true")
    ap.add_argument("--rel-step", type=float, default=1e-3, help="FD step in units of the row's dx scale")
    a = ap.parse_args()
    fpas = [int(t) for t in a.fpas.split(",")]
    tiles = build_window_tiles_multiband(fpas, gjr.MIN_WINDOW, 1.0, 2)
    rows = dict(tiles[a.tile].rows)
    print(f"tile {a.tile}: rows {rows}", flush=True)

    def hook(P):
        mb, fwd, lin = P["mb"], P["forward"], P["linearize"]
        spec = mb.joint
        x0 = spec.x0()
        _, K, _ = lin(x0)
        dxs = spec.dx_scale()
        sl = spec.slices()
        picks = []
        for name, s in sl.items():
            picks.append((name, s.start + (s.stop - s.start) // 2))          # middle element of every free row
        y0 = fwd(x0)
        print(f"joint n_free {x0.size}, n_data {y0.size} ({P['n_data_by_band']}); checking {len(picks)} columns", flush=True)
        worst = 0.0
        splits = np.cumsum([0] + [P["n_data_by_band"][f] for f in P["fpas"]])
        for name, k in picks:
            h = a.rel_step * dxs[k]
            xp, xm = x0.copy(), x0.copy()
            xp[k] += h
            xm[k] -= h
            fd = (fwd(xp) - fwd(xm)) / (2 * h)
            an = K[:, k]
            rel = np.linalg.norm(an - fd) / max(np.linalg.norm(fd), 1e-300)
            per_band = [np.linalg.norm(an[splits[i]:splits[i + 1]] - fd[splits[i]:splits[i + 1]]) /
                        max(np.linalg.norm(fd[splits[i]:splits[i + 1]]), 1e-300) for i in range(len(P["fpas"]))]
            worst = max(worst, rel)
            print(f"  {name:22s} col {k:4d}: |J_an - J_fd|/|J_fd| = {rel:.2e}  per band {[f'{v:.1e}' for v in per_band]}"
                  f"   |J_fd| by band {[f'{np.linalg.norm(fd[splits[i]:splits[i+1]]):.2e}' for i in range(len(P['fpas']))]}", flush=True)
        # albedo columns of the OTHER band must be exactly zero in each band's rows
        for i, b in enumerate(mb.bands):
            j = sl[b.row]
            others = [q for q in range(len(mb.bands)) if q != i]
            blk = np.abs(K[np.ix_(np.concatenate([np.arange(splits[q], splits[q + 1]) for q in others]),
                                  np.arange(j.start, j.stop))]).max()
            print(f"  max |dI/d(albedo of band {b.label})| in the OTHER band's data = {blk:.1e}")
        print("JACOBIAN CHECK", "PASSED" if worst < 5e-3 else "FAILED", f"(worst relative error {worst:.2e})")
        return worst < 5e-3

    ok = gmw.solve_window_multiband(rows, a.free.split(","), g_ratio=a.g_ratio, anchor_mode="cover",
                                    anchor_workers=a.anchor_workers, hook=hook, verbose=False,
                                    solver=(a.solver or ("xrtm" if a.aerosol else "single_scatter")),
                                    prior_fields=a.prior_fields, aerosol=a.aerosol)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
