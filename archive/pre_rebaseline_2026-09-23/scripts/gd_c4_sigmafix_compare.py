"""c4 (co2,p,h2o,T,albedo free, FPA2, no aerosol) three ways on the same 10 windows:
  old single_scatter  -- pre-2026-09-16 sigma fix (T-offset sigma 0.1 K)
  single_scatter + fix -- the `_sigmafix` re-run under current code (T sigma 5 K)
  xrtm + fix          -- the `_xrtm` run (T sigma 5 K)
Tests whether the strong CO2/p/T/h2o error mode of the old c4 came from the
T sigma, not from the solver.
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gd_frozen_var_compare as fv  # noqa: E402

TAG = "co2-p-h2o-t-albedo"
VARIANTS = {"old single_scatter (T sigma 0.1K)": "",
            "single_scatter, sigma fix": "_sigmafix",
            "xrtm, sigma fix": "_xrtm"}


def load(suffix):
    s = fv.STEM.format(tag=TAG) + suffix
    with open(fv.RES / s / f"{s}.pkl", "rb") as f:
        return pickle.load(f)


def main():
    data = {n: load(s) for n, s in VARIANTS.items()}
    tiles = sorted(set.intersection(*[set(d["results"]) for d in data.values()]))
    print(f"{len(tiles)} shared windows\n")
    for n, d in data.items():
        P = d["results"][tiles[0]]["hires"]["params"]
        print(f"{n}: T sigma={P['t_offset_k']['sigma']}, solver={d.get('solver', 'single_scatter')}")
    rows = {n: fv.bin_errors(d, tiles) for n, d in data.items()}
    alb = np.array([r[2] for r in rows[next(iter(rows))]])
    print()
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
    for k, (n, r) in enumerate(rows.items()):
        E = {g: np.array([b[3][g] for b in r]) for g in fv.GASES}
        M = np.stack([E[g] for g in fv.GASES], 1)
        var, v = fv.pca(M)
        C = np.corrcoef(M.T)
        print(f"=== {n}")
        print("  rms: " + "  ".join(f"{fv.SHORT[g]}={np.sqrt(np.mean(E[g] ** 2)):.3g}" for g in fv.GASES))
        print("  bias: " + "  ".join(f"{fv.SHORT[g]}={E[g].mean():+.3g}" for g in fv.GASES))
        print(f"  PCA variance {np.round(var, 3)}; mode-1 " + ", ".join(f"{fv.SHORT[g]}={c:+.2f}" for g, c in zip(fv.GASES, v)))
        print("  corr: " + "; ".join(f"{fv.SHORT[fv.GASES[i]]}-{fv.SHORT[fv.GASES[j]]}={C[i, j]:+.2f}"
                                   for i in range(4) for j in range(i + 1, 4)))
        m = alb > 0.08
        var2, v2 = fv.pca(M[m])
        print(f"  albedo>0.08 PCA variance {np.round(var2, 3)}; mode-1 " + ", ".join(f"{fv.SHORT[g]}={c:+.2f}" for g, c in zip(fv.GASES, v2)) + "\n")
        if k < 3:
            sc = ax[k].scatter(E["co2_ppm"], E["p_surface_hpa"], c=alb, s=14, cmap="viridis", vmin=0, vmax=0.5)
            ax[k].set(title=n, xlabel="CO2 error [ppm]", ylabel="p error [hPa]")
            ax[k].axhline(0, c="k", lw=.4); ax[k].axvline(0, c="k", lw=.4)
    fig.colorbar(sc, ax=ax[:3], label="albedo", shrink=.8)
    w = 0.27
    for k, (n, r) in enumerate(rows.items()):
        rms = [np.sqrt(np.mean(np.array([b[3][g] for b in r]) ** 2)) for g in fv.GASES[:2]]
        ax[3].bar(np.arange(2) + k * w, rms, w, label=n)
    ax[3].set_xticks(np.arange(2) + w); ax[3].set_xticklabels(["CO2 [ppm]", "p [hPa]"])
    ax[3].legend(fontsize=7); ax[3].set_title("rms error")
    out = fv.REPO / "plots/c4_sigmafix_solver_compare.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
