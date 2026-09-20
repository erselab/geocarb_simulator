"""Compare c4 (free co2,p,h2o,T,albedo) against c4t (T frozen at truth) and
c4h (h2o frozen at truth) on the windows they share -- tests which of T / h2o
drives the correlated CO2/p/T/h2o error mode seen in c4.

Usage: python gd_frozen_var_compare.py   (paths hard-wired to the g1/adens4/ovlp2 FPA2 runs)
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/scratch/scrowel3_lab/geocarb_simulator")
sys.path.insert(0, str(REPO))
from geocarb_gert import along_slit_scene as als  # noqa: E402

RES = REPO / "results/realistic_prior"
STEM = "gd_joint_block_whole_slit_fpa2_gratio1_adens4_free-{tag}_analytic_prior-structural_ovlp2_valb_spos-anchor_fapos-anchor"
# (free-set tag, directory suffix of the SINGLE_SCATTER run). The all-free c4 baseline uses
# the "_sigmafix" RE-RUN (2026-09-19): the original single_scatter c4 predates the
# 2026-09-16 state-vector sigma fix (T-offset sigma 0.1 K then, 5 K now), so it is not
# comparable to c4t/c4h/c4x, which all ran under the fixed code (see gd_c4_sigmafix_compare.py
# and docs/PROJECT_STATUS.md Sec.31).
RUNS = {"c4 (all free)": ("co2-p-h2o-t-albedo", "_sigmafix"),
        "c4t (T frozen)": ("co2-p-h2o-albedo", ""),
        "c4h (h2o frozen)": ("co2-p-t-albedo", "")}
GASES = ["co2_ppm", "p_surface_hpa", "h2o_surface_vmr", "t_offset_k"]
SHORT = {"co2_ppm": "CO2", "p_surface_hpa": "p", "h2o_surface_vmr": "h2o", "t_offset_k": "T"}


def load(tag, solver="single_scatter", suffix=""):
    """solver="xrtm" -> the non-aerosol xrtm runs' directory, which gd_joint_block_retrieve.py
    suffixes with `_xrtm` (added 2026-09-18) so it can't overwrite the single_scatter parts."""
    s = STEM.format(tag=tag) + ("_xrtm" if solver == "xrtm" else suffix)
    with open(RES / s / f"{s}.pkl", "rb") as f:
        return pickle.load(f)


def bin_errors(d, tiles):
    """(window, eta, albedo_mean, {gas: retrieved-truth}) per bin, all bins of each window."""
    out = []
    for t in tiles:
        P = d["results"][t]["hires"]["params"]
        x = np.asarray(P["co2_ppm"]["positions"]) * als.SLIT_HALF_KM
        e = {g: np.asarray(P[g]["values"]) - np.asarray(als.STATE_FIELDS[g](
            np.asarray(P[g]["positions"]) * als.SLIT_HALF_KM)) for g in GASES}
        alb = np.asarray(als.SURFACE_FIELDS["albedo"](x, "CO2_strong"))
        for i in range(len(x)):
            out.append((t, x[i], alb[i], {g: e[g][i] for g in GASES}))
    return out


def pca(E):
    """E: (n, k) errors -> variance fractions, loadings of first mode."""
    Z = E - E.mean(0)
    Z = Z / Z.std(0)
    u, s, vt = np.linalg.svd(Z, full_matrices=False)
    var = s ** 2 / np.sum(s ** 2)
    v = vt[0] * np.sign(vt[0][0] if vt[0][0] != 0 else 1)
    return var, v


def main(solver="single_scatter"):
    data = {n: load(tag, solver, sfx) for n, (tag, sfx) in RUNS.items()}
    tiles = sorted(set.intersection(*[set(d["results"]) for d in data.values()]))
    print(f"######## solver = {solver} ########")
    print(f"{len(tiles)} shared windows: {[t for t in tiles]}\n")
    rows = {n: bin_errors(d, tiles) for n, d in data.items()}
    alb = np.array([r[2] for r in rows["c4 (all free)"]])
    masks = {"all": np.ones_like(alb, bool), "albedo>0.3": alb > 0.3,
             "0.08<albedo<0.3": (alb >= 0.08) & (alb <= 0.3), "albedo<0.08": alb < 0.08}

    for n, r in rows.items():
        E = {g: np.array([b[3][g] for b in r]) for g in GASES}
        free = [g for g in GASES if np.std(E[g]) > 1e-12]
        print(f"=== {n}: {len(r)} bins, free gases: {[SHORT[g] for g in free]}")
        for mname, m in masks.items():
            if m.sum() < 5:
                continue
            rms = "  ".join(f"{SHORT[g]}={np.sqrt(np.mean(E[g][m] ** 2)):.3g}" for g in GASES)
            mean = "  ".join(f"{SHORT[g]}={E[g][m].mean():+.3g}" for g in GASES)
            print(f"  [{mname:16s} n={m.sum():3d}] rms: {rms}")
            print(f"  {'':22s}  bias: {mean}")
        m = masks["all"]
        M = np.stack([E[g][m] for g in free], 1)
        var, v = pca(M)
        print(f"  PCA(all bins, {len(free)} free): variance fractions {np.round(var, 3)}; "
              f"mode-1 loadings " + ", ".join(f"{SHORT[g]}={c:+.2f}" for g, c in zip(free, v)))
        C = np.corrcoef(M.T)
        print("  corr: " + "; ".join(f"{SHORT[free[i]]}-{SHORT[free[j]]}={C[i, j]:+.2f}"
                                    for i in range(len(free)) for j in range(i + 1, len(free))))
        # mask-limited PCA on the non-near-zero albedo bins (user's stated regime)
        m2 = alb > 0.08
        var2, v2 = pca(np.stack([E[g][m2] for g in free], 1))
        print(f"  PCA(albedo>0.08): variance {np.round(var2, 3)}; mode-1 "
              + ", ".join(f"{SHORT[g]}={c:+.2f}" for g, c in zip(free, v2)), "\n")

    # figure: CO2-vs-p error scatter and rms bars
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
    for k, (n, r) in enumerate(rows.items()):
        ec = np.array([b[3]["co2_ppm"] for b in r]); ep = np.array([b[3]["p_surface_hpa"] for b in r])
        sc = ax[k].scatter(ec, ep, c=alb, s=14, cmap="viridis", vmin=0, vmax=0.5)
        ax[k].set(title=n, xlabel="CO2 error [ppm]", ylabel="p error [hPa]")
        ax[k].axhline(0, c="k", lw=.4); ax[k].axvline(0, c="k", lw=.4)
    fig.colorbar(sc, ax=ax[:3], label="albedo", shrink=.8)
    w = 0.25
    for k, (n, r) in enumerate(rows.items()):
        rms = [np.sqrt(np.mean(np.array([b[3][g] for b in r]) ** 2)) for g in GASES[:2]]
        ax[3].bar(np.arange(2) + k * w, rms, w, label=n)
    ax[3].set_xticks(np.arange(2) + w); ax[3].set_xticklabels(["CO2 [ppm]", "p [hPa]"])
    ax[3].set_title("rms error, shared windows"); ax[3].legend(fontsize=8)
    fig.suptitle(f"solver = {solver}")
    out = REPO / f"plots/frozen_var_compare_c4_c4t_c4h_{solver}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    for solver in ("single_scatter", "xrtm"):
        main(solver)
