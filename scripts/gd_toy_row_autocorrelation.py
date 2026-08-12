#!/usr/bin/env python3
"""Empirical along-slit autocorrelation of the native-pipeline retrieval
bias -- how many detector rows apart before two retrievals are actually
independent?

The realistic scene (not uniform) is the right test for this. A uniform
scene has spatially FLAT truth, so `radiance(eta)` doesn't depend on `eta`
at all -- every column sees the identical spectrum no matter which true
slit position keystone nominally assigns it, so there is *nothing* for
keystone to blend. A uniform-scene autocorrelation (see
`--scene uniform`) is still a real, valid measurement, but of a narrower
effect: pure wavelength-registration correlation from smile (row-to-row
dispersion drift), with no absorption-line-depth or continuum blending in
it at all, because there's no scene structure to blend. To see the effect
of keystone mixing genuinely different true spectra -- real absorption
depth changes and continuum changes across a row/neighborhood -- the truth
has to actually vary with position, i.e. the realistic scene, which is
this script's default.

Two curves are computed per case, undetrended (unlike the uniform-scene
version of this script, no polynomial trend is removed here -- with a
spatially varying truth, "the trend" is real atmospheric signal, and
removing it deterministically without also removing the real short-range
correlation we're after is not safe; raw correlation is shown instead over
a long enough lag range that the two regimes -- short-range PSF/keystone
excess, long-range shared-truth decay -- are both visible and separable
by eye):

  - native and undistorted separately: undistorted has no PSF and no
    per-pixel geometric rendering, so any correlation IT shows is driven
    purely by the real atmosphere's own spatial correlation acting through
    the retrieval's information content/degeneracy structure -- present in
    BOTH pipelines, and not what this script is trying to isolate.
  - (native - undistorted): the excess correlation specific to native,
    i.e. specifically attributable to keystone+PSF blending genuinely
    different true spectra together, with the shared-truth component
    subtracted out.

A third option, `--scene realistic_barcode`, layers a sharp barcode
brightness pattern on top of the same genuinely-varying realistic truth --
the along-slit atmosphere still has real absorption-depth/continuum
structure, but now with a high-spatial-frequency (few-row-scale) brightness
step added on top, specifically chosen (see
`geocarb_gert.focalplane.barcode_scene`'s own docstring) to stress-test
keystone/PSF row-mixing where the scene varies fastest, rather than only
over the realistic scene's broad, smooth gradients/plumes.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_toy_row_autocorrelation.py
      PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_toy_row_autocorrelation.py --scene uniform
      PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_toy_row_autocorrelation.py --scene realistic_barcode
Output: plots/gd_toy_row_autocorrelation_<scene>.png
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geocarb_gert import chi2_outlier_mask, along_slit_scene as als
from geocarb_gert.cross_band import fpas_tag

REPO_ROOT = Path(__file__).resolve().parent.parent
PRIMARY_QUANTITY = {0: "p_surface", 1: "co2", 2: "co2", 3: "ch4"}
FPA_COLOR = {0: "tab:blue", 1: "tab:orange", 2: "tab:red", 3: "tab:purple"}
KM_PER_ROW = 2.0 * als.SLIT_HALF_KM / 1024.0   # 2.734 km -- real 2.75 km/row N/S pixel pitch
PSF_FWHM_ROWS = 1.5
MODE_SUFFIX = {"realistic": "", "uniform": "_uniform", "barcode": "_barcode",
              "realistic_barcode": "_realistic_barcode"}


def _row_series(fpa: int, pipeline: str, quantity: str, scene: str) -> np.ndarray:
    """Dense array over the full row range, NaN where the row didn't
    converge, was a chi2 outlier, or is simply missing from the pkl."""
    path = REPO_ROOT / "results" / f"gd_joint_{fpas_tag([fpa])}{MODE_SUFFIX[scene]}.pkl"
    with open(path, "rb") as f:
        d = pickle.load(f)
    out = d["out"]
    rows, vals, chi2s = [], [], []
    for (pl, order, rt), v in out.items():
        if pl != pipeline or v is None or not v.get("_conv") or v.get("_diverged"):
            continue
        if quantity not in v:
            continue
        rows.append(rt[0])
        vals.append(v[quantity])
        chi2s.append(v["_chi2"])
    rows = np.array(rows); vals = np.array(vals); chi2s = np.array(chi2s)
    outlier = chi2_outlier_mask(chi2s, n_mad=8.0)
    dense = np.full(1024, np.nan)
    dense[rows[~outlier]] = vals[~outlier]
    return dense


def _raw_acf(dense: np.ndarray, max_lag: int) -> np.ndarray:
    """Pearson autocorrelation of `dense` at lags 0..max_lag, no
    detrending -- pairs where either point is NaN are dropped per-lag."""
    acf = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        a = dense[:len(dense) - k] if k > 0 else dense
        b = dense[k:] if k > 0 else dense
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 20:
            acf[k] = np.nan
            continue
        acf[k] = 1.0 if k == 0 else np.corrcoef(a[ok], b[ok])[0, 1]
    return acf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", choices=["realistic", "uniform", "barcode", "realistic_barcode"],
                    default="realistic")
    ap.add_argument("--max-lag", type=int, default=150,
                    help="rows (default 150 -- long enough to show both the short "
                         "PSF/keystone-scale regime and the long real-atmosphere-scale one)")
    args = ap.parse_args()
    scene, max_lag = args.scene, args.max_lag

    lags = np.arange(max_lag + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))

    print(f"FPA2, {scene} scene: native vs. undistorted raw ACF...", flush=True)
    dense_native = _row_series(2, "native", "co2", scene)
    dense_undist = _row_series(2, "undistorted", "co2", scene)
    acf_native = _raw_acf(dense_native, max_lag)
    acf_undist = _raw_acf(dense_undist, max_lag)
    excess = acf_native - acf_undist

    ax1.plot(lags, acf_native, color="tab:red", lw=1.4, label="FPA2 native (co2 bias)")
    ax1.plot(lags, acf_undist, color="tab:green", lw=1.4, label="FPA2 undistorted (co2 bias)")
    ax1.plot(lags, excess, color="black", lw=1.6, ls="--",
             label="excess (native $-$ undistorted) -- isolates keystone+PSF")
    ax1.axvspan(0, PSF_FWHM_ROWS, color="gray", alpha=0.15, label=f"PSF FWHM ({PSF_FWHM_ROWS} rows)")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xlabel("lag [detector rows]")
    ax1.set_ylabel("autocorrelation (raw, no detrend)")
    ax1.set_title(f"FPA2, {scene} scene: native carries real spectral mixing undistorted can't", fontsize=11)
    ax1.legend(fontsize=8)
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(np.array(ax1.get_xlim()) * KM_PER_ROW)
    ax1_top.set_xlabel("lag [km]")

    print(f"all four bands, {scene} scene: native raw ACF...", flush=True)
    for fpa in (0, 1, 2, 3):
        q = PRIMARY_QUANTITY[fpa]
        dense = _row_series(fpa, "native", q, scene)
        acf = _raw_acf(dense, max_lag)
        ax2.plot(lags, acf, color=FPA_COLOR[fpa], lw=1.2, label=f"FPA{fpa} native ({q})")
    ax2.axvspan(0, PSF_FWHM_ROWS, color="gray", alpha=0.15, label=f"PSF FWHM ({PSF_FWHM_ROWS} rows)")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("lag [detector rows]")
    ax2.set_ylabel("autocorrelation (raw, no detrend)")
    ax2.set_title(f"Native along-slit autocorrelation, all four bands ({scene} scene)", fontsize=11)
    ax2.legend(fontsize=8)
    ax2_top = ax2.twiny()
    ax2_top.set_xlim(np.array(ax2.get_xlim()) * KM_PER_ROW)
    ax2_top.set_xlabel("lag [km]")

    fig.suptitle("How many rows apart before two native retrievals are independent?\n"
                f"({scene} scene -- truth genuinely varies with position)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    plots_dir = REPO_ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)
    out_path = plots_dir / f"gd_toy_row_autocorrelation_{scene}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
