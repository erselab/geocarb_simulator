#!/usr/bin/env python3
"""Qualitative look at native-grid post-fit residuals across the slit: is
there periodic structure, and if so, does its period vary row to row?

Plots, for several rows spanning the slit and both dispersion orders:
1. The real-space residual vs wavenumber.
2. Its power spectrum (periodogram vs period in channels), to assess
   whether any periodicity is a genuine narrow spectral peak or just
   broadband noise, and whether a peak's location shifts across rows.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_native_residual_periodicity.py
Output: plots/gd_native_residual_periodicity_fpa2.png
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_ROWS = [50, 200, 400, 512, 700, 900, 970]


def _is_good(b):
    return (b is not None and not b.get("_diverged") and b.get("_conv")
            and np.isfinite(b.get("co2", np.nan)))


def main() -> int:
    with open(REPO_ROOT / "results" / "gd_dense_sweep.pkl", "rb") as f:
        d = pickle.load(f)
    out = d["out"]

    fig, axes = plt.subplots(len(TEST_ROWS), 4, figsize=(16, 2.2 * len(TEST_ROWS)))

    for i, k in enumerate(TEST_ROWS):
        for j, order in enumerate((0, 2)):
            v = out[("native", order, int(k))]
            resid, nu = v["residual"], v["nu"]
            good = resid is not None and _is_good(v["bias"])

            ax_r = axes[i, 2 * j]
            ax_p = axes[i, 2 * j + 1]
            if not good:
                ax_r.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax_r.transAxes)
                ax_p.axis("off")
                continue

            wn_asc = nu[::-1]     # nu is descending (ascending-wavelength); flip for a natural x-axis
            resid_asc = resid[::-1]
            ax_r.plot(wn_asc, resid_asc, lw=0.6)
            ax_r.set_title(f"row {k}, order={order}: residual", fontsize=8)
            ax_r.tick_params(labelsize=6)

            r = resid_asc - resid_asc.mean()
            F = np.abs(np.fft.rfft(r)) ** 2
            n = len(r)
            period_channels = np.concatenate(([np.inf], n / np.arange(1, len(F))))
            ax_p.plot(period_channels[1:60], F[1:60], lw=0.8)
            ax_p.set_xlim(2, 60)
            ax_p.set_title("power vs period [channels]", fontsize=8)
            ax_p.tick_params(labelsize=6)

    fig.suptitle("Native-pipeline residual periodicity -- FPA2, uniform desert scene", fontsize=13)
    fig.tight_layout()
    out_path = REPO_ROOT / "plots" / "gd_native_residual_periodicity_fpa2.png"
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
