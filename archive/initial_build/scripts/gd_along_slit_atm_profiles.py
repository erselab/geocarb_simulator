#!/usr/bin/env python3
"""Plot the along-slit atmospheric variability (XCO2, XCH4, XCO, H2O,
surface pressure) design for the composition/pressure stress test.

The profile functions themselves live in ``geocarb_gert.along_slit_scene``
(shared with the per-band render/retrieve driver, ``gd_band_stress_test.py``)
-- this script is plotting only.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_along_slit_atm_profiles.py
Output: plots/gd_along_slit_atm_profiles_fpa2.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geocarb_gert import along_slit_scene as als

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    x_km = np.linspace(-als.SLIT_HALF_KM, als.SLIT_HALF_KM, 1024)

    fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)

    axes[0].plot(x_km, als.xco2_ppm(x_km), color="tab:red")
    axes[0].axhline(als.XCO2_BG_PPM, color="gray", lw=0.5, ls="--")
    axes[0].set_ylabel("XCO2 [ppm]")
    axes[0].set_title("Along-slit truth profiles for the composition/pressure stress test")

    axes[1].plot(x_km, als.xch4_ppb(x_km), color="tab:orange")
    axes[1].axhline(als.XCH4_BG_PPB, color="gray", lw=0.5, ls="--")
    axes[1].set_ylabel("XCH4 [ppb]")

    axes[2].plot(x_km, als.xco_ppb(x_km), color="tab:brown")
    axes[2].axhline(als.XCO_BG_PPB, color="gray", lw=0.5, ls="--")
    axes[2].set_ylabel("XCO [ppb]")

    axes[3].plot(x_km, als.h2o_surface_vmr(x_km) * 100, color="tab:blue")
    axes[3].set_ylabel("H2O surface VMR [%]")

    axes[4].plot(x_km, als.p_surface_hpa(x_km), color="tab:green")
    # the background this field actually oscillates below -- NOT P_BG_HPA
    # itself, which is the standard-atmosphere ceiling the field is held
    # P_HEADROOM_HPA clear of so a retrieved p_surface has room to move.
    axes[4].axhline(als.P_BG_HPA - als.P_HEADROOM_HPA, color="gray", lw=0.5, ls="--")
    axes[4].axhline(als.P_BG_HPA, color="crimson", lw=0.7, ls=":")
    axes[4].text(x_km[0], als.P_BG_HPA, f" {als.P_BG_HPA} hPa -- NaN above here",
                 va="bottom", fontsize=7.5, color="crimson")
    axes[4].set_ylabel("Surface pressure [hPa]")
    axes[4].set_xlabel("along-slit distance [km]  (slit length ~2800 km, eta = distance/1400)")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax2 = ax.secondary_xaxis("top", functions=(lambda k: k / als.SLIT_HALF_KM, lambda e: e * als.SLIT_HALF_KM))
        if ax is axes[0]:
            ax2.set_xlabel("eta")
        else:
            ax2.set_xticklabels([])

    fig.tight_layout()
    out_dir = REPO_ROOT / "plots" / "scene_gallery"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gd_along_slit_atm_profiles_fpa2.png"
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
