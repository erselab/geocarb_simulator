#!/usr/bin/env python3
"""Design realistic along-slit atmospheric variability (XCO2, XCH4, XCO,
H2O, surface pressure) for a new single-band (FPA2) stress test: with
along-slit *composition and pressure* structure -- not just albedo (§9j) --
does keystone/smile confuse the retrieval into misattributing a real
geophysical gradient/plume to the wrong slit position?

This script only builds and plots the along-slit truth profiles -- no
rendering or retrieval yet, per instruction ("before you start the GD
pipeline, I would like to see plots").

Coordinate convention: the physical along-slit distance is taken as linear
in eta (eta=-1..+1 -> -1400..+1400 km, matching this repo's real slit
length of ~2800 km -- see geosat_geometry.LongSlitGeoSatellite's
slit_length_km=3000 default, close enough that this is a reasonable
approximation; the real GD curves' eta-to-real-row mapping is nonlinear
per FPA, but that nonlinearity is a geometric-distortion effect this study
is deliberately isolating, not something to bake into the *truth* scene).

Scenario design (baselines from geocarb_gert.scene.reference_atmosphere /
_WELL_MIXED: CO2 415 ppm, CH4 1900 ppb, CO 100 ppb, H2O surface 1% VMR,
p_surface 1013.25 hPa):

- A broad, smooth background gradient in every quantity (synoptic-scale
  variability), plus
- A CO2+CO enhancement co-located with a topographic depression (a
  wildfire-in-mountainous-terrain scenario -- CO2 and CO are commonly
  co-emitted by combustion), and
- A CH4 enhancement at a *different* slit location (an independent
  wetland/oil-and-gas-basin-type source), to test whether GD blending
  smears one feature into the other's rows.
- A smooth humidity (H2O) gradient across the whole transect (arid to
  humid climate zones), not a localized plume -- humidity doesn't have
  point sources the way the trace gases do.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_along_slit_atm_profiles.py
Output: plots/gd_along_slit_atm_profiles_fpa2.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent

# -- baselines (geocarb_gert.scene.reference_atmosphere / _WELL_MIXED) --
XCO2_BG_PPM   = 415.0
XCH4_BG_PPB   = 1900.0
XCO_BG_PPB    = 100.0
H2O_BG_VMR    = 1.0e-2      # surface VMR, fraction
P_BG_HPA      = 1013.25

SLIT_HALF_KM = 1400.0        # -> 2800 km total slit length


def _gauss(x, x0, width, amp):
    return amp * np.exp(-0.5 * ((x - x0) / width) ** 2)


# Small, localized "hot spots" -- point-source-scale features, deliberately
# narrower than the along-slit ground sample distance (~2.7 km/row for the
# ~2800 km slit over 1024 rows) times a handful of rows, i.e. only a few
# rows wide -- to stress-test whether keystone/smile row-crossing preserves
# or smears a genuinely localized source, as opposed to the broad regional
# plumes/gradients above. Placed away from the existing features so each
# shows up in isolation.
HOTSPOTS_CO2 = [(-1100.0, 10.0, 5.0), (1050.0, 9.0, 4.0)]     # (x0_km, width_km, amp_ppm)
HOTSPOTS_CH4 = [(-50.0, 12.0, 35.0), (1050.0, 9.0, 25.0)]     # (x0_km, width_km, amp_ppb) -- second co-located with the CO2 one at +1050 km (a small combined facility)
HOTSPOTS_CO  = [(-1100.0, 10.0, 35.0), (1050.0, 9.0, 28.0)]   # (x0_km, width_km, amp_ppb) -- coincident with both CO2 hot spots (combustion co-emission)


def xco2_ppm(x_km):
    background = XCO2_BG_PPM + 2.0 * np.sin(2 * np.pi * (x_km + 1400) / 3200)
    plume = _gauss(x_km, x0=-500.0, width=60.0, amp=6.0)
    hotspots = sum(_gauss(x_km, x0, w, amp) for x0, w, amp in HOTSPOTS_CO2)
    return background + plume + hotspots


def xco_ppb(x_km):
    background = XCO_BG_PPB + 10.0 * np.sin(2 * np.pi * (x_km + 1400) / 2600 + 1.0)
    plume = _gauss(x_km, x0=-500.0, width=55.0, amp=160.0)   # co-located with the CO2 plume
    hotspots = sum(_gauss(x_km, x0, w, amp) for x0, w, amp in HOTSPOTS_CO)
    return background + plume + hotspots


def xch4_ppb(x_km):
    background = XCH4_BG_PPB + 15.0 * np.sin(2 * np.pi * (x_km + 1400) / 2200 + 2.5)
    plume = _gauss(x_km, x0=650.0, width=100.0, amp=45.0)   # separate location from CO2/CO
    hotspots = sum(_gauss(x_km, x0, w, amp) for x0, w, amp in HOTSPOTS_CH4)
    return background + plume + hotspots


def h2o_surface_vmr(x_km):
    # smooth arid -> humid climatological gradient across the whole transect
    t = 0.5 * (1.0 + np.tanh(x_km / 500.0))
    return 0.005 + t * (0.018 - 0.005)


def p_surface_hpa(x_km):
    background = P_BG_HPA - 3.0 * np.sin(2 * np.pi * (x_km + 1400) / 2800 + 0.5)
    mountain = _gauss(x_km, x0=-250.0, width=140.0, amp=-250.0)   # topographic depression
    return background + mountain


def main() -> int:
    x_km = np.linspace(-SLIT_HALF_KM, SLIT_HALF_KM, 1024)
    eta = x_km / SLIT_HALF_KM

    fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)

    axes[0].plot(x_km, xco2_ppm(x_km), color="tab:red")
    axes[0].axhline(XCO2_BG_PPM, color="gray", lw=0.5, ls="--")
    axes[0].set_ylabel("XCO2 [ppm]")
    axes[0].set_title("Along-slit truth profiles for the composition/pressure stress test (FPA2)")

    axes[1].plot(x_km, xch4_ppb(x_km), color="tab:orange")
    axes[1].axhline(XCH4_BG_PPB, color="gray", lw=0.5, ls="--")
    axes[1].set_ylabel("XCH4 [ppb]")

    axes[2].plot(x_km, xco_ppb(x_km), color="tab:brown")
    axes[2].axhline(XCO_BG_PPB, color="gray", lw=0.5, ls="--")
    axes[2].set_ylabel("XCO [ppb]")

    axes[3].plot(x_km, h2o_surface_vmr(x_km) * 100, color="tab:blue")
    axes[3].set_ylabel("H2O surface VMR [%]")

    axes[4].plot(x_km, p_surface_hpa(x_km), color="tab:green")
    axes[4].axhline(P_BG_HPA, color="gray", lw=0.5, ls="--")
    axes[4].set_ylabel("Surface pressure [hPa]")
    axes[4].set_xlabel("along-slit distance [km]  (slit length ~2800 km, eta = distance/1400)")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax2 = ax.secondary_xaxis("top", functions=(lambda k: k / SLIT_HALF_KM, lambda e: e * SLIT_HALF_KM))
        if ax is axes[0]:
            ax2.set_xlabel("eta")
        else:
            ax2.set_xticklabels([])

    fig.tight_layout()
    out_path = REPO_ROOT / "plots" / "gd_along_slit_atm_profiles_fpa2.png"
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
