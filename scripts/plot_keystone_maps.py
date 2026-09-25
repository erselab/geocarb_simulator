"""Keystone maps (2026-09-25): for each FPA, the slit-position shift of every detector pixel relative
to its own row's nominal (centre-column, col 512) slit position -- i.e. how far along the slit the
pixel actually looks compared with what the row is assumed to see -- from the real GD mapping
polynomials (geocarb_gert.gd_polynomials.xy_to_wavelength_slit, eta via eta_of_s). Shown in
km along the slit (eta * SLIT_HALF_KM) and in rows (eta / the band's mean row spacing).
Output: plots/keystone_maps_fpa0-3.png (km) and plots/keystone_maps_rows_fpa0-3.png (rows).
    PYTHONPATH=.:<gert> python scripts/plot_keystone_maps.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from geocarb_gert import along_slit_scene as als  # noqa: E402
from geocarb_gert.gd_polynomials import N_PX, eta_of_s, xy_to_wavelength_slit  # noqa: E402
from geocarb_gert.multiband_geometry import band_tables  # noqa: E402

NAMES = {0: "FPA0 O2-A", 1: "FPA1 CO2 weak", 2: "FPA2 CO2 strong", 3: "FPA3 CH4/CO"}
cols, rows = np.meshgrid(np.arange(N_PX, dtype=float), np.arange(N_PX, dtype=float))   # [row, col]
shift = {}
for f in range(4):
    _, s = xy_to_wavelength_slit(f, cols.ravel(), rows.ravel())
    eta = eta_of_s(f, s).reshape(N_PX, N_PX)
    shift[f] = eta - eta[:, [512]]                       # eta shift vs the row's centre column
    print(f"FPA{f}: keystone shift range {shift[f].min()*als.SLIT_HALF_KM:+.2f} .. {shift[f].max()*als.SLIT_HALF_KM:+.2f} km, "
          f"{shift[f].min()/band_tables(f)['spacing']:+.2f} .. {shift[f].max()/band_tables(f)['spacing']:+.2f} rows", flush=True)

for unit, scale_fn, tag, lab in (("km", lambda f: als.SLIT_HALF_KM, "", "shift along slit [km]"),
                                 ("rows", lambda f: 1.0 / abs(band_tables(f)["spacing"]), "_rows", "shift along slit [detector rows]")):
    fig, axs = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    for a, f in zip(axs.ravel(), range(4)):
        img = shift[f] * scale_fn(f)
        v = np.abs(img).max()
        im = a.imshow(img, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, aspect="equal", interpolation="nearest")
        a.set_title(f"{NAMES[f]}  (max |shift| {v:.3g} {unit})", loc="left", fontsize=10)
        a.set_xlabel("detector column")
        a.set_ylabel("detector row")
        fig.colorbar(im, ax=a, shrink=0.8, label=lab)
    fig.suptitle(f"Keystone: slit position of each pixel relative to its row's centre column ({unit})", fontsize=11)
    out = REPO / f"plots/keystone_maps{tag}_fpa0-3.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", out.name)
