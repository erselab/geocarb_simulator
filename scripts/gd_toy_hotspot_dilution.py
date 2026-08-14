#!/usr/bin/env python3
"""How much of a real, single-footprint-scale point source survives native
retrieval, as a function of how many rows keystone splices into that row?

Direct answer to the thought experiment: near the end of the slit, a native
detector row's spectral columns don't all sample the same ground location --
keystone means each pixel (row, col) gets its own true along-slit position
(`geocarb_gert.gd_render.image()` evaluates every pixel's own true slit
position, not one per-row value -- see `gd_render.real_row_eta()`'s own
docstring). Near FPA2's high-keystone end (`rows_crossed(2, row)` approaches
~10 near row 1023), a single row's spectrum is genuinely spliced together,
end to end across its columns, from ~10 contiguous footprints' worth of true
along-slit atmosphere -- exactly the "10 contiguous spatial regions" thought
experiment. If one of those regions has a strong, localized anomaly (a real
point-source plume, or equivalently a strong albedo gradient), only the
fraction of the row's columns/lines whose true position lands on it carries
the anomaly's signature; a retrieval fitting ONE atmosphere to the whole row
should therefore only partially "see" it -- diluted by the rest of the row's
columns, which still see the unperturbed background.

`along_slit_scene.py`'s realistic scene already has exactly this stress
test built in: real, ~9-10 km wide Gaussian CO2/CH4/CO point sources
(`HOTSPOTS_CO2` etc.), deliberately narrower than the ~2.7 km/row ground
sample distance. One CO2 hot spot (x0=+1050 km) happens to sit almost
exactly at FPA2's highest-keystone rows (row ~910, rows_crossed~9.3); this
script also looks at the other one (x0=-1100 km, row ~112, rows_crossed
~0.96) as a low-keystone control -- same true source, ~10x less keystone
splicing.

Uses the already-existing `results/gd_joint_fpa2.pkl` (plain realistic
scene, no noise) -- no new retrieval run needed.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_toy_hotspot_dilution.py
Output: plots/gd_toy_hotspot_dilution_fpa2.png
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geocarb_gert import along_slit_scene as als
from geocarb_gert.gd_polynomials import rows_crossed, xy_to_wavelength_slit
from geocarb_gert.gd_render import s_max

REPO_ROOT = Path(__file__).resolve().parent.parent
FPA = 2
# (label, x0_km, row window to plot, background row for the "before the
# bump" reference level)
CASES = [
    ("low keystone (row~112, near slit start)", -1100.0, (95, 130), 70),
    ("high keystone (row~910, near slit end)", 1050.0, (890, 935), 880),
]


def _load():
    with open(REPO_ROOT / "results" / f"gd_joint_fpa{FPA}.pkl", "rb") as f:
        d = pickle.load(f)
    return d["out"]


def _series(out: dict, pipeline: str, quantity: str = "co2") -> tuple[np.ndarray, np.ndarray]:
    rows_, vals_ = [], []
    for (pl, order, rt), v in out.items():
        if pl != pipeline or v is None or not v.get("_conv") or v.get("_diverged"):
            continue
        if quantity not in v:
            continue
        rows_.append(rt[0])
        vals_.append(v[quantity])
    rows_ = np.array(rows_)
    vals_ = np.array(vals_)
    order = np.argsort(rows_)
    return rows_[order], vals_[order]


def _capture_fraction(rows, retrieved, true_co2, r0, r1, bg_row) -> float:
    mask = (rows >= r0) & (rows <= r1)
    rr = rows[mask].astype(int)
    true_bg = true_co2[bg_row]
    true_peak = true_co2[rr].max()
    bg_idx = rr[np.argmin(np.abs(rr - bg_row))]
    retrieved_bg = retrieved[mask][np.argmin(np.abs(rr - bg_row))]
    retrieved_peak = retrieved[mask][np.argmax(true_co2[rr])]
    return (retrieved_peak - retrieved_bg) / (true_peak - true_bg)


def main() -> int:
    out = _load()

    cols_center = np.full(1024, 512.0)
    _, s_of_row = xy_to_wavelength_slit(FPA, cols_center, np.arange(1024.0))
    x_km_of_row = (s_of_row / s_max(FPA)) * als.SLIT_HALF_KM
    true_co2 = als.xco2_ppm(x_km_of_row)

    rows_n, bias_n = _series(out, "native")
    rows_u, bias_u = _series(out, "undistorted")
    retrieved_n = true_co2[rows_n.astype(int)] + bias_n
    retrieved_u = true_co2[rows_u.astype(int)] + bias_u

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, (label, x0, (r0, r1), bg_row) in zip(axes, CASES):
        rows_all = np.arange(r0, r1 + 1)
        k = np.array([rows_crossed(FPA, r) for r in rows_all])

        ax.plot(rows_all, true_co2[rows_all], color="k", lw=2.0, label="true CO2")
        m = (rows_n >= r0) & (rows_n <= r1)
        ax.plot(rows_n[m], retrieved_n[m], color="tab:red", lw=1.6, marker=".", ms=4,
                label="native retrieved")
        m = (rows_u >= r0) & (rows_u <= r1)
        ax.plot(rows_u[m], retrieved_u[m], color="tab:green", lw=1.6, marker=".", ms=4,
                label="undistorted retrieved")

        f_n = _capture_fraction(rows_n, retrieved_n, true_co2, r0, r1, bg_row)
        f_u = _capture_fraction(rows_u, retrieved_u, true_co2, r0, r1, bg_row)

        ax.set_xlabel("detector row")
        ax.set_ylabel("CO2 [ppm]")
        ax.set_title(f"{label}\nkeystone at center: {k[len(k)//2]:.1f} rows crossed\n"
                    f"peak-enhancement captured: native {f_n*100:.0f}%, undistorted {f_u*100:.0f}%",
                    fontsize=10)
        ax.legend(fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(rows_all, k, color="gray", lw=1.0, ls=":", alpha=0.7)
        ax2.set_ylabel("keystone (rows crossed)", color="gray", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="gray")

    fig.suptitle("Does a real point source survive native retrieval? Same true CO2 hot spot,\n"
                "two along-slit locations with very different keystone amplitude (FPA2, realistic scene)",
                fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    plots_dir = REPO_ROOT / "plots" / "toy_diagnostics"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"gd_toy_hotspot_dilution_fpa{FPA}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
