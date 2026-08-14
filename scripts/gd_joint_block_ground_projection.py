#!/usr/bin/env python3
"""Project a joint-block window's bin centers/boundaries and real detector
row centers onto the Earth's surface, for a hypothetical geometry with the
slit center pointing at a given sub-satellite point (default: equator,
95 deg W, the same longitude geosat_geometry.geocarb_demo() already uses).

This is possible, and meaningful, because eta (the normalized along-slit
coordinate used throughout this whole investigation) is *already* defined
as the true object-space slit position -- "eta = -1 and eta = +1 always
correspond to the true top and bottom of that FPA's slit" (gd_render.py).
Keystone and smile are detector-side effects: they scramble *which*
detector row captures a given eta, not what eta physically means. So
projecting eta -> ground position via the correct (undistorted) viewing
geometry is exactly right regardless of which row happens to sample it --
that's what makes the bin-vs-row ground comparison meaningful at all.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_joint_block_ground_projection.py \\
        [--row-min 890] [--row-max 935] [--n-bins 15] \\
        [--center-lat 0.0] [--center-lon -95.0] [--context-pad 60]
Output: plots/joint_block/gd_joint_block_ground_projection_fpa2_row{row_min}-{row_max}_G{n_bins}.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from gd_joint_block_retrieve import FPA, _eta_of  # noqa: E402
from gd_joint_block_diagnostics import pixel_density_bin_centers  # noqa: E402

from geocarb_gert import along_slit_scene as als  # noqa: E402
from geosat_geometry import (LongSlitGeoSatellite, geodetic_to_ecef,  # noqa: E402
                             ecef_to_geodetic, _ray_ellipsoid_intersect)


def eta_to_latlon(sat: LongSlitGeoSatellite, center_lat_deg: float, center_lon_deg: float,
                  eta: np.ndarray):
    """Generalizes LongSlitGeoSatellite.slit_centers (fixed n_pixels grid)
    to arbitrary continuous eta in [-1, 1] -- same math, same convention
    (eta=-1 southernmost), just parameterized by eta instead of pixel
    index, so bin centers/boundaries (which don't land on integer pixel
    positions) can be projected too."""
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    gnd_center = geodetic_to_ecef(float(center_lat_deg), float(center_lon_deg))
    L0 = gnd_center - sat.sat_ecef
    L0 = L0 / np.linalg.norm(L0)
    north = np.array([0.0, 0.0, 1.0])
    slit_dir = north - np.dot(north, L0) * L0
    slit_dir = slit_dir / np.linalg.norm(slit_dir)
    half_angle = (sat.n_pixels - 1) / 2.0 * sat.ifov_ns_rad
    alphas = eta * half_angle
    look_dirs = (np.cos(alphas)[:, None] * L0 + np.sin(alphas)[:, None] * slit_dir)
    gnd_pts = _ray_ellipsoid_intersect(sat.sat_ecef, look_dirs)
    lats, lons, _ = ecef_to_geodetic(gnd_pts)
    return lats, lons


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row-min", type=int, default=890)
    ap.add_argument("--row-max", type=int, default=935)
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--center-lat", type=float, default=0.0, help="sub-satellite/slit-center latitude [deg N]")
    ap.add_argument("--center-lon", type=float, default=-95.0, help="sub-satellite/slit-center longitude [deg E]")
    ap.add_argument("--context-pad", type=int, default=60, help="extra rows shown either side in the context panel")
    ap.add_argument("--map-extent", type=float, nargs=4, default=[-100, -60, -25, 25],
                    metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
                    help="context-panel map extent")
    args = ap.parse_args()

    row_min, row_max, G = args.row_min, args.row_max, args.n_bins
    center_lat, center_lon = args.center_lat, args.center_lon
    if row_min < 0 or row_max > 1023 or row_min >= row_max:
        raise SystemExit(f"invalid row range: {row_min}-{row_max} (must be within 0-1023, row_min < row_max)")
    if G < 2:
        raise SystemExit(f"--n-bins must be >= 2, got {G}")

    n_px = 1024
    slit_km = 2 * als.SLIT_HALF_KM
    sat = LongSlitGeoSatellite(sat_lon_deg=center_lon, slit_length_km=slit_km,
                               pixel_size_ew_km=6.0, pixel_size_ns_km=slit_km / n_px,
                               integration_time_s=10.0)
    print(f"satellite: lat={center_lat}, lon={center_lon}, n_pixels={sat.n_pixels}, "
         f"pixel_size_ns_km={sat.pixel_size_ns_km:.3f}")
    print(f"window: rows {row_min}-{row_max} ({row_max-row_min+1} rows), G={G} bins", flush=True)

    # -- row centers (real, keystone/smile-affected row -> eta map) --
    row_lo_ctx = max(0, row_min - args.context_pad)
    row_hi_ctx = min(1023, row_max + args.context_pad)
    rows_context = np.arange(row_lo_ctx, row_hi_ctx + 1)
    eta_rows_context = _eta_of(FPA, np.full(len(rows_context), 512.0), rows_context.astype(float))
    lat_rows_context, lon_rows_context = eta_to_latlon(sat, center_lat, center_lon, eta_rows_context)

    rows_win = np.arange(row_min, row_max + 1)
    eta_rows_win = _eta_of(FPA, np.full(len(rows_win), 512.0), rows_win.astype(float))
    lat_rows_win, lon_rows_win = eta_to_latlon(sat, center_lat, center_lon, eta_rows_win)

    # -- bin centers/boundaries (pixel-density, same scheme as throughout) --
    cols = np.arange(1024.0)
    eta_all = np.stack([_eta_of(FPA, cols, np.full(1024, float(i))) for i in rows_win])
    bin_centers = pixel_density_bin_centers(eta_all.ravel(), G)
    bin_edges = 0.5 * (bin_centers[:-1] + bin_centers[1:])
    lat_bins, lon_bins = eta_to_latlon(sat, center_lat, center_lon, bin_centers)
    lat_edges, lon_edges = eta_to_latlon(sat, center_lat, center_lon, bin_edges) if len(bin_edges) else (np.array([]), np.array([]))

    lat_full, lon_full = eta_to_latlon(sat, center_lat, center_lon, np.array([-1.0, 0.0, 1.0]))
    print(f"full slit spans lat {lat_full[0]:.3f} to {lat_full[-1]:.3f}, "
         f"lon {lon_full[0]:.3f} to {lon_full[-1]:.3f}")
    print(f"window (rows {row_min}-{row_max}) spans "
         f"lat {lat_rows_win.min():.4f} to {lat_rows_win.max():.4f}, "
         f"lon {lon_rows_win.min():.4f} to {lon_rows_win.max():.4f}")

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5})

    fig = plt.figure(figsize=(13, 9.5))

    # -- context panel: full slit on a real map, window marked --
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
        ax1.set_extent(args.map_extent, crs=ccrs.PlateCarree())
        ax1.add_feature(cfeature.LAND, facecolor="#EDE8DD")
        ax1.add_feature(cfeature.OCEAN, facecolor="#D7E4EC")
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#3a3a3a")
        ax1.gridlines(draw_labels=True, linewidth=0.3, color="0.7")
        ax1.plot(lon_rows_context, lat_rows_context, "-", color="#A0631B", lw=2.5,
                 transform=ccrs.PlateCarree(), label=f"rows {rows_context[0]}-{rows_context[-1]}")
        ax1.plot([center_lon], [center_lat], "*", color="black", ms=14,
                 transform=ccrs.PlateCarree(), label="slit center / sub-satellite point")
        ax1.legend(fontsize=8.5, loc="lower left")
        ax1.set_title(f"Context: FPA{FPA} slit near rows {row_min}-{row_max}, "
                     f"sub-satellite point ({center_lat}°N, {center_lon}°E)", fontsize=11)
    except Exception as e:
        print("cartopy unavailable or failed, falling back to plain scatter:", e)
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(lon_rows_context, lat_rows_context, "-", color="#A0631B", lw=2.5,
                 label=f"rows {rows_context[0]}-{rows_context[-1]}")
        ax1.plot([center_lon], [center_lat], "*", color="black", ms=14, label="slit center")
        ax1.set_xlabel("longitude"); ax1.set_ylabel("latitude")
        ax1.legend(fontsize=8.5)
        ax1.set_title("Context (no coastlines available)", fontsize=11)

    # -- zoomed panel: bin centers/boundaries vs row centers, target window --
    ax2 = fig.add_subplot(2, 1, 2)
    for lat in lat_rows_win:
        ax2.axhline(lat, color="0.85", lw=0.4, zorder=0)
    ax2.plot(np.zeros_like(lat_rows_win), lat_rows_win, "o", ms=3.5, color="#3e6f8e",
             label=f"detector row centers (col 512), n={len(rows_win)}", zorder=3)
    ax2.plot(np.full_like(lat_bins, 1.0), lat_bins, "o", ms=8, color="#A0631B",
             mec="white", mew=0.8, label=f"bin centers, G={G}", zorder=4)
    for lat in lat_edges:
        ax2.axhline(lat, color="#e8543a", lw=1.0, ls="--", alpha=0.7, zorder=2)
    ax2.plot([], [], color="#e8543a", lw=1.0, ls="--", label="bin boundaries")
    ax2.annotate(f"row {row_min}", xy=(0, lat_rows_win[0]), xytext=(-0.25, lat_rows_win[0]),
                fontsize=8, ha="right", va="center", color="#3e6f8e")
    ax2.annotate(f"row {row_max}", xy=(0, lat_rows_win[-1]), xytext=(-0.25, lat_rows_win[-1]),
                fontsize=8, ha="right", va="center", color="#3e6f8e")
    ax2.set_xlim(-0.6, 1.6)
    ax2.set_xticks([])
    lon_span = lon_rows_win.max() - lon_rows_win.min()
    lon_note = ("longitude is exactly constant here -- the slit lies along the local "
               "meridian by symmetry" if abs(center_lon - sat.sat_lon_deg) < 1e-9 and abs(lon_span) < 1e-9
               else f"longitude spans {lon_rows_win.min():.4f} to {lon_rows_win.max():.4f} across this window")
    ax2.set_ylabel("latitude [deg]")
    ax2.set_title(f"Ground projection, rows {row_min}-{row_max}: bin centers/boundaries vs. "
                 f"real detector row centers\n({lon_note} -- shown as latitude only)",
                 fontsize=10.5)
    ax2.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.tight_layout()
    plots_dir = REPO_ROOT / "plots" / "joint_block"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"gd_joint_block_ground_projection_fpa{FPA}_row{row_min}-{row_max}_G{G}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
