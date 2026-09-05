#!/usr/bin/env python3
"""Direct old-vs-new forward-model regression comparison (2026-09-04),
the last open item on the footprint-integrated-forward-model merge
checklist: how much do actual pixel radiances change between the OLD
mechanism (`als.build_lookup_radiance`'s two-sample linear spectral
blend, n_samples=400 default -- what every production sweep used before
this branch) and the NEW mechanism (`render_at_anchors`'s real per-
anchor RT + `footprint_average_scene`'s exact sub-pixel integration),
for the SAME physical truth scene (default STATE_FIELDS, no barcode/
uniform/resolution-matching)?

Both mechanisms share the exact same ForwardModel construction (same
absco/geo/solar/albedo/molecules) -- the only thing that differs is how
many along-slit samples are taken and whether a pixel gets a footprint
integral or a 2-point linear-interpolated stand-in.
"""
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/scratch/scrowel3_lab/geocarb_simulator")
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import gd_test as gdt  # noqa: E402
from gd_joint_block_retrieve import FPA, GERT_ROOT, band_basics, _eta_of  # noqa: E402
from gd_joint_block_whole_slit_sweep import _make_state_spectrum  # noqa: E402
from geocarb_gert.joint_state import render_at_anchors  # noqa: E402
from geocarb_gert.focalplane import footprint_average_scene  # noqa: E402
from geocarb_gert import gd_render, along_slit_scene as als, sample_geometries  # noqa: E402
import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402

fpa = FPA
rows_win = np.arange(400, 451)   # a generic mid-slit 51-row window

print("Loading absco/solar/geometry...", flush=True)
block = gg.geocarb_demo(verbose=False)["blocks"][0]
_, _, geo = sample_geometries(block, n=1, seed=0)[0]
absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
atm_center = als.atmosphere_at(0.0)
wide_win, wide_inst, albedo = band_basics(fpa, atm_center, absco, geo, solar)
print("done.\n", flush=True)

# ---- OLD mechanism: two-sample linear spectral blend ----
print("=== OLD: build_lookup_radiance (n_samples=400, the pre-migration default) ===", flush=True)
t0 = time.time()
wn_hires_old, radiance_old = als.build_lookup_radiance(
    absco, wide_inst, geo, solar, [albedo], n_samples=400, n_workers=16)
print(f"  precompute: {time.time()-t0:.1f}s", flush=True)
t0 = time.time()
A_old = gd_render.predict_neighborhood(fpa, rows_win, wn_hires_old, radiance_old,
                                       wide_win.ils, pad=4, footprint=False)
print(f"  render:     {time.time()-t0:.1f}s  shape={A_old.shape}", flush=True)

# ---- NEW mechanism: real per-anchor RT + exact footprint integration ----
print("\n=== NEW: render_at_anchors + footprint_average_scene (dx_km=2.0) ===", flush=True)
spectrum = _make_state_spectrum(absco, wide_inst, geo, solar, albedo)
dx_km = 2.0
pad = 4
a_lo, a_hi = int(rows_win.min()) - pad, int(rows_win.max()) + pad
anchor_rows = np.arange(a_lo, a_hi + 1e-9, 1.0)
anchor_etas_edges = _eta_of(fpa, np.full(anchor_rows.shape, 512.0), anchor_rows)
lo_km, hi_km = anchor_etas_edges.min() * als.SLIT_HALF_KM, anchor_etas_edges.max() * als.SLIT_HALF_KM
n_anchor = max(2, int(round((hi_km - lo_km) / dx_km)) + 1)
anchor_x_km = np.linspace(lo_km, hi_km, n_anchor)
anchor_etas = anchor_x_km / als.SLIT_HALF_KM
print(f"  anchor range: x_km=[{lo_km:.1f},{hi_km:.1f}], n_anchor={n_anchor}", flush=True)
atm_params = {n: np.asarray(fn(anchor_x_km), dtype=float) for n, fn in als.STATE_FIELDS.items()}
surf_params = {}   # no vary_albedo -- matches build_lookup_radiance's own vary_albedo=False default

t0 = time.time()
A_new = render_at_anchors(fpa, rows_win, anchor_etas, atm_params, surf_params, spectrum,
                          wide_win.wn_hires, wide_win.ils, pad=4, n_workers=16)
print(f"  render:     {time.time()-t0:.1f}s  shape={A_new.shape}", flush=True)

# ---- Compare ----
diff = A_new - A_old
rel = diff / np.maximum(np.abs(A_old), 1e-12)
print("\n=== Comparison (same truth scene, same rows, different rendering mechanism) ===")
print(f"A_old range: [{A_old.min():.5g}, {A_old.max():.5g}]")
print(f"A_new range: [{A_new.min():.5g}, {A_new.max():.5g}]")
print(f"abs diff:  rms={np.sqrt(np.mean(diff**2)):.5g}  max={np.max(np.abs(diff)):.5g}")
print(f"rel diff:  rms={np.sqrt(np.mean(rel**2)):.5g}  max={np.max(np.abs(rel)):.5g}  "
     f"(as % of local radiance)")
worst = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
print(f"worst pixel: row={rows_win[worst[0]]} col={worst[1]}  "
     f"old={A_old[worst]:.5g} new={A_new[worst]:.5g} diff={diff[worst]:.5g}")
print("DONE")
