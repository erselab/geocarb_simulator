#!/usr/bin/env python3
"""Mode-1 dense (500m) truth with the along-slit profile REVERSED
(2026-09-06, user: "In order to isolate the effects of keystone versus
the structure in the truth, we can run parallel experiments with the
truth profile along the slit reversed").

Mirrors every truth field spatially -- reversed_fn(x_km) =
original_fn(-x_km) -- while leaving the instrument geometry (row<->eta
mapping, keystone curve) completely untouched. The keystone-smearing
strength at a given ROW (rows_crossed, Sec.5) is a pure property of the
geometry and does NOT depend on the truth at all, so this reversal
changes WHICH physical along-slit content sits at row 1013-1023 (the
high-keystone far edge, Sec.5) without changing how much keystone
smearing happens there.

If rows 1013-1023 shows comparably large error under the REVERSED
truth too, that is decisive confirmation the effect is a geometry
(keystone) effect, not a coincidence of truth structure happening to
sit at that location. If the error instead follows the truth (moves to
wherever the mirrored structure that used to sit at the OTHER end now
lands), the keystone explanation would need revisiting.

Builds via the SAME shared no-spectral-interpolation forward model
(render_at_anchors/footprint_average_scene) every other truth image
this project uses, just with STATE_FIELDS/SURFACE_FIELDS monkeypatched
to their reversed versions for the duration of this standalone script.
"""
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/scratch/scrowel3_lab/geocarb_simulator")
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import gd_test as gdt  # noqa: E402
from gd_joint_block_retrieve import FPA, GERT_ROOT, band_basics  # noqa: E402
from gd_joint_block_whole_slit_sweep import _make_state_spectrum  # noqa: E402
from gd_build_resolution_matched_truth import build_whole_slit_truth  # noqa: E402
import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, sample_geometries  # noqa: E402
from geocarb_gert.instrument import GEOCARB_BANDS  # noqa: E402


def _reversed1(fn):
    return lambda x_km, _fn=fn: _fn(-np.asarray(x_km, dtype=float))


def _reversed2(fn):
    return lambda x_km, label, _fn=fn: _fn(-np.asarray(x_km, dtype=float), label)


# Build reversed versions of every field this project's truth/prior
# machinery reads, and monkeypatch them in as the module's OWN
# STATE_FIELDS/SURFACE_FIELDS -- render_dense_truth_window's own
# `fields`/`surface_fields` parameters default to `als.STATE_FIELDS`/
# `als.SURFACE_FIELDS` read at CALL TIME, so this reaches it without
# needing build_whole_slit_truth to expose those parameters itself.
_ORIG_STATE_FIELDS = dict(als.STATE_FIELDS)
_ORIG_STATE_FIELDS_PRIOR = dict(als.STATE_FIELDS_PRIOR)
_ORIG_SURFACE_FIELDS = dict(als.SURFACE_FIELDS)
_ORIG_SURFACE_FIELDS_PRIOR = dict(als.SURFACE_FIELDS_PRIOR)

als.STATE_FIELDS = {name: _reversed1(fn) for name, fn in _ORIG_STATE_FIELDS.items()}
als.STATE_FIELDS_PRIOR = {name: _reversed1(fn) for name, fn in _ORIG_STATE_FIELDS_PRIOR.items()}
als.SURFACE_FIELDS = {name: _reversed2(fn) for name, fn in _ORIG_SURFACE_FIELDS.items()}
als.SURFACE_FIELDS_PRIOR = {name: _reversed2(fn) for name, fn in _ORIG_SURFACE_FIELDS_PRIOR.items()}
# PRIOR_FIELD_SETS/SURFACE_PRIOR_FIELD_SETS are dicts of REFERENCES to
# the above -- rebuild them too so any code reading by name (e.g.
# "exact"/"structural") also sees the reversed versions.
als.PRIOR_FIELD_SETS["exact"] = als.STATE_FIELDS
als.PRIOR_FIELD_SETS["structural"] = als.STATE_FIELDS_PRIOR
als.SURFACE_PRIOR_FIELD_SETS["exact"] = als.SURFACE_FIELDS
als.SURFACE_PRIOR_FIELD_SETS["structural"] = als.SURFACE_FIELDS_PRIOR

print("Reversed STATE_FIELDS/STATE_FIELDS_PRIOR/SURFACE_FIELDS/"
     "SURFACE_FIELDS_PRIOR installed (fn(x_km) -> fn(-x_km)).", flush=True)
print("Sanity check -- xco2_ppm(500) [reversed] should equal "
     f"original xco2_ppm(-500): {als.STATE_FIELDS['co2_ppm'](500.0):.4f} vs "
     f"{_ORIG_STATE_FIELDS['co2_ppm'](-500.0):.4f}", flush=True)

fpa = FPA
band_label = GEOCARB_BANDS[fpa][0]
print("Loading absco/solar...", flush=True)
block = gg.geocarb_demo(verbose=False)["blocks"][0]
_, _, geo = sample_geometries(block, n=1, seed=0)[0]
absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
# atmosphere_at(0.0) is symmetric under reversal (x_km=0 maps to itself),
# so the center atmosphere used for band_basics's own albedo reference
# is unaffected either way.
atm_center = als.atmosphere_at(0.0)
snr = gdt.DEFAULT_SNR_BY_FPA[fpa]
wide_win, wide_inst, albedo = band_basics(fpa, atm_center, absco, geo, solar)
spectrum = _make_state_spectrum(absco, wide_inst, geo, solar, albedo)
wn_hires = wide_win.wn_hires
ils = wide_win.ils
print("done.\n", flush=True)

print("=== building whole-slit REVERSED truth, mode=dense, dx_km=0.5 ===", flush=True)
t0 = time.time()
A = build_whole_slit_truth(fpa, "dense", spectrum, wn_hires, ils, band_label,
                           dx_km=0.5, n_workers=16)
print(f"  done in {time.time()-t0:.0f}s, shape={A.shape}, "
     f"range=[{A.min():.4g},{A.max():.4g}]", flush=True)

out_path = REPO_ROOT / "scratch_work" / "whole_slit_truth_reversed.pkl"
with open(out_path, "wb") as f:
    pickle.dump({"dense": A}, f)
print(f"\nsaved {out_path}")
