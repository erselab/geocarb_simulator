#!/usr/bin/env python3
"""Mode-1 dense (500m) truth rendered with a WIDER along-slit PSF, to
simulate the telescope's focal-adjustment mechanism defocused (2026-09-06,
user: "I'd like to plan some experiments with wider PSFs to simulate a
defocusing effect ... It doesn't affect the spectral resolution, just the
spatial blurring").

Unlike the reversed-truth control (`build_whole_slit_truth_reversed.py`),
no field monkeypatching is needed here -- the truth fields themselves are
untouched; only the final along-slit Gaussian PSF blur
(`geocarb_gert.gd_render.predict_neighborhood`'s own `spatial_psf_fwhm_px`)
widens. Reads the target FWHM from the `PSF_FWHM_PX` env var (no silent
default -- must be set explicitly at submission time, per this session's
own GRATIO near-miss lesson).

Builds via the SAME shared no-spectral-interpolation forward model
(render_at_anchors/footprint_average_scene) every other truth image this
project uses, via `gd_build_resolution_matched_truth.build_whole_slit_
truth(..., spatial_psf_fwhm_px=<value>)` -- `pad` auto-scales via
`geocarb_gert.joint_state.default_pad_for_psf` to avoid truncating the
wider PSF's own Gaussian kernel at window edges.
"""
import os
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/scratch/scrowel3_lab/geocarb_simulator")
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from gd_joint_block_retrieve import FPA, GERT_ROOT, band_basics  # noqa: E402
from gd_joint_block_whole_slit_sweep import _make_state_spectrum  # noqa: E402
from gd_build_resolution_matched_truth import build_whole_slit_truth  # noqa: E402
import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import along_slit_scene as als, sample_geometries  # noqa: E402
from geocarb_gert.instrument import GEOCARB_BANDS  # noqa: E402

psf_fwhm_px = float(os.environ["PSF_FWHM_PX"])
print(f"defocus PSF FWHM = {psf_fwhm_px} px (nominal 1.5px)", flush=True)

fpa = FPA
band_label = GEOCARB_BANDS[fpa][0]
print("Loading absco/solar...", flush=True)
block = gg.geocarb_demo(verbose=False)["blocks"][0]
_, _, geo = sample_geometries(block, n=1, seed=0)[0]
absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
atm_center = als.atmosphere_at(0.0)
import gd_test as gdt  # noqa: E402
snr = gdt.DEFAULT_SNR_BY_FPA[fpa]
wide_win, wide_inst, albedo = band_basics(fpa, atm_center, absco, geo, solar)
spectrum = _make_state_spectrum(absco, wide_inst, geo, solar, albedo)
wn_hires = wide_win.wn_hires
ils = wide_win.ils
print("done.\n", flush=True)

print(f"=== building whole-slit DEFOCUSED truth, mode=dense, dx_km=0.5, "
     f"spatial_psf_fwhm_px={psf_fwhm_px} ===", flush=True)
t0 = time.time()
A = build_whole_slit_truth(fpa, "dense", spectrum, wn_hires, ils, band_label,
                           dx_km=0.5, n_workers=16, spatial_psf_fwhm_px=psf_fwhm_px)
print(f"  done in {time.time()-t0:.0f}s, shape={A.shape}, "
     f"range=[{A.min():.4g},{A.max():.4g}]", flush=True)

fwhm_tag = str(psf_fwhm_px).replace(".", "p")
out_path = REPO_ROOT / "scratch_work" / f"whole_slit_truth_defocus_fwhm{fwhm_tag}px.pkl"
with open(out_path, "wb") as f:
    pickle.dump({"dense": A}, f)
print(f"\nsaved {out_path}")
