#!/usr/bin/env python3
"""Single-band composition/pressure stress test: render + rectify + retrieve
a full band using the along-slit-varying truth atmosphere (geocarb_gert.
along_slit_scene) instead of the uniform scene used throughout Sec. 9.

Extends the Sec. 9m/9n dense-sweep pipeline (previously FPA2/CO2_strong
only, uniform composition, albedo-only variability) two ways at once:
1. The truth atmosphere now varies continuously along the slit (realistic
   XCO2/XCH4/XCO/H2O/surface-pressure gradients, plumes, and small hot
   spots -- see geocarb_gert.along_slit_scene and scripts/
   gd_along_slit_atm_profiles.py's plot).
2. Runs on whichever band is requested via --fpa, retrieving that band's
   own appropriate gas set (e.g. FPA1/CO2_weak retrieves co2+h2o;
   FPA3/CH4_CO retrieves ch4+co+h2o) plus p_scale, which every band's
   StateVector already includes by default. Well-mixed gases (o2, n2o --
   see WELL_MIXED_GASES) are deliberately EXCLUDED from the retrieved gas
   list even when present in the band's molecule set: their true column
   never varies along the slit (see xtrue_of_row below), so retrieving a
   separate {gas}_scale for them is nearly degenerate with p_scale (both
   change a well-mixed gas's column via the same total-air-mass path).
   Found 2026-07-24: floating o2_scale alongside p_scale for FPA0/O2_A
   produced near-100% non-convergence (chi2 already good, but dx_norm
   never settling -- the fit wandering along the o2_scale/p_scale ridge).
   FPA0 now retrieves h2o_scale + p_scale only; p_scale alone carries the
   surface-pressure signal, exactly as an operational O2-A retrieval does.

Three pipelines, each retrieved at every row with the dispersion order
matched to what's physically appropriate for it (see pipeline_orders in
main() for the full rationale):
  "native"        (order=2 only) -- each row's own true per-pixel grid
                     (real keystone/smile/clocking baked in via the raw
                     render, real spatial PSF blur across rows).
  "rectified"     (order=2 only) -- rectified onto the shared nominal
                     wn_grid (adds bilinear-interpolation error on top of
                     "native"). Both of these have real wavelength-
                     calibration error for the dispersion polynomial to
                     absorb; order=0 has consistently been the noisy,
                     less-informative case for such pipelines throughout
                     this study, so it's skipped here to save compute.
  "undistorted"   (order=2, since 2026-07-28 -- see below) -- no keystone/
                     smile/clocking/spatial PSF at all: the true spectrum
                     evaluated directly at each row's own real per-pixel
                     positions (the same 1024 native positions "native"
                     uses -- bypasses gd_render.image()/rectify() entirely,
                     and shares native's exact sampling density, not the
                     coarser shared nominal wn_grid this used before
                     2026-07-24). A baseline for whether a retrieval
                     difficulty is a fundamental RT/information-content
                     limit or something geometric distortion makes worse --
                     added 2026-07-24 to check the H2O/p_scale degeneracy
                     found near the pressure mountain in the first FPA1 run
                     (confirmed: the same dip appears here too, with zero
                     distortion, so it's a real RT/retrieval limit).
                     Matching native's real per-row grid (rather than the
                     old shared nominal grid, which sampled ~38% more
                     coarsely than the real detector for FPA3) removes a
                     second confound: native vs. undistorted now differ
                     only in whether the spectrum carries real geometric
                     distortion, not also in sampling density.

                     Originally order=0 (no real miscalibration for
                     dispersion to correct, and floating it was assumed to
                     risk spurious degeneracy with h2o_scale/p_scale --
                     see the historical note in `pipeline_orders` below).
                     Changed to order=2 2026-07-28 (KEYSTONE_SMILE_BIAS_
                     PLAN.md Sec. 11k) after the joint-retrieval study
                     found and root-caused a real, confirmed forward-model
                     self-consistency floor that specifically afflicts
                     order=0 retrievals: `gert.forward_model`'s ILS
                     convolution only uses exact (un-snapped) channel
                     centering when dispersion is present in the state
                     vector (`wn_centers is not None` triggers `gert.
                     instrument.ILS.convolve`'s `exact_center=True`);
                     without it, the retrieval's own forward model
                     evaluates against grid-snapped centers while this
                     project's truth-rendering path (`gd_render.
                     _diagonal_ils_convolve`, shared by native and
                     undistorted) always uses exact centers -- a small but
                     real, near-row-independent chi2 floor with nothing to
                     do with atmospheric retrieval quality. Empirically,
                     floating dispersion collapsed this floor (and the
                     spurious-degeneracy worry above did not materialize --
                     see the joint-retrieval diagnostic, `scripts/
                     gd_joint_undistorted_dispersion_diag.py`, and Sec. 11k
                     for the full evidence). Retrieved dispersion
                     coefficients here are still expected to sit near zero;
                     they're floated purely as a numerical workaround, not
                     because undistorted now has real calibration error to
                     correct.

Known ABSCO coverage requirement (found 2026-07-24): FPA3 (CH4_CO) needs
the ch4/h2o/co ABSCO blocks extended to ~4400 cm-1 (see
KEYSTONE_SMILE_BIAS_PLAN.md) -- this script will raise a clear ValueError
from gert.absco if that hasn't been done yet. FPA0/FPA1 have no known gaps.

Added 2026-07-25:
- Every row's retrieval now saves its complete state vector (retrieved
  value, prior, 1-sigma posterior uncertainty for every element -- not
  just the derived gas/p_surface biases) under bias["_state_full"], plus
  a convenience retrieved-minus-prior view for the nuisance elements
  under bias["_state"] (see _retrieve()). Residuals were already saved
  per row (bias["residual"] via _worker()'s return).
- --snr / --noise / --noise-seed: SNR-based Sy_inv weighting (gert.
  instrument.FlatSNR's definition) always replaces the old ad hoc 0.3%-
  of-peak hack now; --noise additionally draws one random Gaussian
  realization per pixel and adds it consistently across all three
  pipelines (see main()'s noise_arr, shared via _G so native/rectified/
  undistorted see the same noisy data, not independent draws). Default
  SNR is band-specific (DEFAULT_SNR_BY_FPA): 400/300/300/200 for FPA0-3.
- --barcode / --barcode-bars: an alternative to the along-slit composition
  scene using geocarb_gert.focalplane.barcode_scene (brightness-only
  variation, uniform composition truth) -- see Sec. 9j.

Run:  PYTHONPATH=. /path/to/analysis/env/bin/python scripts/gd_band_stress_test.py --fpa 1
Output: results/gd_band_stress_test_fpa<N>.pkl
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import time
from pathlib import Path

import numpy as np

import geosat_geometry as gg
from geocarb_gert import GEOCARB_BANDS, albedo_for, along_slit_scene as als, sample_geometries
from geocarb_gert import gd_render
from geocarb_gert import build_geocarb_instrument
from geocarb_gert.gd_polynomials import (real_wavenumber_range, xy_to_wavelength_slit,
                                         perturbed_coeffs, xy_to_wavelength_slit_assumed)
from geocarb_gert.gd_render import available_cpus, s_max
from geocarb_gert.gd_render import _diagonal_ils_convolve

import gert
from gert.forward_model import ForwardModel
from gert.instrument import ILS, SpectralWindow
from gert.instrument_config import Instrument
from gert.retrieval import GERTRetrieval, StateVector
from gert.rt_solver import SingleScatterSolver

GERT_ROOT = Path("/scratch/scrowel3_lab/gert")
REPO_ROOT = Path(__file__).resolve().parent.parent

# Default per-band SNR (--snr overrides for whichever band --fpa targets).
# O2-A is the brightest/highest-SNR channel; the CH4/CO band is the
# weakest. Matches gert.instrument.FlatSNR's definition (sigma = peak
# in-band radiance / SNR) -- see main()'s sigma_band computation.
DEFAULT_SNR_BY_FPA = {0: 400.0, 1: 300.0, 2: 300.0, 3: 200.0}

# Gases whose true column is constant along the slit (see xtrue_of_row in
# main()) -- retrieving a separate {gas}_scale for these is nearly
# degenerate with p_scale (every StateVector floats p_scale by default),
# since both change a well-mixed gas's column via the same total-air-mass
# path. Excluded from `gases` even when present in a band's molecule set;
# p_scale alone carries the surface-pressure signal instead. See module
# docstring.
WELL_MIXED_GASES = {"o2", "n2o"}

# -- globals populated in main() before the Pool is forked --
_G = {}


def _retrieve(y_dist: np.ndarray, obs_grid: np.ndarray, order: int, xtrue_row: dict):
    g = _G
    win = SpectralWindow(wn_min=g["wn_min"], wn_max=g["wn_max"],
                         ils=ILS(type="gaussian", fwhm=g["fwhm_cm"]),
                         molecules=list(g["mols"]), label=g["label"], obs_grid=obs_grid)
    inst = Instrument(windows=[win], snr=g["snr"])
    # The retrieval's prior/forward-model atmosphere is deliberately the
    # fixed slit-centre state (g["atm"]), NOT the row's true state -- the
    # along-slit deviation from this prior is exactly what each row's
    # retrieval must recover via its gas-scale/p_scale state elements.
    fm = ForwardModel(g["atm"], g["absco"], inst, g["geo"],
                      solver=SingleScatterSolver(), solar_spectrum=g["solar"])
    # SNR-based sigma (gert.instrument.FlatSNR's own definition: peak
    # in-band radiance / SNR), computed once in main() from the noiseless
    # raw image and always used here for Sy_inv -- replaces the old ad hoc
    # "0.3% of this row's peak" hack (2026-07-25, confirmed with the user
    # this should apply to every run, not just ones with --noise on).
    sigma = g["sigma_band"]
    Sy_inv = np.diag(np.full(len(y_dist), 1.0 / sigma ** 2))
    sv = StateVector.gas_scaling(prior_albedo=g["albedo"], prior_albedo_slope=np.zeros(1),
                                 gases=g["gases"],   # co2/ch4 uncertainties default to 0.10/0.20,
                                 gas_uncerts={"h2o": 0.60},  # default 0.10 is far tighter than this
                                 # scene's true h2o swing (+-56% of prior, i.e. ~5.6 prior-sigma at
                                 # 10%) -- found 2026-07-24 comparing to p_scale's default 10%, which
                                 # is a comfortable ~2.1 prior-sigma match for this scene's true
                                 # p_surface swing. 0.60 is a measured sweet spot, not a guess: FPA0
                                 # smoke tests at 10%/60%/200% gave h2o bias std of 1164/615/904 ppm
                                 # respectively -- looser isn't better past ~60%, since with little
                                 # prior regularization left the fit starts drifting on H2O's real
                                 # but weak sensitivity in this band (same failure flavor as the
                                 # o2_scale/p_scale ridge above, just softer). p_surface bias is
                                 # ~0.24 hPa std at all three settings -- this doesn't perturb that.
                                 include_dispersion=(order > 0),   # everything else 0.10 -- gas_scaling's own defaults
                                 dispersion_order=max(order, 0), dispersion_uncert=2.0)
    ret = GERTRetrieval(fm, y_dist, Sy_inv, sv, prior_albedo=g["albedo"],
                        prior_albedo_slope=np.zeros(1), analytical_jacobians=True,
                        max_iter=14, verbose=False, convergence_criterion="dx_norm", dx_tol=0.01)
    names = sv.names
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            res = ret.run()
    except (ValueError, np.linalg.LinAlgError) as e:
        nl = {gas: np.nan for gas in g["gases"]}
        nl["p_surface"] = np.nan
        nl["_state"] = {}
        nl["_state_full"] = {}
        nl["_chi2"] = np.nan
        nl["_conv"] = False
        nl["_diverged"] = str(e)
        return nl, None
    # bias = retrieved column - TRUE column at this row (not the prior) --
    # retrieved = prior_atm_value * scale; true is xtrue_row (already in the
    # reported units, see gas_units in main()).
    nl = {}
    for gas in g["gases"]:
        prior_val = float(np.mean(g["atm"].gases[gas])) * g["gas_units"][gas]
        retrieved_val = prior_val * res.x_ret[names.index(gas + "_scale")]
        nl[gas] = retrieved_val - xtrue_row[gas]
    # p_scale is always in the state vector (StateVector.gas_scaling adds it
    # by default regardless of `gases`) -- track its bias unconditionally so
    # bands that exclude o2/n2o from `gases` (see WELL_MIXED_GASES) still
    # get a surface-pressure bias metric.
    p_scale_ret = float(res.x_ret[names.index("p_scale")])
    nl["p_surface"] = g["p_surface_prior"] * p_scale_ret - xtrue_row["p_surface"]
    # Every other state-vector element (T_offset, albedo_0, albedo_slope_0,
    # and disp_a{0,1,2}_0 when order>0) has no clean along-slit "truth" to
    # diff against the way gases/p_surface do -- the scene doesn't vary
    # temperature or albedo independently, and there's no single scalar
    # "true" dispersion coefficient. Tracked as retrieved-minus-prior
    # instead (T_offset/albedo priors are 0.0/the fixed desert albedo;
    # dispersion priors are always 0.0), added 2026-07-25 so the plot
    # script can show every retrieved parameter, not just the "science"
    # ones -- e.g. dispersion coefficients wandering non-smoothly along the
    # slit would be a visible sign of them absorbing noise/degeneracy
    # rather than real wavelength-calibration error (a question raised
    # earlier about native's unusually low chi2, never directly checked).
    _tracked = {f"{gas}_scale" for gas in g["gases"]} | {"p_scale"}
    nl["_state"] = {name: float(res.x_ret[i] - prior)
                    for i, (name, prior) in enumerate(zip(names, sv.prior))
                    if name not in _tracked}
    # Complete raw state vector + posterior uncertainty (added 2026-07-25,
    # per user request): every element's retrieved value, prior, and
    # 1-sigma posterior uncertainty (Rodgers-style, from the final Ŝ =
    # (KᵀSy⁻¹K + Sa⁻¹)⁻¹), independent of the derived bias/diff quantities
    # above -- this is the raw material anything else can be recomputed
    # from, not just a convenience view.
    nl["_state_full"] = {
        "names": list(names),
        "x_ret": res.x_ret.copy(),
        "x_prior": sv.prior.copy(),
        "sigma": res.posterior_sigma(),
    }
    nl["_chi2"] = float(res.chisq_reduced)
    nl["_conv"] = res.converged
    if res.diverged:
        nl["_diverged"] = "gert: chi2 non-finite"
        return nl, None
    residual = y_dist - res.y_ret
    return nl, residual


def _worker(task):
    pipeline, order, k = task
    g = _G
    xtrue_row = {gas: float(g["xtrue_of_row"][gas][k]) for gas in g["gases"]}
    xtrue_row["p_surface"] = float(g["xtrue_of_row"]["p_surface"][k])
    try:
        if pipeline == "native":
            cols = np.arange(1024.0)
            # ASSUMED calibration (Sec. 12) when g["mismatch"] is set: this
            # is the retrieval's own position bookkeeping -- what it
            # believes each column's wavenumber is -- so it's the one that
            # should see the mismatch, not g["A"] (rendered with the REAL
            # mapping in main(), untouched here).
            if g["mismatch"] is not None:
                lam_row, _ = xy_to_wavelength_slit_assumed(g["fpa"], cols, np.full(1024, float(k)), g["mismatch"])
            else:
                lam_row, _ = xy_to_wavelength_slit(g["fpa"], cols, np.full(1024, float(k)))
            nu_row = 1e4 / lam_row
            y_dist = g["A"][k, :]
            # gert's y/y_ret is always in ascending-wavelength (descending
            # wavenumber) order. Column order happens to already match that
            # for FPA2 (its real dispersion runs the other way), which is
            # why no reversal was ever needed there -- but FPA1's real
            # dispersion direction is the opposite (found 2026-07-24: every
            # native-pipeline retrieval failed catastrophically, even at
            # the slit centre where truth == prior, until this was fixed).
            # Check explicitly rather than assume either direction.
            if nu_row[0] < nu_row[-1]:   # ascending wavenumber -> wrong order, flip
                nu_row = nu_row[::-1]
                y_dist = y_dist[::-1]
            nl, resid = _retrieve(y_dist, nu_row, order, xtrue_row)
            nu_out = nu_row
        elif pipeline == "rectified":
            row = g["Rimg"][k, :]
            valid = ~np.isnan(row)
            if valid.sum() < 0.5 * len(row):
                return (pipeline, order, k), {"bias": None, "residual": None, "nu": None, "n_bad": int((~valid).sum())}
            nu_valid = g["wn_grid"][valid]
            y_dist = row[valid][::-1]
            nl, resid = _retrieve(y_dist, nu_valid, order, xtrue_row)
            nu_out = nu_valid[::-1]
            nl["_n_bad"] = int((~valid).sum())
        else:  # "undistorted" -- no keystone/smile/clocking, no spatial PSF:
            # the true hi-res spectrum evaluated directly at this row's own
            # real per-pixel positions -- the SAME 1024 native positions
            # "native" uses above, not the separate shared nominal wn_grid
            # this used to use. Bypasses gd_render.image()/rectify() (no
            # geometric distortion in the spectrum), but now (2026-07-24)
            # also matches native's real per-row sampling exactly: the old
            # shared wn_grid had ~38% coarser density than the real detector
            # (e.g. FPA3: 8.36 vs 11.56 points/cm-1) -- an unwanted sampling-
            # density confound on top of the distortion difference this
            # baseline is meant to isolate. native and undistorted now share
            # an identical grid per row; the only remaining difference is
            # whether the spectrum itself carries real keystone/smile/PSF
            # distortion (native, via gd_render.image()) or not (undistorted,
            # evaluated directly from the truth lookup table).
            cols = np.arange(1024.0)
            # Same ASSUMED-calibration substitution as native above --
            # undistorted's own spectrum (S_row/eta_row below) is still
            # looked up at the REAL x_km_of_row position (that's the "no
            # geometric distortion" baseline this pipeline exists to be);
            # only its belief about each column's wavenumber changes.
            if g["mismatch"] is not None:
                lam_row, _ = xy_to_wavelength_slit_assumed(g["fpa"], cols, np.full(1024, float(k)), g["mismatch"])
            else:
                lam_row, _ = xy_to_wavelength_slit(g["fpa"], cols, np.full(1024, float(k)))
            nu_row = 1e4 / lam_row
            reverse = nu_row[0] < nu_row[-1]   # match native's convention (see above)
            if reverse:
                nu_row = nu_row[::-1]
            eta_row = g["x_km_of_row"][k] / als.SLIT_HALF_KM
            S_row = np.asarray(g["radiance"](np.full(len(nu_row), eta_row)), dtype=float)
            y_dist = _diagonal_ils_convolve(g["wn_hires"], S_row, nu_row, g["ils"])
            if g["noise_arr"] is not None:
                # Same per-(row,column) noise realization native sees via
                # g["A"][k,:] (added once to the raw image in main(), not
                # redrawn here) -- undistorted now shares native's exact
                # grid (see the module docstring's 2026-07-24 sampling-
                # density fix), so this keeps them pixel-exact-comparable:
                # the only remaining difference is real geometric distortion,
                # not also an independent noise draw. Column order matches
                # g["A"]'s raw (unreversed) storage, so the same reversal
                # flag applied to nu_row above applies here too.
                noise_row = g["noise_arr"][k, :]
                if reverse:
                    noise_row = noise_row[::-1]
                y_dist = y_dist + noise_row
            nl, resid = _retrieve(y_dist, nu_row, order, xtrue_row)
            nu_out = nu_row
    except Exception as e:   # noqa: BLE001 -- keep the sweep alive
        nl = {gas: np.nan for gas in g["gases"]}
        nl["p_surface"] = np.nan
        nl["_state"] = {}
        nl["_state_full"] = {}
        nl["_chi2"] = np.nan
        nl["_conv"] = False
        nl["_diverged"] = f"{type(e).__name__}: {e}"
        resid, nu_out = None, None
    return (pipeline, order, k), {"bias": nl, "residual": resid, "nu": nu_out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fpa", type=int, required=True, choices=[0, 1, 2, 3])
    ap.add_argument("--row-step", type=int, default=1, help="subsample rows (1 = every row)")
    ap.add_argument("--n-lookup-samples", type=int, default=400,
                    help="along-slit truth-atmosphere lookup-table density (default resolves "
                         "the ~10 km hot spots with ~3 samples across their FWHM)")
    ap.add_argument("--n-workers", type=int, default=None, help="default: all available cores")
    ap.add_argument("--out-tag", type=str, default=None,
                    help="append _<tag> to the output filename (e.g. --out-tag smoketest) so a "
                         "test run never overwrites a real run's results/gd_band_stress_test_fpa<N>.pkl")
    ap.add_argument("--uniform", action="store_true",
                    help="disable along-slit composition/pressure variation -- every row's truth "
                         "atmosphere is the fixed slit-centre prior, isolating pure geometric-"
                         "distortion bias for comparison against a normal (varying) run of the same "
                         "band. Output filename gets a _uniform suffix automatically. Mutually "
                         "exclusive with --barcode.")
    ap.add_argument("--barcode", action="store_true",
                    help="use a diffuser-style barcode illumination (geocarb_gert.focalplane."
                         "barcode_scene) instead of the along-slit composition scene: same "
                         "spectral shape at every row (uniform composition truth, like --uniform), "
                         "but brightness alternates in --barcode-bars bars -- probes keystone/PSF "
                         "row-mixing at sharp spatial transitions (see KEYSTONE_SMILE_BIAS_PLAN.md "
                         "Sec. 9j, which found a real, narrowly-localized bias spike near bar "
                         "boundaries). Mutually exclusive with --uniform.")
    ap.add_argument("--barcode-bars", type=int, default=32,
                    help="number of alternating brightness bars for --barcode (default 32, matching "
                         "Sec. 9j's original test). Brightness pattern is a fixed [1.0, 0.2] repeat.")
    ap.add_argument("--snr", type=float, default=None,
                    help="signal-to-noise ratio for this band's Sy_inv weighting (gert.instrument."
                         "FlatSNR definition: sigma = peak in-band radiance / SNR) and, if --noise is "
                         "also given, the actual noise realization added to the simulated data. "
                         "Default: DEFAULT_SNR_BY_FPA[fpa] (400/300/300/200 for FPA0-3).")
    ap.add_argument("--noise", action="store_true",
                    help="add a random Gaussian noise realization (sigma from --snr) to the raw "
                         "rendered image once, shared consistently across all three pipelines (see "
                         "main()'s noise_arr and _worker()'s undistorted branch) -- without this "
                         "flag the simulated data stays deterministic/noiseless as in every run "
                         "before 2026-07-25. Output filename gets a _noise suffix.")
    ap.add_argument("--noise-seed", type=int, default=0,
                    help="seed for --noise's random Gaussian draw (reproducible by default).")
    ap.add_argument("--mismatch-mode", type=str, default="none",
                    choices=["none", "wavelength", "slit", "both"],
                    help="calibration-knowledge mismatch (KEYSTONE_SMILE_BIAS_PLAN.md Sec. 12): "
                         "give the retrieval's position bookkeeping (native/undistorted's nu_row) "
                         "an ASSUMED GD mapping that differs from the REAL one gd_render.image() "
                         "renders truth with, via geocarb_gert.gd_polynomials.perturbed_coeffs(). "
                         "'wavelength' perturbs only the smile/dispersion map (partially "
                         "correctable via this pipeline's own dispersion-coefficient nuisance "
                         "parameters, order=2); 'slit' perturbs only the keystone/slit-position "
                         "map (NOT correctable -- no equivalent nuisance parameter exists in "
                         "StateVector.gas_scaling()); 'both' perturbs both independently. Default "
                         "'none' reproduces every earlier run exactly (perfect calibration "
                         "knowledge). NOTE: 'rectified' is dropped from this run's pipeline list "
                         "whenever mismatch-mode != none -- gd_render.rectify() only has a real-"
                         "calibration inverse mapping today; see Sec. 12d.")
    ap.add_argument("--mismatch-wn-bias-cm1", type=float, default=0.0,
                    help="deterministic wavenumber bias [cm-1] (on-orbit thermal-drift stand-in).")
    ap.add_argument("--mismatch-wn-noise-cm1", type=float, default=0.0,
                    help="RMS of a random smooth wavenumber perturbation [cm-1] over the full "
                         "detector (ground-test fit-noise stand-in). Redrawn only when "
                         "--mismatch-seed changes.")
    ap.add_argument("--mismatch-slit-bias-km", type=float, default=0.0,
                    help="deterministic along-slit position bias [km].")
    ap.add_argument("--mismatch-slit-noise-km", type=float, default=0.0,
                    help="RMS of a random smooth along-slit position perturbation [km].")
    ap.add_argument("--mismatch-seed", type=int, default=0,
                    help="seed for the --mismatch-*-noise-* draws (reproducible by default; "
                         "wavelength and slit noise fields use seed and seed+1 respectively).")
    args = ap.parse_args()
    if args.uniform and args.barcode:
        ap.error("--uniform and --barcode are mutually exclusive (both force uniform composition "
                "truth already; choose one along-slit radiance pattern)")
    FPA = args.fpa
    snr = args.snr if args.snr is not None else DEFAULT_SNR_BY_FPA[FPA]

    # ASSUMED (mismatched) calibration -- see Sec. 12 / --mismatch-mode's
    # help text. None (perfect calibration knowledge, every earlier run's
    # behavior) unless a mismatch mode was requested. Built once here, not
    # per-row -- it's the retrieval's fixed *belief* about the geometry,
    # not something that varies row to row.
    mismatch = None
    if args.mismatch_mode != "none":
        use_wn = args.mismatch_mode in ("wavelength", "both")
        use_slit = args.mismatch_mode in ("slit", "both")
        mismatch = perturbed_coeffs(
            FPA,
            wn_bias_cm1=args.mismatch_wn_bias_cm1 if use_wn else 0.0,
            wn_noise_rms_cm1=args.mismatch_wn_noise_cm1 if use_wn else 0.0,
            slit_bias_km=args.mismatch_slit_bias_km if use_slit else 0.0,
            slit_noise_rms_km=args.mismatch_slit_noise_km if use_slit else 0.0,
            seed=args.mismatch_seed)
        print(f"calibration mismatch active: mode={args.mismatch_mode} "
             f"wn_bias={args.mismatch_wn_bias_cm1 if use_wn else 0.0}cm-1 "
             f"wn_noise_rms={args.mismatch_wn_noise_cm1 if use_wn else 0.0}cm-1 "
             f"slit_bias={args.mismatch_slit_bias_km if use_slit else 0.0}km "
             f"slit_noise_rms={args.mismatch_slit_noise_km if use_slit else 0.0}km "
             f"seed={args.mismatch_seed} -- 'rectified' pipeline dropped this run (Sec. 12d)",
             flush=True)

    label, wn_min_nom, wn_max_nom, mols, R = GEOCARB_BANDS[FPA]
    mols = list(mols)
    wn_min, wn_max = real_wavenumber_range(FPA, margin_cm1=10.0)
    print(f"FPA{FPA} ({label}): molecules={mols}  wn=[{wn_min:.2f},{wn_max:.2f}]", flush=True)

    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))

    wn_c = 0.5 * (wn_min_nom + wn_max_nom)
    fwhm_cm = wn_c / float(R)
    wide_win = SpectralWindow(wn_min=wn_min, wn_max=wn_max, ils=ILS(type="gaussian", fwhm=fwhm_cm),
                              molecules=mols, label=label, hires_spacing=0.01, channels_per_fwhm=3)
    wide_inst = Instrument(windows=[wide_win], snr=snr)
    albedo = albedo_for(wide_inst, "desert")

    # Retrieval prior/forward-model atmosphere: the fixed slit-centre state
    # (x_km=0), used for every row -- deliberately NOT each row's true
    # state (see _retrieve's docstring comment). Computed here (moved up
    # 2026-07-25) since --barcode needs it to build its single hi-res
    # spectrum before the along-slit-vs-barcode branch below.
    atm_center = als.atmosphere_at(0.0)

    t0 = time.time()
    if args.barcode:
        # Diffuser-style barcode illumination (geocarb_gert.focalplane.
        # barcode_scene): same spectral shape at every row (from atm_center,
        # the same prior atmosphere used everywhere else), only brightness
        # varies -- see Sec. 9j. brightness pattern: fixed [1.0, 0.2] repeat,
        # resized to --barcode-bars (handles odd counts by truncation).
        from geocarb_gert.focalplane import barcode_scene
        fm_center = ForwardModel(atm_center, absco, wide_inst, geo,
                                 solver=SingleScatterSolver(), solar_spectrum=solar)
        res_center = fm_center.run(albedo=albedo, albedo_slope=[0.0])
        S_center = np.asarray(res_center.I_hires[0], dtype=float)
        brightness = np.resize([1.0, 0.2], args.barcode_bars)
        radiance = barcode_scene(S_center, brightness=brightness, widths=None, softness=0.0)
        wn_hires = wide_win.wn_hires
        print(f"barcode scene built ({args.barcode_bars} bars, {time.time()-t0:.1f}s)", flush=True)
    else:
        wn_hires, radiance = als.build_lookup_radiance(
            absco, wide_inst, geo, solar, albedo,
            n_samples=args.n_lookup_samples, n_workers=args.n_workers, uniform=args.uniform)
        n_samples_built = 1 if args.uniform else args.n_lookup_samples
        print(f"lookup table built ({n_samples_built} samples, {time.time()-t0:.1f}s)", flush=True)

    A = gd_render.image(FPA, wn_hires, radiance, wide_win.ils, spatial_psf_fwhm_px=1.5,
                        n_workers=args.n_workers)
    print(f"raw render done ({time.time()-t0:.1f}s)", flush=True)

    # SNR-based sigma (gert.instrument.FlatSNR's definition: peak in-band
    # radiance / SNR), computed once from the noiseless raw image so it
    # doesn't shift depending on whether --noise then perturbs individual
    # pixels. Always used for the retrieval's Sy_inv (see _retrieve()) --
    # 2026-07-25, replaces the old ad hoc 0.3%-of-peak hack for every run,
    # not just ones with --noise on (confirmed with the user).
    sigma_band = float(np.max(np.abs(A))) / snr
    noise_arr = None
    if args.noise:
        # Added ONCE to the raw per-pixel image, not independently per
        # pipeline -- physically there's one noisy detector readout; native
        # reads it directly (g["A"][k,:]), rectified interpolates it
        # (gd_render.rectify below operates on this same noisy A), and
        # undistorted (which bypasses gd_render.image()/rectify() entirely)
        # looks up this same per-(row,column) noise value in _worker() so
        # it stays pixel-exact-comparable with native, not just same-
        # statistics -- see the module docstring's 2026-07-24 sampling-
        # density-confound fix, which this mirrors for noise.
        rng = np.random.default_rng(args.noise_seed)
        noise_arr = rng.normal(0.0, sigma_band, size=A.shape)
        A = A + noise_arr
        print(f"noise added (SNR={snr:.0f}, sigma={sigma_band:.4g}, seed={args.noise_seed})", flush=True)

    nominal_inst = build_geocarb_instrument()
    nominal_win = nominal_inst.windows[FPA]
    wn_grid = nominal_win.wn_instrument
    sm = s_max(FPA)
    s_grid = np.linspace(-sm, sm, 1024)
    Rimg = gd_render.rectify(FPA, A, s_grid, wn_grid)
    print(f"rectified {Rimg.shape} ({time.time()-t0:.1f}s)", flush=True)

    # -- along-slit truth at each row's real slit position, for scoring bias --
    # (row index -> real s -> eta -> x_km -> true gas column at that position)
    cols_center = np.full(1024, 512.0)
    _, s_of_row = xy_to_wavelength_slit(FPA, cols_center, np.arange(1024.0))
    x_km_of_row = (s_of_row / sm) * als.SLIT_HALF_KM
    # x_km_of_row (above) is each row's real physical slit position -- always
    # kept as-is, since it drives the geometric distortion (keystone/smile
    # row-crossing) and is what plots use for the x-axis. xtrue_x_km (below)
    # is the position used to evaluate TRUTH for scoring bias -- in --uniform
    # or --barcode mode this is pinned to 0 (the prior's own position) for
    # every row, so truth == prior everywhere and bias reflects pure
    # geometric distortion (+ real brightness-transition contamination for
    # --barcode), not composition-tracking error -- barcode only varies
    # brightness, never composition, so its truth scoring is identical to
    # --uniform's. Both are independent of the along-slit-vs-barcode
    # radiance construction above, which affects rendering.
    xtrue_x_km = np.zeros(1024) if (args.uniform or args.barcode) else x_km_of_row
    # h2o's true/prior comparison must be on the same basis _retrieve() uses
    # for every gas (mean of the full vertical profile, since h2o_scale
    # multiplies the whole prior profile shape) -- not the surface VMR
    # directly, which would introduce a spurious constant offset since h2o
    # decays sharply with height. True and prior share the same scale
    # height (atmosphere_at always uses the default), so the ratio of
    # surface VMRs equals the ratio of profile means exactly.
    h2o_mean_prior_ppm = float(np.mean(atm_center.gases["h2o"])) * 1e6
    h2o_surface_ratio = als.h2o_surface_vmr(xtrue_x_km) / als.h2o_surface_vmr(0.0)
    xtrue_of_row = {
        "co2": als.xco2_ppm(xtrue_x_km),
        "ch4": als.xch4_ppb(xtrue_x_km),
        "co": als.xco_ppb(xtrue_x_km),
        "h2o": h2o_mean_prior_ppm * h2o_surface_ratio,   # ppm, profile-mean basis (see above)
        "o2": np.full(1024, 0.2095 * 1e6),                # well-mixed, not varied along slit
        "n2o": np.full(1024, 330.0),                      # well-mixed, not varied along slit (ppb)
        "p_surface": als.p_surface_hpa(xtrue_x_km),       # hPa -- not a "gas", tracked via p_scale
    }
    gas_units = {"co2": 1e6, "ch4": 1e9, "co": 1e9, "h2o": 1e6, "o2": 1e6, "n2o": 1e9}
    p_surface_prior = float(als.p_surface_hpa(np.array([0.0]))[0])

    # h2o always last (cosmetic only); well-mixed gases (o2, n2o) excluded --
    # see WELL_MIXED_GASES.
    gases = [m for m in mols if m != "h2o" and m not in WELL_MIXED_GASES] + ["h2o"]

    rows = list(range(0, 1024, args.row_step))
    n_workers = args.n_workers if args.n_workers is not None else available_cpus()

    out = {}
    # "undistorted": no keystone/smile/clocking/spatial-PSF -- the true
    # spectrum at each row's exact intended slit position, evaluated
    # directly (see _worker). Isolates whether a retrieval difficulty
    # (e.g. the H2O/p_scale degeneracy near the pressure mountain, found
    # 2026-07-24 in the native/rectified FPA1 run) is a fundamental RT/
    # information-content limit -- present even with a perfect instrument
    # -- or something geometric distortion is making worse.
    #
    # Dispersion order: native/rectified have real keystone/smile-driven
    # wavelength-calibration error for the dispersion polynomial to absorb
    # (order=2 only -- order=0 has consistently been the noisy, less-
    # informative case throughout this study for pipelines with real
    # distortion, not worth the extra compute). undistorted ALSO gets
    # order=2 as of 2026-07-28 (previously order=0) -- not because it has
    # real calibration error to correct (it doesn't), but because floating
    # dispersion is the only way to make `gert.forward_model` evaluate its
    # ILS convolution with exact (un-snapped) channel centering, matching
    # this project's truth-rendering convention (`gd_render.
    # _diagonal_ils_convolve`, used to build undistorted's own measurement,
    # always uses exact centers). Without it, order=0 carries a small,
    # confirmed, spurious chi2 floor that has nothing to do with retrieval
    # quality -- see the module docstring's "undistorted" bullet and
    # KEYSTONE_SMILE_BIAS_PLAN.md Sec. 11k for the full root-cause and
    # empirical evidence (a joint-retrieval diagnostic showed this floating
    # dispersion collapses the floor rather than creating the spurious
    # h2o_scale/p_scale degeneracy originally feared -- see the historical
    # comment this replaced). Retrieved dispersion coefficients are still
    # expected to sit near zero.
    pipeline_orders = {"native": [2], "rectified": [2], "undistorted": [2]}
    if mismatch is not None:
        # gd_render.rectify() (already run above, Rimg) only has a REAL-
        # calibration inverse mapping (wavelength_slit_to_xy) -- there's no
        # assumed-inverse counterpart yet (Sec. 12d), so rectified's
        # position bookkeeping can't honor `mismatch` today. Dropping it
        # rather than silently running it on the real grid, which would
        # make it look artificially immune to a mismatch it was never
        # actually exposed to.
        del pipeline_orders["rectified"]
    tasks = [(pipeline, order, k) for pipeline, orders in pipeline_orders.items()
            for order in orders for k in rows]
    print(f"{len(tasks)} retrievals ({len(rows)} rows x "
         f"{sum(len(v) for v in pipeline_orders.values())} pipeline/order combos), "
         f"{n_workers} workers, gases={gases}", flush=True)

    _G.update(dict(fpa=FPA, A=A, Rimg=Rimg, wn_grid=wn_grid, wn_hires=wn_hires, radiance=radiance,
                   ils=wide_win.ils, wn_min=wn_min, wn_max=wn_max, snr=snr, sigma_band=sigma_band,
                   noise_arr=noise_arr, mismatch=mismatch,
                   fwhm_cm=fwhm_cm, mols=mols, label=label, atm=atm_center, absco=absco,
                   geo=geo, solar=solar, albedo=albedo, gases=gases, p_surface_prior=p_surface_prior,
                   xtrue_of_row=xtrue_of_row, gas_units=gas_units, x_km_of_row=x_km_of_row))

    ctx = mp.get_context("fork")
    n_done = 0
    with ctx.Pool(n_workers) as pool:
        for key, val in pool.imap_unordered(_worker, tasks, chunksize=4):
            out[key] = val
            n_done += 1
            if n_done % 200 == 0:
                print(f"  {n_done}/{len(tasks)} done ({time.time()-t0:.0f}s)", flush=True)

    print(f"all done ({time.time()-t0:.0f}s)", flush=True)
    n_diverged = sum(1 for v in out.values() if v["bias"] is not None and v["bias"].get("_diverged"))
    n_offdet = sum(1 for v in out.values() if v["bias"] is None)
    print(f"diverged: {n_diverged}, off-detector (rectified only): {n_offdet}, total: {len(out)}")

    mode_suffix = "_uniform" if args.uniform else ("_barcode" if args.barcode else "")
    noise_suffix = "_noise" if args.noise else ""
    mismatch_suffix = f"_mismatch{args.mismatch_mode}" if args.mismatch_mode != "none" else ""
    tag_suffix = f"_{args.out_tag}" if args.out_tag else ""
    out_path = (REPO_ROOT / "results" /
               f"gd_band_stress_test_fpa{FPA}{mode_suffix}{noise_suffix}{mismatch_suffix}{tag_suffix}.pkl")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"out": out, "rows": rows, "wn_grid": wn_grid, "s_grid": s_grid,
                    "A_raw": A, "R_rectified": Rimg, "FPA": FPA, "gases": gases,
                    "uniform": args.uniform, "barcode": args.barcode,
                    "barcode_bars": args.barcode_bars if args.barcode else None,
                    "snr": snr, "sigma_band": sigma_band, "noise": args.noise,
                    "noise_seed": args.noise_seed if args.noise else None,
                    "mismatch_mode": args.mismatch_mode,
                    "mismatch_wn_bias_cm1": args.mismatch_wn_bias_cm1,
                    "mismatch_wn_noise_cm1": args.mismatch_wn_noise_cm1,
                    "mismatch_slit_bias_km": args.mismatch_slit_bias_km,
                    "mismatch_slit_noise_km": args.mismatch_slit_noise_km,
                    "mismatch_seed": args.mismatch_seed if args.mismatch_mode != "none" else None,
                    "xtrue_of_row": xtrue_of_row, "x_km_of_row": x_km_of_row}, f)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
