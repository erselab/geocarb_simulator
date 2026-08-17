#!/usr/bin/env python3
"""Does RETRIEVING the nuisance state remove the residual that STEPPING it
creates? The direct test of the joint block's dominant error term.

Background. The hi-res forward model assigns pixels to anchors by nearest
(`nearest_bin_scene`), so every quantity it carries is a step function in
eta at anchor spacing h -- a first-order representation error ~(1/4)|f'|h.
Measured on the realistic scene, the post-fit residual tracks the NUISANCE
state's own version of that error, not CO2's: per-row residual RMS
correlates +0.94 with surface pressure's nearest-anchor error across the
topographic depression (rows 329-527) and +0.89 with H2O's globally, versus
+0.59 for CO2 and +0.38 for keystone. `--anchor-density 4` shrinks h and
duly cuts the residual 4.13x, but the mechanism survives intact (+0.890
correlation there afterwards) because those quantities are still *stepped*
rather than *retrieved* -- they sit at local truth, exactly right at each
anchor and wrong in between, with no free parameter able to absorb the
difference.

This script asks the obvious follow-up: give the solver that free parameter.
Same window, same data, same anchors -- only which rows of the state vector
are free changes:

    co2                 today's configuration (CO2 only)
    co2+p_surface       surface pressure unfrozen
    co2+p_surface+h2o   the two measured worst offenders unfrozen

Every configuration is one `StateSpec.unfreeze()` call, not a code path --
that is the point of `geocarb_gert.joint_state`. CO2 itself is an ordinary
row and could equally be frozen.

Honest expectations, stated before running so the result cannot be
rationalised afterwards. Freeing a parameter must reduce the RESIDUAL (more
freedom always fits the data at least as well). Whether it improves the CO2
BIAS is the real question and is NOT obvious: the nuisance state is
currently pinned to *perfect local truth*, so unfreezing it can only move it
AWAY from the correct value, and the extra freedom may be absorbed by
CO2-nuisance degeneracy instead. A residual that drops while the bias
worsens would be a clean demonstration that residual and bias measure
different things -- which the anchor-density result already hinted at
(gratio3: residual 4.16x better, max|bias| slightly worse).

Run:  PYTHONPATH=. python3 scripts/gd_joint_block_freeze_test.py
      PYTHONPATH=. python3 scripts/gd_joint_block_freeze_test.py \\
        --row-min 460 --row-max 482 --n-workers 6
Output: prints a comparison table; --plot writes
        plots/joint_block/gd_joint_block_freeze_test_fpa<F>_row<lo>-<hi>.png
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import gd_test as gdt  # noqa: E402
from gd_joint_block_retrieve import FPA, GERT_ROOT, _eta_of, band_basics  # noqa: E402
from gd_joint_block_diagnostics import pixel_density_bin_centers  # noqa: E402

import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
from geocarb_gert import (along_slit_scene as als, gd_render,  # noqa: E402
                          nearest_bin_scene, sample_geometries)
from geocarb_gert.joint_state import gauss_newton_state, state_spec_from_scene  # noqa: E402

from gert.forward_model import ForwardModel  # noqa: E402
from gert.rt_solver import SingleScatterSolver  # noqa: E402

PAD = 4


def make_state_spectrum_fn(absco, wide_inst, geo, solar, albedo):
    """spectrum(params_dict) -> hi-res radiance.

    Builds the atmosphere directly from state parameters via
    `als.atmosphere_from_params`, rather than going through gert's
    `StateVector.gas_scaling`, which only knows how to scale gases and would
    reintroduce exactly the CO2-is-special asymmetry this is removing.
    Surface pressure and H2O are ordinary keys here.

    Albedo stays fixed: it is not part of `AtmosphericProfile` (it goes to
    `ForwardModel.run`), so making it a free row needs the combined
    state+surface record noted in `als.STATE_FIELDS`. Flagged, not silently
    ignored.
    """
    def spectrum(params: dict) -> np.ndarray:
        atm = als.atmosphere_from_params(**params)
        fm = ForwardModel(atm, absco, wide_inst, geo, solver=SingleScatterSolver(),
                          solar_spectrum=solar)
        res = fm.run(albedo=np.array([albedo]), albedo_slope=np.zeros(1))
        return np.asarray(res.I_hires[0], dtype=float)

    return spectrum


def build_forward_state(fpa, rows_win, anchor_rows, spec, spectrum, wn_hires, ils, pad=PAD):
    """forward(x) -> raveled sub-image, with EVERY state row interpolated
    from its own positions onto the anchors (`StateSpec.interp_to`).

    Caches each anchor's spectrum keyed on that anchor's full parameter
    vector, so an iteration only re-runs RT for anchors whose state actually
    moved -- the multi-parameter generalization of the original per-bin CO2
    cache.
    """
    anchor_etas = np.sort(_eta_of(fpa, np.full(len(anchor_rows), 512.0),
                                  np.asarray(anchor_rows, dtype=float)))
    G_eff = len(anchor_etas)
    names = [p.name for p in spec.params]
    cache_key = np.full((G_eff, len(names)), np.nan)
    cache_S = [None] * G_eff

    def forward(x):
        vals = spec.interp_to(anchor_etas, x)          # {name: (G_eff,)}
        key = np.column_stack([vals[n] for n in names])
        for g in range(G_eff):
            if cache_S[g] is None or not np.array_equal(key[g], cache_key[g]):
                cache_S[g] = spectrum({n: float(vals[n][g]) for n in names})
                cache_key[g] = key[g]
        radiance = nearest_bin_scene(anchor_etas, cache_S)
        return gd_render.predict_neighborhood(fpa, rows_win, wn_hires, radiance,
                                              ils, pad=pad).ravel()

    return forward


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row-min", type=int, default=460)
    ap.add_argument("--row-max", type=int, default=482,
                    help="default window sits inside the surface-pressure depression, "
                         "where the nuisance-stepping residual is largest")
    ap.add_argument("--n-bins", type=int, default=None, help="default: one bin per row")
    ap.add_argument("--corr-length", type=float, default=0.02,
                    help="prior correlation length in eta, applied to every row")
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    rows_win = np.arange(args.row_min, args.row_max + 1)
    G = args.n_bins or len(rows_win)

    print(f"Building realistic-scene FPA{FPA} band (gert at {GERT_ROOT}) ...", flush=True)
    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(GERT_ROOT / "input/solar/solar.h5"))
    atm_center = als.atmosphere_at(0.0)
    gdt._G.update(dict(atm=atm_center, absco=absco, geo=geo, solar=solar))
    snr = gdt.DEFAULT_SNR_BY_FPA[FPA]
    band = gdt._band_setup(FPA, atm_center, absco, geo, solar, snr, 400,
                           args.n_workers, False, False, 32, False, 0)
    _, wide_inst, albedo = band_basics(FPA, atm_center, absco, geo, solar)
    wn_hires, ils = band["wn_hires"], band["ils"]
    print("done.\n", flush=True)

    cols = np.arange(1024.0)
    eta_all = np.stack([_eta_of(FPA, cols, np.full(1024, float(i))) for i in rows_win])
    bin_centers = pixel_density_bin_centers(eta_all.ravel(), G)
    anchor_rows = np.arange(max(0, args.row_min - PAD), min(1023, args.row_max + PAD) + 1)

    y_true = band["A"][rows_win, :].ravel()
    Sy_inv_diag = np.full(y_true.size, 1.0 / float(np.mean(np.abs(y_true))) ** 2)
    spectrum = make_state_spectrum_fn(absco, wide_inst, geo, solar, albedo)

    eta_rows = _eta_of(FPA, np.full(len(rows_win), 512.0), rows_win.astype(float))
    true_co2_rows = als.xco2_ppm(eta_rows * als.SLIT_HALF_KM)

    configs = [("co2", ("co2_ppm",)),
               ("co2+p_surface", ("co2_ppm", "p_surface_hpa")),
               ("co2+p_surface+h2o", ("co2_ppm", "p_surface_hpa", "h2o_surface_vmr"))]

    print(f"window rows {args.row_min}-{args.row_max} ({len(rows_win)} rows), G={G} bins, "
          f"{len(anchor_rows)} anchors, corr_length={args.corr_length} eta\n")
    results = {}
    for label, free in configs:
        spec = state_spec_from_scene(bin_centers, free=free, corr_length=args.corr_length)
        fwd = build_forward_state(FPA, rows_win, anchor_rows, spec, spectrum, wn_hires, ils)
        t0 = time.time()
        x = gauss_newton_state(fwd, y_true, spec, Sy_inv_diag, label=label)
        resid = y_true - fwd(x)
        vals = spec.unpack(x)
        co2_bin = vals["co2_ppm"]
        co2_rows = np.interp(eta_rows, bin_centers, co2_bin)
        bias = co2_rows - true_co2_rows
        # Standing rule: save the ENTIRE state vector (free AND frozen rows,
        # with their priors, sigmas and correlation lengths) and the FULL
        # residual field -- never just the summary scalars. `spec.snapshot`
        # makes the record self-describing, so it can be interpreted without
        # the code that produced it.
        results[label] = dict(
            snapshot=spec.snapshot(x, label=label, free=list(free)),
            resid=resid.copy(),
            resid_rms=float(np.sqrt(np.mean(resid ** 2))),
            bias=bias.copy(), bias_rms=float(np.sqrt(np.mean(bias ** 2))),
            max_bias=float(np.max(np.abs(bias))),
            n_free=spec.n_free, vals=vals, t=time.time() - t0)
        print(f"  -> {label}: n_free={spec.n_free}  ({results[label]['t']:.0f}s)\n")

    base = results["co2"]
    print(f"{'config':22s} {'n_free':>7s} {'residRMS':>11s} {'vs co2':>8s} "
          f"{'biasRMS ppm':>12s} {'max|bias|':>10s}")
    for label, _ in configs:
        r = results[label]
        print(f"{label:22s} {r['n_free']:7d} {r['resid_rms']:11.4e} "
              f"{base['resid_rms']/r['resid_rms']:7.2f}x {r['bias_rms']:12.4f} "
              f"{r['max_bias']:10.4f}")

    # How far did the freed nuisance rows move from their (perfect) truth?
    print("\nfreed nuisance rows, fractional departure from local truth:")
    for label, free in configs[1:]:
        v = results[label]["vals"]
        for name in free:
            if name == "co2_ppm":
                continue
            truth = state_spec_from_scene(bin_centers)[name].prior
            dev = (v[name] - truth) / truth
            print(f"  {label:22s} {name:18s} mean={dev.mean():+.3e}  max|dev|={np.abs(dev).max():.3e}")

    out_pkl = (REPO_ROOT / "results" /
               f"gd_joint_block_freeze_test_fpa{FPA}_row{args.row_min}-{args.row_max}.pkl")
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(dict(results=results, fpa=FPA, row_lo=args.row_min, row_hi=args.row_max,
                         rows_win=rows_win, bin_centers=bin_centers,
                         anchor_rows=anchor_rows, G=G, pad=PAD,
                         corr_length=args.corr_length, y_true=y_true,
                         true_co2_rows=true_co2_rows, eta_rows=eta_rows,
                         Sy_inv_diag=Sy_inv_diag), f)
    print(f"\nsaved {out_pkl}  ({out_pkl.stat().st_size/1e6:.1f} MB)")

    # Degeneracy: does CO2 error move OPPOSITE the freed nuisance row?
    print("\nCO2-nuisance degeneracy (per-bin correlation):")
    base_co2 = results["co2"]["vals"]["co2_ppm"]
    for label, free in configs[1:]:
        v = results[label]["vals"]
        dco2 = v["co2_ppm"] - base_co2
        for name in free:
            if name == "co2_ppm":
                continue
            truth = state_spec_from_scene(bin_centers)[name].prior
            dev = (v[name] - truth) / truth
            if np.std(dev) > 0 and np.std(dco2) > 0:
                r = float(np.corrcoef(dco2 / base_co2, dev)[0, 1])
                print(f"  {label:22s} corr(dCO2/CO2, d{name}/{name}) = {r:+.3f}"
                      f"   slope = {np.polyfit(dev, dco2/base_co2, 1)[0]:+.2f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "serif", "font.size": 10.5})
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.axhline(0, color="black", lw=0.6)
        for (label, _), c in zip(configs, ["0.25", "tab:blue", "tab:orange"]):
            r = results[label]
            ax.plot(rows_win, r["bias"], lw=1.1, color=c,
                    label=f"{label} (resid={r['resid_rms']:.3e}, bias rms={r['bias_rms']:.4f} ppm)")
        ax.set_xlabel("detector row"); ax.set_ylabel("CO2 bias [ppm]")
        ax.set_title(f"FPA{FPA} rows {args.row_min}-{args.row_max}: effect of unfreezing "
                     f"nuisance state", fontsize=11)
        ax.legend(fontsize=8)
        out = REPO_ROOT / "plots" / "joint_block" / \
            f"gd_joint_block_freeze_test_fpa{FPA}_row{args.row_min}-{args.row_max}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
