"""Solve ONE window jointly across one or more bands (docs/MULTIBAND_PLAN.md phases 3-4).

The hi-res path of ``gd_joint_block_retrieve._solve_window`` generalised to N bands
(no aerosol for now, per the plan): one joint ``StateSpec`` (``multiband.MultiBandState``)
shared by all bands with independent per-band albedo rows; each band's forward model
and analytic Jacobian are the EXISTING single-band ones (``build_forward_state`` /
``jac.linearize``) evaluated on that band's view; the joint forward/Jacobian are those
stacked. With ONE band and ``--anchor-mode nominal`` this reproduces the single-band
driver's hi-res solve (the phase-3 gate).

Usage (real polynomials only -- keystone is never turned off):
  python gd_multiband_window.py --fpas 2 --rows 2:25-37 --anchor-mode nominal \
      --free co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo --anchor-workers 4
  python gd_multiband_window.py --fpas 0,2 --tile 12 --free p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import geosat_geometry as gg  # noqa: E402
import gert  # noqa: E402
import gd_joint_block_retrieve as gjr  # noqa: E402
from geocarb_gert import GEOCARB_BANDS, along_slit_scene as als, sample_geometries  # noqa: E402
from geocarb_gert import jacobians as jac  # noqa: E402
from geocarb_gert.joint_state import (build_forward_state, default_pad_for_psf, gauss_newton_state,  # noqa: E402
                                       pixel_density_bin_centers, state_spec_from_scene)
from geocarb_gert.multiband import BandRef, joint_spec_from_scene  # noqa: E402
from geocarb_gert.multiband_geometry import (build_window_tiles_multiband, band_tables,  # noqa: E402
                                              joint_anchor_eta_range, eta_to_row)

_NO_AEROSOL = ("amplitude_aerosol", "height_aerosol", "thickness_aerosol")


def load_inputs():
    """absco, solar, geo, atm_center -- same construction as the driver's main()."""
    block = gg.geocarb_demo(verbose=False)["blocks"][0]
    _, _, geo = sample_geometries(block, n=1, seed=0)[0]
    absco = gert.ABSCOTable.load_all(str(gjr.GERT_ROOT / "input/absco/absco.h5"))
    solar = gert.SolarSpectrum.load(str(gjr.GERT_ROOT / "input/solar/solar.h5"))
    return absco, solar, geo, als.atmosphere_at(0.0)


def solve_window_multiband(rows_by_fpa: dict, free, *, inputs=None, prior_fields="structural",
                           g_ratio=None, anchor_density=4, anchor_mode="cover", solver="xrtm",
                           state_interp=None, prior_form=None, gamma=None, anchor_workers=1,
                           psf_fwhm_px=1.5, verbose=True, hook=None, aerosol=False,
                           bin_centers_override=None, noise_seed=None):
    """Joint hi-res solve of one window.

    rows_by_fpa : {fpa: (row_lo, row_hi)}; the FIRST entry is the reference band (sets G).
    anchor_mode : "nominal" -- anchors at the reference band's rows +/- pad at the centre
                  column (exactly the single-band driver); "cover" -- a uniform eta grid
                  spanning every pixel of every band's padded rows (keystone included).
    bin_centers_override (2026-09-23, user: geometry-config override for cross-run bin
        alignment -- see scripts/gd_export_geometry_config.py's own docstring): an explicit eta
        array replacing `pixel_density_bin_centers`'s own computation, e.g. so this run's bins
        land at the SAME eta positions as a single-band run given the same geometry config. `G`
        is then `len(bin_centers_override)`, not `width_ref / g_ratio`.
    """
    cfg = gjr._defaults
    g_ratio = cfg.g_ratio if g_ratio is None else g_ratio
    state_interp = cfg.state_interp if state_interp is None else state_interp
    prior_form = cfg.prior_form if prior_form is None else prior_form
    gamma = cfg.gamma if gamma is None else gamma
    free = tuple(free)
    fpas = list(rows_by_fpa)
    ref = fpas[0]
    absco, solar, geo, atm_center = inputs or load_inputs()
    retrieval_pad = default_pad_for_psf(psf_fwhm_px)

    # ---- per-band optics/instrument ------------------------------------------------
    B = {}
    for f in fpas:
        wide_win, wide_inst, albedo = gjr.band_basics(f, atm_center, absco, geo, solar)
        lo, hi = rows_by_fpa[f]
        B[f] = dict(fpa=f, label=GEOCARB_BANDS[f][0], wide_inst=wide_inst, albedo=albedo,
                    wn_hires=wide_inst.windows[0].wn_hires, ils=wide_inst.windows[0].ils,
                    rows=np.arange(lo, hi + 1),
                    spectrum=gjr._make_state_spectrum(absco, wide_inst, geo, solar, albedo, solver=solver),
                    spectrum_jac=jac.make_spectrum_jac(absco, wide_inst, geo, solar, albedo, solver=solver))
        B[f]["eta_all"] = np.stack([gjr._eta_of(f, np.arange(1024.0), np.full(1024, float(i)))
                                    for i in B[f]["rows"]])

    # ---- shared eta bins and anchors ------------------------------------------------
    width_ref = len(B[ref]["rows"])
    if bin_centers_override is not None:
        bin_centers = np.asarray(bin_centers_override, dtype=float)
        G = bin_centers.size
    else:
        G = max(2, int(round(width_ref / g_ratio)))
        bin_centers = pixel_density_bin_centers(np.concatenate([B[f]["eta_all"].ravel() for f in fpas]), G)
    anchor_ext = max(gjr.PAD, retrieval_pad)
    if anchor_mode == "nominal":
        lo, hi = rows_by_fpa[ref]
        a_lo, a_hi = max(0, lo - anchor_ext), min(gjr.ROW_MAX_IDX, hi + anchor_ext)
        anchor_rows = np.arange(a_lo, a_hi + 1e-9, 1.0 / anchor_density)
        anchor_etas = np.sort(gjr._eta_of(ref, np.full(len(anchor_rows), 512.0), anchor_rows.astype(float)))
    elif anchor_mode == "cover":
        class _W:                      # minimal MBWindow-shaped holder for joint_anchor_eta_range
            rows = rows_by_fpa
        e_lo, e_hi = joint_anchor_eta_range(_W, anchor_ext)
        step = band_tables(ref)["spacing"] / anchor_density
        anchor_etas = np.arange(e_lo, e_hi + 0.5 * step, step)
    else:
        raise ValueError(f"anchor_mode must be 'nominal' or 'cover', got {anchor_mode!r}")

    # ---- truth per band (the retrieval's own forward model at a fully frozen state) ----
    strip = () if aerosol else _NO_AEROSOL     # aerosol arm: the truth scene carries the aerosol layer
    truth_surface = {k: v for k, v in als.SURFACE_FIELDS.items() if k not in strip}
    y_true, Sy, noise_check = {}, {}, {}
    for f in fpas:
        b = B[f]
        truth_spec = state_spec_from_scene(anchor_etas, free=(), fields=als.STATE_FIELDS, band_label=b["label"],
                                           surface_fields=truth_surface, surface_positions=anchor_etas,
                                           kinds=gjr.ROW_KINDS, sigmas=gjr.ROW_SIGMAS)
        fwd_t = build_forward_state(f, b["rows"], anchor_etas, truth_spec, b["spectrum"], b["wn_hires"], b["ils"],
                                    pad=retrieval_pad, state_interp=state_interp, n_workers=anchor_workers,
                                    spatial_psf_fwhm_px=psf_fwhm_px)
        y_true[f] = fwd_t(np.array([]))
        sigma = gjr.geocarb_noise_model(f).sigma([y_true[f]], [None])
        Sy[f] = 1.0 / np.maximum(sigma, gjr._SIGMA_FLOOR) ** 2
        if noise_seed is not None:
            # signal-dependent shot noise (sigma^2 = N0^2 + N1|I|, the band's own noise model) drawn on
            # top of the noiseless representative truth; Sy above stays the noiseless-signal sigma, as in
            # the single-band driver. Seed keyed on (seed, fpa, row range) so the SAME band/tile gets the
            # bit-identical realization in the single-band driver (paired comparison).
            rng = np.random.default_rng([int(noise_seed), int(f), int(rows_by_fpa[f][0]), int(rows_by_fpa[f][1])])
            nz = rng.normal(0.0, sigma)
            noise_check[f] = (int(nz.size), float(nz.sum()))
            y_true[f] = y_true[f] + nz

    # ---- joint state -----------------------------------------------------------------
    free_set = set(free)
    imperfect = als.PRIOR_FIELD_SETS[prior_fields]
    imperfect_surface = als.SURFACE_PRIOR_FIELD_SETS.get(prior_fields, als.SURFACE_FIELDS)
    prior_fields_resolved = {n: (fn if n in free_set else als.STATE_FIELDS[n]) for n, fn in imperfect.items()}
    surface_resolved = {n: (fn if n in free_set else als.SURFACE_FIELDS[n]) for n, fn in imperfect_surface.items()
                        if n not in strip}
    row_positions = {n: anchor_etas for n in prior_fields_resolved if n not in free_set}
    if aerosol:     # aerosol is gas-like (broad scale): free aerosol rows ride the coarse eta bins, as in the single-band driver
        for n in ("amplitude_aerosol", "height_aerosol"):
            if n in free_set:
                row_positions[n] = bin_centers
    bands = [BandRef(f, B[f]["label"]) for f in fpas]
    mb = joint_spec_from_scene(bin_centers, bands, free=free, corr_length=None, prior_form=prior_form,
                               uniform=False, fields=prior_fields_resolved, surface_fields=surface_resolved,
                               surface_positions=anchor_etas, row_positions=row_positions,
                               prior_anchor_density=None, row_sub_bin_anomaly=None, kinds=gjr.ROW_KINDS,
                               gamma=gamma, row_bounds=gjr.ROW_BOUNDS, sigmas=gjr.ROW_SIGMAS)

    # ---- per-band forwards / Jacobians on the views -------------------------------------
    pools = []          # one mutable single-element holder per band (so a hung pool can be replaced)
    forwards, lins = [], []
    try:
        for i, f in enumerate(fpas):
            b, view = B[f], mb.view(i)
            forwards.append(build_forward_state(f, b["rows"], anchor_etas, view, b["spectrum"], b["wn_hires"],
                                                b["ils"], pad=retrieval_pad, state_interp=state_interp,
                                                n_workers=anchor_workers, spatial_psf_fwhm_px=psf_fwhm_px))
            holder = [jac.LinearizePool(anchor_workers, b["spectrum_jac"]) if anchor_workers > 1 else None]
            pools.append(holder)

            def lin(x, f=f, b=b, view=view, holder=holder):
                # 2026-09-21: sweep tile 11 (job 1945175_11) died with a PoolHangError -- one run_L stalled from
                # the START of an iteration for the whole 3600 s guard, with no OOM (real memory ~40 GB of a
                # 160 GB limit) and healthy sibling tasks. Cause unknown; the cheap mitigation is to rebuild that
                # band's pool and repeat the (deterministic, side-effect-free) linearization instead of losing
                # the window's hours of work.
                for attempt in range(3):
                    try:
                        return jac.linearize(f, b["rows"], anchor_etas, view, b["spectrum_jac"], b["wn_hires"],
                                             b["ils"], x, pad=gjr.PAD, state_interp=state_interp,
                                             n_workers=anchor_workers, pool=holder[0])
                    except jac.PoolHangError as e:
                        if holder[0] is None or attempt == 2:
                            raise
                        print(f"  PoolHangError in FPA{f} linearize (attempt {attempt + 1}/3): rebuilding that "
                              f"band's pool and retrying -- {str(e)[:120]}", flush=True)
                        holder[0].close()
                        holder[0] = jac.LinearizePool(anchor_workers, b["spectrum_jac"])
            lins.append(lin)
        fwd = mb.stack_forward(forwards)
        jac_joint = mb.stack_linearize(lins)
        y = np.concatenate([y_true[f].ravel() for f in fpas])
        Sy_inv = mb.stack_Sy_inv_diag([Sy[f] for f in fpas])
        if hook is not None:       # e.g. the finite-difference Jacobian check: hand over the built problem
            return hook(dict(mb=mb, forward=fwd, linearize=jac_joint, y=y, Sy_inv=Sy_inv, fpas=fpas,
                             B=B, bin_centers=bin_centers, anchor_etas=anchor_etas, G=G,
                             n_data_by_band={f: y_true[f].size for f in fpas}))
        t0 = time.time()
        label = "+".join(f"FPA{f}[{rows_by_fpa[f][0]}-{rows_by_fpa[f][1]}]" for f in fpas)
        x, S_ret, avk = gauss_newton_state(fwd, y, mb.joint, Sy_inv, label=label, verbose=verbose,
                                           jacobian_fn=jac_joint, return_cov=True, return_avk=True)
        resid = y - fwd(x)
    finally:
        for holder in pools:
            if holder[0] is not None:
                holder[0].close()
    sizes = [y_true[f].size for f in fpas]
    splits = np.cumsum([0] + sizes)
    return dict(fpas=fpas, rows_by_fpa={f: tuple(rows_by_fpa[f]) for f in fpas}, G=G, bin_centers=bin_centers,
                anchor_etas=anchor_etas, anchor_mode=anchor_mode, free=free, solver=solver, aerosol=aerosol, noise_seed=noise_seed, noise_check=noise_check,
                joint=mb.joint.snapshot(x, resid=resid, resid_rms=float(np.sqrt(np.mean(resid ** 2))),
                                        cov=S_ret, avk=avk, dof=float(np.trace(avk))),
                x=x, resid=resid, resid_rms=float(np.sqrt(np.mean(resid ** 2))),
                resid_rms_by_band={f: float(np.sqrt(np.mean(resid[splits[i]:splits[i + 1]] ** 2)))
                                   for i, f in enumerate(fpas)},
                n_data_by_band=dict(zip(fpas, sizes)), n_free=int(np.asarray(x).size),
                t_solve=time.time() - t0, eta_convention="slit_image_long_wl")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fpas", required=True, help="comma list, e.g. 2 or 0,2 (first = reference band)")
    ap.add_argument("--rows", default=None, help="explicit rows per band, e.g. 2:25-37 or 0:100-120,2:98-118")
    ap.add_argument("--tile", type=int, default=None, help="index into build_window_tiles_multiband(fpas, ..., overlap=2)")
    ap.add_argument("--free", required=True)
    ap.add_argument("--anchor-mode", default="cover", choices=["nominal", "cover"])
    ap.add_argument("--anchor-density", type=int, default=4)
    ap.add_argument("--anchor-workers", type=int, default=1)
    ap.add_argument("--solver", default=None, choices=["single_scatter", "xrtm"],
                    help="default (omit): 'xrtm' if --aerosol, else 'single_scatter' (2026-09-23, user: "
                         "'when aerosol is not included, the RT model should default to single_scatter' -- "
                         "confirmed equivalent there, see gd_joint_block_retrieve.py's own --solver help).")
    ap.add_argument("--g-ratio", type=float, default=None, help="bins per row ratio (default: config; the production sweeps use 1)")
    ap.add_argument("--prior-fields", default="structural", choices=sorted(als.PRIOR_FIELD_SETS),
                    help="named prior set (als.PRIOR_FIELD_SETS); 'realistic' = ACOS-like climatological gases + "
                         "reanalysis-like T/p/h2o/aerosol, never exactly the truth (2026-09-21)")
    ap.add_argument("--overlap", type=int, default=2)
    ap.add_argument("--aerosol-type", default=None,
                    help="aerosol type, realistic per-band optical properties (geocarb_gert/aerosol_mie.py: Mie at each FPA's centre; "
                         "AOD defined at O2-A): smoke | dust | sulfate | sea_salt | cloud_water. Required with --aerosol (no default). Legacy two-slot registry "
                         "smoke (FPA1-3 share values, AOD defined at 1.6 um), also reachable as registry_<type>, kept so earlier runs "
                         "stay reproducible. Truth and retrieval use the same set. Sets GEOCARB_AEROSOL_TYPE so workers inherit it.")
    ap.add_argument("--check-aerosol", action="store_true",
                    help="print the per-band aerosol optical properties that this command line selects, then exit "
                         "(no inputs are loaded)")
    ap.add_argument("--noise-seed", type=int, default=None,
                    help="add signal-dependent shot noise (the band's geocarb_noise_model) to the observed spectrum; "
                         "default off. Realization keyed on (seed, fpa, rows) -- identical in the single-band driver.")
    ap.add_argument("--aerosol", action="store_true",
                    help="truth scene and retrieval include the Gaussian aerosol layer (rows in --free are retrieved, "
                         "the rest frozen at truth); default off = no aerosol")
    ap.add_argument("--geometry-config", default=None,
                    help="JSON from gd_export_geometry_config.py, OR hand-authored with the same schema -- "
                         "{\"tiles\": [{\"tile\": i, \"eta_lo\":..., \"eta_hi\":..., \"bin_centers\": [...], "
                         "\"rows_by_fpa\": {\"0\": [lo,hi], ...}}]} -- gives this tile's bin_centers, and (if "
                         "--rows/--tile alone weren't enough) its row ranges too: 'rows_by_fpa' if present, else "
                         "derived per band from eta_lo/eta_hi via multiband_geometry.eta_to_row (2026-09-23, user: "
                         "\"the ability to specify values explicitly instead of just pointing to a reference pkl "
                         "file\" -- a config entry with only eta_lo/eta_hi/bin_centers, no rows_by_fpa, works too; "
                         "an optional per-tile 'overlap' integer then symmetrically pads that derived range at BOTH "
                         "edges -- adjacent tiles' resulting coverage is NOT checked for overlap with each other). "
                         "Use with --tile to pick the entry (by its 'tile' field, not necessarily this run's own "
                         "tile numbering); --rows still overrides rows if given alongside.")
    ap.add_argument("--out", default=None, help="output pickle (default under results/realistic_prior/multiband/)")
    a = ap.parse_args()
    fpas = [int(t) for t in a.fpas.split(",")]
    geom_bin_centers = None
    geom_rows = None
    if a.geometry_config:
        import json
        geom = json.load(open(a.geometry_config))
        if a.tile is None:
            ap.error("--geometry-config requires --tile (to pick which entry to use)")
        entry = next((t for t in geom["tiles"] if t["tile"] == a.tile), None)
        if entry is None:
            ap.error(f"--geometry-config has no entry for tile {a.tile}")
        geom_bin_centers = entry["bin_centers"]
        rbf = entry.get("rows_by_fpa")
        if rbf is not None:
            geom_rows = {f: tuple(rbf[str(f)]) for f in fpas}
        else:
            # 2026-09-23 (user: "let's add the 'overlap' field to symmetrically pad the rows"):
            # an optional per-tile integer, added/subtracted at BOTH edges of the eta-derived row
            # range -- unlike build_window_tiles_multiband's own overlap (interior boundaries
            # only, never the outermost edge of the whole tile set), since a single hand-authored
            # entry has no neighbor context to know which edges are "interior." If adjacent
            # tiles' padded ranges end up covering the same rows, nothing here detects or
            # prevents it -- see this flag's own --geometry-config help for what that means.
            ov = int(entry.get("overlap", 0))
            geom_rows = {f: (max(0, int(eta_to_row(f, entry["eta_lo"])) - ov),
                             min(gjr.ROW_MAX_IDX, int(eta_to_row(f, entry["eta_hi"])) + ov)) for f in fpas}
            print(f"geometry-config: tile {a.tile} has no rows_by_fpa -- derived from eta bounds via eta_to_row"
                 + (f", padded by {ov} rows at both edges" if ov else ""))
        print(f"geometry-config: tile {a.tile} from {a.geometry_config} "
             f"({len(geom_bin_centers)} bins, eta [{entry['eta_lo']:.4f},{entry['eta_hi']:.4f}], rows {geom_rows})")
    if a.rows:
        rows = {int(k): tuple(int(v) for v in r.split("-")) for k, r in (t.split(":") for t in a.rows.split(","))}
        rows = {f: rows[f] for f in fpas}
    elif geom_rows is not None:
        rows = geom_rows
    elif a.tile is not None:
        tiles = build_window_tiles_multiband(fpas, gjr.MIN_WINDOW, 1.0, a.overlap)
        rows = dict(tiles[a.tile].rows)
        print(f"tile {a.tile}/{len(tiles)}: eta [{tiles[a.tile].eta_lo:.4f},{tiles[a.tile].eta_hi:.4f}] rows {rows}")
    else:
        ap.error("give --rows or --tile")
    if a.solver is None:
        a.solver = "xrtm" if a.aerosol else "single_scatter"
        print(f"--solver not given: defaulting to '{a.solver}' ({'aerosol' if a.aerosol else 'no aerosol'})")
    from geocarb_gert.aerosol_defaults import (band_props_for_wavelength, resolve_aerosol_type,
                                              validate_aerosol_type)
    if a.aerosol and not a.aerosol_type:
        ap.error("--aerosol requires --aerosol-type (2026-09-25: no silent default). Choose smoke | dust | sulfate | "
                 "sea_salt | cloud_water for realistic per-band properties, or registry_<type> (e.g. registry_smoke) to "
                 "reproduce the earlier two-slot runs.")
    if a.aerosol_type:
        validate_aerosol_type(a.aerosol_type)
        if not a.aerosol:
            ap.error("--aerosol-type has no effect without --aerosol")
        os.environ["GEOCARB_AEROSOL_TYPE"] = a.aerosol_type
    if a.aerosol:
        # what the forward model will actually use per band: (ssa, g, tau relative to the 1.6 um reference)
        from geocarb_gert.aerosol_defaults import amplitude_reference_um
        print(f"aerosol type: {resolve_aerosol_type()} (amplitude_aerosol / scene AOD defined at {amplitude_reference_um():.3f} um)", flush=True)
        for f in fpas:
            wn_lo, wn_hi = GEOCARB_BANDS[f][1], GEOCARB_BANDS[f][2]
            wl = 1e4 / (0.5 * (wn_lo + wn_hi))
            ssa, g, ts = band_props_for_wavelength(None, wl)
            print(f"  FPA{f} ({wl:.3f} um): ssa={ssa:.3f} g={g:.3f} tau/tau(ref)={ts:.3f}", flush=True)
    if a.check_aerosol:
        return
    res = solve_window_multiband(rows, a.free.split(","), noise_seed=a.noise_seed, g_ratio=a.g_ratio, anchor_density=a.anchor_density,
                                 anchor_mode=a.anchor_mode, solver=a.solver, anchor_workers=a.anchor_workers, prior_fields=a.prior_fields,
                                 aerosol=a.aerosol, bin_centers_override=geom_bin_centers)
    name = f"mb_fpa{'-'.join(map(str, fpas))}_" + "_".join(f"r{f}-{rows[f][0]}-{rows[f][1]}" for f in fpas) \
        + f"_free-{'-'.join(t.split('_')[0] for t in a.free.split(','))}_{a.anchor_mode}_g{a.g_ratio if a.g_ratio is not None else 'cfg'}_etaslit" \
        + ("" if a.prior_fields == "structural" else f"_prior-{a.prior_fields}") + ("_aero" if a.aerosol else "") \
        + ("_geomcfg" if a.geometry_config else "") + (f"_{a.aerosol_type}" if (a.aerosol and a.aerosol_type) else "") + (f"_noise{a.noise_seed}" if a.noise_seed is not None else "")
    from geocarb_gert.aerosol_defaults import resolve_aerosol_type
    res["aerosol_type"] = resolve_aerosol_type() if a.aerosol else None
    out = Path(a.out) if a.out else REPO / "results/realistic_prior/multiband" / f"{name}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(res, fh)
    print(f"resid_rms {res['resid_rms']:.3e} (by band {res['resid_rms_by_band']}), n_free {res['n_free']}, "
          f"solve {res['t_solve']:.0f}s\nsaved {out}")


if __name__ == "__main__":
    main()
