"""The one place that turns (atm_params, surface) into a `gert.ForwardModel`
run.

**Why this module exists** (2026-09-09). Before this, geocarb_simulator had
9 independent call sites that each built their own `gert.ForwardModel(...)`
and called `.run(...)`, hand-extracting atmosphere/surface/aerosol state on
their own -- see `docs/PROJECT_STATUS.md` Sec.12 for the full inventory.
Six of those sites were near-byte-for-byte duplicates of the same ~15-line
aerosol-kwarg-building block, independently pasted in as the aerosol rows
(`tau_aerosol`/`height_aerosol`) were added -- which is exactly how a real
bug (the truth-rendering path in `gd_per_row_retrieve.py::_band_setup` never threading
aerosol through at all) hid for so long: threading a new physical
capability through the model meant remembering to touch every single site,
and one was missed.

`simulate_spectrum` is the forward-only entry point every live call site is
being migrated to (`docs/PROJECT_STATUS.md` Sec.12's migration log tracks
which sites are done). `spectrum_and_jacobian` is its analytic-Jacobian
counterpart, relocated from `jacobians.make_spectrum_jac` (which is now a
thin wrapper around it, kept for any external caller relying on the old
name).

Adding a NEW physical capability from here on means touching this module
once, not N call sites.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import gert
from gert.forward_model import ForwardModel
from gert.rt_solver import SingleScatterSolver, XRTMSolver

from . import along_slit_scene as als
from .aerosol_defaults import (aerosol_scalars_for, aerosol_band_props,
                               band_slot_for_wavelength_um, resolve_aerosol_type,
                               band_props_for_wavelength)

#: 2026-09-15 (Phase 2 of the XRTM integration plan): the one place a
#: `solver="single_scatter"|"xrtm"` string resolves to an actual gert
#: `RTSolver` instance -- every `simulate_spectrum`/`spectrum_and_jacobian`
#: call site goes through this, so adding a third solver later means
#: touching this dict once, not every call site. `n_streams=2` (user,
#: 2026-09-15) overrides gert's own "recommended" default of 8 -- a
#: deliberate speed/accuracy tradeoff for whole-slit-sweep cost, not yet
#: independently validated at this project's own scale (see Phase 5 of
#: the plan) -- confirm radiance accuracy before trusting it blindly.
#: XRTM needs NO separate phase-function-moment config here: `XRTMSolver.
#: solve()` computes its own Legendre moments internally from the SAME
#: `g_aerosol`/`ssa_aerosol` kwargs `_build_aerosol_kwargs` already builds
#: for `SingleScatterSolver` (via `gert.legendre.combined_layer_coef`,
#: gert-internal) -- the `xrtm_config.py` module this project's own plan
#: anticipated needing turned out to be unnecessary.
def _ensure_xrtm_importable():
    """`XRTMSolver._load_xrtm()` does a bare `import xrtm` -- but the
    compiled extension lives at `gert/xrtm/interfaces/xrtm.so`, not
    `gert/xrtm/` itself (that directory has no `__init__.py`, so without
    its own `interfaces` subdir on `sys.path`, `import xrtm` silently
    resolves to an EMPTY namespace package instead of raising ImportError
    -- confirmed directly, 2026-09-15: `hasattr(xrtm, 'xrtm')` is `False`,
    not an exception, so this failure mode is easy to miss). Every
    existing sbatch script's `PYTHONPATH` only has the `gert` repo root,
    not this subpath -- add it here, once, rather than requiring every
    caller to remember XRTMSolver's own docstring instruction
    (`export PYTHONPATH=$PWD/interfaces:$PYTHONPATH`).
    """
    import os
    import sys
    import gert as _gert_pkg
    interfaces_dir = os.path.join(os.path.dirname(os.path.dirname(_gert_pkg.__file__)),
                                  "xrtm", "interfaces")
    if interfaces_dir not in sys.path:
        sys.path.insert(0, interfaces_dir)


def _build_solver(solver: str, jacobians: bool):
    """2026-09-15 (user, after the first cost benchmark): `method=
    'eig_add'` even at its OWN minimum `n_streams=2` is NOT "the 2-stream
    method" -- it's still the full general discrete-ordinates eigenvalue-
    decomposition machinery, just solving a smaller matrix.
    `method='two_stream'` is a SEPARATE, purpose-built implementation
    (the same one `gert.rt_solver.LSISolver` uses as its own "fast"
    solver) -- measured directly: 3.27s vs `eig_add(n_streams=2)`'s
    105.28s per anchor at this band's ~12353 hi-res wavenumber points, a
    ~32x difference from switching METHOD, not the streams number. Made
    the default here.
    """
    if solver == "single_scatter":
        return SingleScatterSolver(jacobians=jacobians)
    if solver == "xrtm":
        _ensure_xrtm_importable()
        return XRTMSolver(method="two_stream", jacobians=jacobians)
    raise ValueError(f"unknown solver {solver!r} -- expected 'single_scatter' or 'xrtm'")


@dataclass
class SpectrumResult:
    """What every call site actually wants, plus the raw result for
    Jacobian dispatch / debugging."""
    I_hires: np.ndarray            # res.I_hires[0] -- the hi-res spectrum
    result: "gert.ForwardResult"   # the full raw ForwardModel result
    atm: object                    # the AtmosphericProfile actually used
    aerosol_kwargs: dict           # what was passed to fm.run for aerosol (empty if none)


def aerosol_band_for(wide_inst, aerosol_type: str) -> tuple[float, float, float]:
    """`(ssa, g, tau_scale)` for the band of `wide_inst`'s first window
    (2026-09-21: per-band GERT scalars; O2-A uses registry slot 0 with tau
    scaled by qext_norm[0]/qext_norm[1], every longer-wavelength band slot 1
    and scale 1 -- so FPA1-3 results are unchanged)."""
    wn = np.asarray(wide_inst.windows[0].wn_hires, dtype=float)
    # 2026-09-25: 'smoke_mie' = per-FPA Mie properties (aerosol_mie.py), all four bands distinct
    return band_props_for_wavelength(aerosol_type, 1e4 / float(wn.mean()))


def _build_aerosol_kwargs(surface: Optional[dict], n_wn: int, geo,
                          aerosol_type: str, band_props=None) -> dict:
    """`{}` when neither `tau_aerosol` nor `height_aerosol` is present in
    `surface` -- matches every existing call site's `.get(...)` convention:
    `amplitude_aerosol=None` is treated the same as `gert.ForwardModel.run`'s
    own documented "aerosol term omitted" (`tau_aerosol=None`) -- zero
    behavior change by default.

    2026-09-15 (user: "introduce the Gaussian parameters into the state
    vector instead of a tau_aerosol"): `gert`'s own `ForwardModel.run`
    interface is UNCHANGED -- it still wants a scalar column `tau_aerosol`.
    This function is where geocarb_simulator's own `amplitude_aerosol`/
    `thickness_aerosol` state gets converted to that column value (the
    Gaussian integral, `tau = amplitude * sigma * sqrt(2*pi)`, matching
    `forward_model.py`'s own normalized-weight convention for
    `aerosol_profile_shape="gaussian"`) -- confined to this one adapter
    function rather than pushed into the shared `gert` library.
    `thickness_aerosol` (now the real per-anchor state value, not the old
    fixed `als.AEROSOL_THICKNESS_PA` constant) is passed straight through
    unchanged.
    """
    amp_aer = (surface or {}).get("amplitude_aerosol")
    height_aer = (surface or {}).get("height_aerosol")
    thickness_aer = (surface or {}).get("thickness_aerosol", als.AEROSOL_THICKNESS_PA)
    if amp_aer is None:
        return {}
    if band_props is None:
        ssa, g, qext_norm = aerosol_scalars_for(aerosol_type)
        tau_scale = 1.0
    else:
        ssa, g, tau_scale = band_props
        qext_norm = 1.0
    tau_aer = float(amp_aer) * float(thickness_aer) * np.sqrt(2.0 * np.pi) * tau_scale
    p_aer_val = als.aerosol_phase_hg(g, np.cos(geo.scattering_angle))
    return dict(
        tau_aerosol=tau_aer,
        height_aerosol=height_aer,
        aerosol_profile_shape="gaussian",
        thickness_aerosol=thickness_aer,
        ssa_aerosol=[np.full(n_wn, ssa)],
        g_aerosol=[g],
        qext_aerosol=[np.full(n_wn, qext_norm)],
        P_aerosol=[np.full(n_wn, p_aer_val)],
    )


def simulate_spectrum(atm_params: dict, surface: Optional[dict], absco, wide_inst,
                      geo, solar, jacobians: bool = False,
                      aerosol_type: str | None = None,
                      solver: str = "xrtm") -> SpectrumResult:
    """Build the atmosphere from `atm_params` (the kwargs
    `along_slit_scene.atmosphere_from_params` takes) and run one
    `gert.ForwardModel` pass.

    `surface` follows the convention already used at every migrated call
    site: `{"albedo": ..., "albedo_slope": ..., "amplitude_aerosol": ...,
    "height_aerosol": ..., "thickness_aerosol": ...}` (the Gaussian
    vertical-profile parameterization -- see `_build_aerosol_kwargs`'s own
    docstring; `tau_aerosol` was a free row here until 2026-09-15, now
    only a derived diagnostic), any key optional/absent -> that physics is
    omitted, exactly like today. `None` is equivalent to `{}`.

    **Albedo fallback is the CALLER's responsibility.** Every existing call
    site resolves its own module-level fallback scalar (`surface.get(
    "albedo", albedo)`) before this point -- this function's own
    `surface.get("albedo", 0.0)` is a last-resort default that real
    callers should never actually hit, not a substitute for that
    resolution. Passing `surface` without an explicit `"albedo"` key when
    a non-zero fallback is wanted is a caller bug, not something this
    function can silently do the right thing about (it doesn't have
    access to whatever scene-specific fallback the caller would use).
    """
    atm = als.atmosphere_from_params(**atm_params)
    surface = surface or {}
    alb = float(surface.get("albedo", 0.0))
    slope = float(surface.get("albedo_slope", 0.0))
    n_wn = len(wide_inst.windows[0].wn_hires)
    aer_kwargs = _build_aerosol_kwargs(surface, n_wn, geo, aerosol_type,
                                       band_props=aerosol_band_for(wide_inst, aerosol_type))

    fm = ForwardModel(atm, absco, wide_inst, geo,
                      solver=_build_solver(solver, jacobians),
                      solar_spectrum=solar)
    res = fm.run(albedo=np.array([alb]), albedo_slope=np.array([slope]),
                jacobians=jacobians, **aer_kwargs)
    return SpectrumResult(I_hires=np.asarray(res.I_hires[0], dtype=float),
                          result=res, atm=atm, aerosol_kwargs=aer_kwargs)


def spectrum_and_jacobian(atm_params: dict, rows, absco, wide_inst, geo, solar,
                          surface: Optional[dict] = None,
                          aerosol_type: str | None = None,
                          solver: str = "xrtm"):
    """``(S_hires, {row: dS/d(param)})`` -- the analytic-Jacobian
    counterpart of `simulate_spectrum`. Row dispatch is delegated to
    `geocarb_gert.jacobians` (`SURFACE_ROW_JACOBIAN`, `gas_dI_dparam`,
    `p_surface_dI_dparam`, `t_offset_dI_dparam`,
    `height_aerosol_dI_dparam`) -- unchanged from `make_spectrum_jac`,
    just relocated to call through `simulate_spectrum` for the nominal
    state instead of building its own `ForwardModel`.

    `solver` (2026-09-15, Phase 2/3 of the XRTM integration plan) picks
    between two DIFFERENT aerosol-height/thickness Jacobian compositions,
    not just a different forward-model call: `SingleScatterSolver`'s
    `tau_abv`-based RT-level-FD (`height_aerosol_dI_dparam`,
    `thickness_aerosol_dI_dparam`'s AOD-scaling-only version) vs XRTM's
    exact analytic `K_aer_lay_hires` composition (`*_xrtm` variants) --
    see both functions' own docstrings for why one solver's `K_ssa_lay==0`
    makes the latter degenerate there.
    """
    from . import jacobians as jac  # local import: jacobians imports this module too

    sr = simulate_spectrum(atm_params, surface, absco, wide_inst, geo, solar,
                           jacobians=True, aerosol_type=aerosol_type, solver=solver)
    res, atm = sr.result, sr.atm
    surface = surface or {}
    alb = float(surface.get("albedo", 0.0))
    slope = float(surface.get("albedo_slope", 0.0))
    amp_aer = surface.get("amplitude_aerosol")
    height_aer = surface.get("height_aerosol")
    thickness_aer = surface.get("thickness_aerosol", als.AEROSOL_THICKNESS_PA)
    # 2026-09-15: `tau_aer` here is the SAME derived column value
    # `_build_aerosol_kwargs` computed for the nominal-state `sr` above --
    # recomputed (not re-derived from `res`) since `p_surface_dI_dparam`'s
    # own RT-fallback FD path (jacobians.py) needs it as a plain kwarg to
    # rebuild a perturbed `ForwardModel.run()` call, same as before.
    a_ssa, a_g, a_scale = aerosol_band_for(wide_inst, aerosol_type)
    tau_aer = (float(amp_aer) * float(thickness_aer) * np.sqrt(2.0 * np.pi) * a_scale
              if amp_aer is not None else None)

    d = {}
    for row in rows:
        if row == "height_aerosol":
            d[row] = (jac.height_aerosol_dI_dparam_xrtm(res, atm, float(height_aer), float(thickness_aer))
                      if solver == "xrtm" else
                      jac.height_aerosol_dI_dparam(res, atm, float(height_aer),
                                                   thickness_aerosol=thickness_aer))
        elif row == "amplitude_aerosol":
            d[row] = jac.amplitude_aerosol_dI_dparam(res, float(thickness_aer)) * a_scale
        elif row == "thickness_aerosol":
            d[row] = (jac.thickness_aerosol_dI_dparam_xrtm(
                        res, atm, float(height_aer), float(thickness_aer), float(amp_aer) * a_scale)
                      if solver == "xrtm" else
                      jac.thickness_aerosol_dI_dparam(
                        res, atm, float(height_aer), float(thickness_aer), float(amp_aer) * a_scale))
        elif row in jac.SURFACE_ROW_JACOBIAN:
            d[row] = jac.surface_dI_dparam(res, row)
        elif row == "p_surface_hpa":
            d[row] = jac.p_surface_dI_dparam(res, atm_params, absco=absco, wide_inst=wide_inst,
                                             geo=geo, solar=solar, alb=alb, slope=slope,
                                             tau_aer=tau_aer, height_aer=height_aer,
                                             thickness_aer=thickness_aer,
                                             aer_props=(a_ssa, a_g), surface=surface,
                                             solver=solver, aerosol_type=aerosol_type)
        elif row == "t_offset_k":
            d[row] = jac.t_offset_dI_dparam(res)
        elif row in jac.GAS_ROW_MOLECULE:
            d[row] = jac.gas_dI_dparam(res, jac.GAS_ROW_MOLECULE[row], atm_params[row])
        else:
            raise NotImplementedError(f"no analytic derivative for row {row!r}")
    return sr.I_hires, d
