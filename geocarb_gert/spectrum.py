"""The one place that turns (atm_params, surface) into a `gert.ForwardModel`
run.

**Why this module exists** (2026-09-09). Before this, geocarb_simulator had
9 independent call sites that each built their own `gert.ForwardModel(...)`
and called `.run(...)`, hand-extracting atmosphere/surface/aerosol state on
their own -- see `docs/PROJECT_STATUS.md` Sec.12 for the full inventory.
Six of those sites were near-byte-for-byte duplicates of the same ~15-line
aerosol-kwarg-building block, independently pasted in as the aerosol rows
(`tau_aerosol`/`height_aerosol`) were added -- which is exactly how a real
bug (the truth-rendering path in `gd_test.py::_band_setup` never threading
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
from gert.rt_solver import SingleScatterSolver

from . import along_slit_scene as als
from .aerosol_defaults import aerosol_scalars_for


@dataclass
class SpectrumResult:
    """What every call site actually wants, plus the raw result for
    Jacobian dispatch / debugging."""
    I_hires: np.ndarray            # res.I_hires[0] -- the hi-res spectrum
    result: "gert.ForwardResult"   # the full raw ForwardModel result
    atm: object                    # the AtmosphericProfile actually used
    aerosol_kwargs: dict           # what was passed to fm.run for aerosol (empty if none)


def _build_aerosol_kwargs(surface: Optional[dict], n_wn: int, geo,
                          aerosol_type: str) -> dict:
    """`{}` when neither `tau_aerosol` nor `height_aerosol` is present in
    `surface` -- matches every existing call site's `.get(...)` convention:
    `tau_aerosol=None` is `gert.ForwardModel.run`'s own documented
    "aerosol term omitted", zero behavior change by default.
    """
    tau_aer = (surface or {}).get("tau_aerosol")
    height_aer = (surface or {}).get("height_aerosol")
    if tau_aer is None:
        return {}
    ssa, g, qext_norm = aerosol_scalars_for(aerosol_type)
    p_aer_val = als.aerosol_phase_hg(g, np.cos(geo.scattering_angle))
    return dict(
        tau_aerosol=tau_aer,
        height_aerosol=height_aer,
        aerosol_profile_shape="gaussian",
        thickness_aerosol=als.AEROSOL_THICKNESS_PA,
        ssa_aerosol=[np.full(n_wn, ssa)],
        g_aerosol=[g],
        qext_aerosol=[np.full(n_wn, qext_norm)],
        P_aerosol=[np.full(n_wn, p_aer_val)],
    )


def simulate_spectrum(atm_params: dict, surface: Optional[dict], absco, wide_inst,
                      geo, solar, jacobians: bool = False,
                      aerosol_type: str = "smoke") -> SpectrumResult:
    """Build the atmosphere from `atm_params` (the kwargs
    `along_slit_scene.atmosphere_from_params` takes) and run one
    `gert.ForwardModel` pass.

    `surface` follows the convention already used at every migrated call
    site: `{"albedo": ..., "albedo_slope": ..., "tau_aerosol": ...,
    "height_aerosol": ...}`, any key optional/absent -> that physics is
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
    aer_kwargs = _build_aerosol_kwargs(surface, n_wn, geo, aerosol_type)

    fm = ForwardModel(atm, absco, wide_inst, geo,
                      solver=SingleScatterSolver(jacobians=jacobians),
                      solar_spectrum=solar)
    res = fm.run(albedo=np.array([alb]), albedo_slope=np.array([slope]),
                jacobians=jacobians, **aer_kwargs)
    return SpectrumResult(I_hires=np.asarray(res.I_hires[0], dtype=float),
                          result=res, atm=atm, aerosol_kwargs=aer_kwargs)


def spectrum_and_jacobian(atm_params: dict, rows, absco, wide_inst, geo, solar,
                          surface: Optional[dict] = None,
                          aerosol_type: str = "smoke"):
    """``(S_hires, {row: dS/d(param)})`` -- the analytic-Jacobian
    counterpart of `simulate_spectrum`. Row dispatch is delegated to
    `geocarb_gert.jacobians` (`SURFACE_ROW_JACOBIAN`, `gas_dI_dparam`,
    `p_surface_dI_dparam`, `t_offset_dI_dparam`,
    `height_aerosol_dI_dparam`) -- unchanged from `make_spectrum_jac`,
    just relocated to call through `simulate_spectrum` for the nominal
    state instead of building its own `ForwardModel`.
    """
    from . import jacobians as jac  # local import: jacobians imports this module too

    sr = simulate_spectrum(atm_params, surface, absco, wide_inst, geo, solar,
                           jacobians=True, aerosol_type=aerosol_type)
    res, atm = sr.result, sr.atm
    surface = surface or {}
    alb = float(surface.get("albedo", 0.0))
    slope = float(surface.get("albedo_slope", 0.0))
    tau_aer = surface.get("tau_aerosol")
    height_aer = surface.get("height_aerosol")

    d = {}
    for row in rows:
        if row == "height_aerosol":
            d[row] = jac.height_aerosol_dI_dparam(res, atm, float(height_aer),
                                                   thickness_aerosol=als.AEROSOL_THICKNESS_PA)
        elif row in jac.SURFACE_ROW_JACOBIAN:
            d[row] = jac.surface_dI_dparam(res, row)
        elif row == "p_surface_hpa":
            d[row] = jac.p_surface_dI_dparam(res, atm_params, absco=absco, wide_inst=wide_inst,
                                             geo=geo, solar=solar, alb=alb, slope=slope,
                                             tau_aer=tau_aer, height_aer=height_aer)
        elif row == "t_offset_k":
            d[row] = jac.t_offset_dI_dparam(res)
        elif row in jac.GAS_ROW_MOLECULE:
            d[row] = jac.gas_dI_dparam(res, jac.GAS_ROW_MOLECULE[row], atm_params[row])
        else:
            raise NotImplementedError(f"no analytic derivative for row {row!r}")
    return sr.I_hires, d
