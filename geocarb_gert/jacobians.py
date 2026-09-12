"""Analytic Jacobians for the joint-block state vector.

Replaces `gauss_newton_state`'s finite differences with derivatives assembled
from `gert`'s own per-layer analytic arrays, composed with this package's
detector operator. Everything here reads what `ForwardModel.run(jacobians=
True)` already returns. **`gert` itself was additive-only modified once**
(2026-09-09): `ForwardResult` gained `airmass_hires`/`tau_abv_hires`/
`I_scatter_hires` (already-computed `SingleScatterSolver` internals that had
nowhere to go), specifically so `height_aerosol_dI_dparam` below could
become a true analytic composition instead of an RT-level finite
difference -- see that function's docstring.

Why bother, given it is barely faster
-------------------------------------
Measured on the real 670-700 window (width 31, 39 anchors, 62 free elements):
RT costs 17 ms per anchor (48 ms with `jacobians=True`), while one
application of the detector operator costs 325 ms. A GN iteration is
therefore ~63 x L either way, and analytic Jacobians only remove the RT
redundancy -- about 10%. The speedup worth having is sparsity in L (a
column's derivative is nonzero at only the 2-3 anchors its own
interpolation touches), which is independent of this module.

The case for analytic derivatives is CORRECTNESS, not speed:

* No step size, so no step-size tuning and no interaction between the probe
  and a parameter's physical bounds. The 2026-08-18 `p_surface` failure --
  26% of the slit lost because the `p * 1.001` probe crossed the standard-
  atmosphere ceiling -- is structurally impossible here.
* `gauss_newton_state` requires `kind="scale"` solely because one shared
  finite-difference step assumes every element is an O(1) multiplier.
  Analytic columns carry their own units, so `kind="absolute"` becomes
  available.

The composition
---------------
The detector operator is LINEAR in the hi-res radiance field: every pixel is
an ILS-weighted sum over its own anchor's spectrum
(`gd_render._diagonal_ils_convolve`), followed by `gaussian_blur_rows`.
`predict_neighborhood` takes the scene as a *closure*, so pushing a Jacobian
direction through is not new machinery -- it is the same call with a closure
returning `dS/dparam` instead of `S`:

    dY/dx_k = predict_neighborhood(..., footprint=True, radiance=footprint_average_scene(etas, dS_k))

with `dS_k[g] = (dS/dparam_r)[g] * W[g, k] * dval_dx_r[k]`, where `W` is
`StateSpec.interp_weights` (the state-space interpolation written as a
matrix) and `dval_dx_r[k]` is `d(value)/dx` -- `prior_r[k]` for
`kind="scale"` (the retrieved number multiplies the prior), or `1.0` for
`kind="absolute"` (the retrieved number IS the physical value -- 2026-09-07,
first exercised end-to-end by the `t_offset_k` row; see `linearize`'s own
`dval_dx` branch, kept in lockstep with `ParamSpec.apply`'s identical
`kind` branch by construction).

`instrument` rows are the exception: they never enter the radiance at all,
so they use `gd_render._diagonal_ils_convolve_dnu` instead. Not implemented
here yet -- see `dispersion` in the roadmap below.

Coverage
--------
=================  ===========  ==================================================
row                target       derivative
=================  ===========  ==================================================
co2_ppm            atmosphere   gert `K_mol_lay_hires` x `tau_gas_layer_hires`
ch4_ppb            atmosphere   same (exactly zero in a band without ch4)
co_ppb             atmosphere   same
h2o_surface_vmr    atmosphere   same, plus a known omitted q-coupling term below
p_surface_hpa      atmosphere   composite chain rule; 2.1e-5, see below
t_offset_k         atmosphere   gert `dtau_mol_dT_lay_hires` x `K_mol_lay_hires`,
                                no chain rule (uniform additive shift, no
                                altitude/VMR feedback) -- exact, like albedo
albedo             surface      gert `K_albedo_hires` -- exact, nothing to chain
albedo_slope       surface      gert `K_slope_hires`  -- exact (spectral slope)
tau_aerosol        surface      gert `K_tau_aer_hires` -- exact, nothing to chain
height_aerosol     surface      analytic composition through `tau_abv`, using
                                gert `I_scatter_hires`/`airmass_hires`
                                (2026-09-09) -- zero extra RT calls, but
                                genuinely non-smooth (hard pressure-layer
                                mask), see the function's own docstring;
                                `height_aerosol_dI_dparam_fd` kept as an
                                RT-level cross-check
dispersion         instrument   derivative built (`_diagonal_ils_convolve_dnu`),
                                but the forward model does not yet APPLY a
                                dispersion perturbation, so there is no row to
                                differentiate. That wiring is the next step.
=================  ===========  ==================================================

Validation status, measured by `scripts/gd_jacobian_validate.py --h-scan`.
Every row but `p_surface_hpa` tracks central differences down to roundoff,
with an h-curve whose SHAPE is itself a check: the gases show h^2 truncation
falling then roundoff rising (a V), while albedo shows no truncation at all
and pure 1/h roundoff, because with no aerosol the model is exactly linear
in albedo.

`p_surface_hpa` is the exception and stays the least exact row, though it
improved 20x when the scene moved to pure-sigma levels (`geocarb_gert.levels`)
on 2026-08-18. Before: a hard FLOOR at ~4e-4 that no step size could reduce,
from the Laprise `p_top` offset. After: a genuine V-curve bottoming at 2.1e-5
(h=1e-3), i.e. FD truncation and roundoff rather than a floor -- so the
coordinate mismatch is gone.

What remains is NOT the pressure path and NOT this module's internal
finite difference on `atmosphere_from_params`, which was measured stable to
7e-9 across h_rel from 1e-3 to 1e-7. By elimination it is the TEMPERATURE
path: gert's own `dtau_mol_dT_lay_hires` and the layer-midpoint chain rule
both feed it, and neither is exact the way the VMR and albedo derivatives
are. 2e-5 is far below anything that matters for a Gauss-Newton step, and
chasing it further would mean reaching into gert. Recorded so the asymmetry
between this row and the others is not mistaken for a bug.

Omitted h2o coupling term -- measured, not assumed
--------------------------------------------------
Our `atmosphere_from_params` derives specific humidity from the H2O VMR
(``w = _MW_RATIO*h2o; q = w/(1+w)``), and `q` sets the dry-air mass every
OTHER gas's layer amount is computed from. So changing H2O perturbs the CO2
optical depth by a path this chain rule does not include (`gert` treats
`h2o_scale` as a pure gas scaling too). Predicted a priori to be ~`q` in
relative size, i.e. sub-1% -- which would have mattered.

It is not. `scripts/gd_jacobian_validate.py --h-scan` puts the h2o column's
disagreement with central differences at 1.2e-9 (h=1e-4) falling to 2.1e-10
(h=1e-5) and rising again at 1e-6, i.e. pure FD truncation-then-roundoff
with no floor: the coupling contributes below 1e-10 relative, ~7 orders
under the estimate. Recorded here because the estimate was wrong in the safe
direction and someone will otherwise re-derive the same worry.
"""
from __future__ import annotations

import numpy as np

from . import gd_render
from .focalplane import footprint_average_scene
from .joint_state import ParamSpec, StateSpec

#: State row -> the `gert` molecule whose optical depth it scales. A row
#: whose molecule is absent from the band gets an exact zero column (that
#: band simply has no sensitivity to it), which the validator asserts rather
#: than assumes.
GAS_ROW_MOLECULE = {
    "co2_ppm": "co2",
    "ch4_ppb": "ch4",
    "co_ppb": "co",
    "h2o_surface_vmr": "h2o",
}


def gas_dI_dparam(res, mol: str, param_value: float, window: int = 0):
    """``dI_hires/d(param)`` for a row that scales one gas's VMR uniformly.

    Every gas row here maps to a profile that is LINEAR in the row's value:
    `atmosphere_from_params` sets ``gases[mol] = value * const`` for the
    well-mixed gases, and H2O's ``value * exp(-z/H)`` is likewise linear in
    the surface VMR. Optical depth is linear in VMR, so
    ``tau_mol(value) = value/value_0 * tau_mol(value_0)`` and

        dI/d(value) = sum_l K_mol_lay[l, :] * tau_layer[:, l] / value

    which is `gert`'s own ``dR/d(scale) = ILS(K_mol . tau_gas)/scale``
    (retrieval.py) written per layer and before the ILS, since this package
    applies its own ILS downstream.

    Returns zeros when the molecule is not one of the band's absorbers.
    """
    K_lay = res.K_mol_lay_hires[window]
    tau_lay = res.tau_gas_layer_hires[window]
    n_hires = np.shape(res.I_hires[window])[0]
    if mol not in K_lay or mol not in tau_lay:
        return np.zeros(n_hires)
    if param_value == 0.0:
        raise ValueError(f"{mol}: cannot form d/d(value) at value == 0 "
                         f"(tau is linear in it, so the derivative is tau/value)")
    # K_mol_lay is (n_lay, n_wn); tau_gas_layer is (n_wn, n_lay).
    return np.einsum("lw,wl->w", K_lay[mol], tau_lay[mol]) / float(param_value)


#: `surface`-target rows and the `ForwardResult` field holding their exact
#: hi-res derivative. These come free with ``jacobians=True`` -- no chain
#: rule, no approximation: gert differentiates the surface term directly.
SURFACE_ROW_JACOBIAN = {
    "albedo": "K_albedo_hires",
    "albedo_slope": "K_slope_hires",
    "tau_aerosol": "K_tau_aer_hires",
}


def surface_dI_dparam(res, row: str, window: int = 0):
    """``dI_hires/d(param)`` for a `surface` row, straight from gert.

    Unlike the gas rows there is nothing to chain: ``K_albedo_hires`` IS
    ``dI/d(albedo)`` at the hi-res grid, before any ILS. (gert's own
    ``K_albedo = I_direct/albedo`` for the single-scatter solver, but this
    reads the returned array rather than re-deriving it, so it stays correct
    if the solver changes.)
    """
    field = SURFACE_ROW_JACOBIAN.get(row)
    if field is None:
        raise NotImplementedError(f"no surface derivative for row {row!r}")
    arr = getattr(res, field, None)
    if arr is None:
        raise RuntimeError(f"{field} is None -- was the ForwardModel run with "
                           f"jacobians=True and a jacobian-enabled solver?")
    return np.asarray(arr[window], dtype=float)


def _layer_mid(a):
    """Level array (n_lev,) -> layer array (n_lev-1,) by midpoint."""
    a = np.asarray(a, dtype=float)
    return 0.5 * (a[:-1] + a[1:])


def p_surface_dI_dparam(res, params, window: int = 0, h_rel: float = 1e-6,
                        absco=None, wide_inst=None, geo=None, solar=None,
                        alb=None, slope=None, tau_aer=None, height_aer=None,
                        h_rel_rt: float = 1e-3):
    """``dI_hires/d(p_surface_hpa)`` -- dispatches to the fast analytic
    composition (`_p_surface_dI_dparam_analytic`) when no aerosol row is
    present (`tau_aer is None` -- every existing caller, zero behavior
    change), or a genuine RT-level finite difference when aerosol IS
    present.

    **Why aerosol needs the RT fallback** (2026-09-08, found while
    investigating why a joint co2/p_surface/tau_aerosol retrieval failed
    to converge): the analytic composition below is exact for the gas-
    amount-change paths (pressure/temperature/VMR), including their OWN
    interaction with the aerosol scattering term -- `K_mol_lay_hires`
    already has `K_tau_lay`'s own `above_frac[l,nu]` weighting baked in
    (`gert.rt_solver.SingleScatterSolver`'s own per-layer Jacobian
    formula), and `above_frac` is itself a SMOOTH function of cumulative
    gas+Rayleigh optical depth, not a hard threshold. But changing
    `p_surface_hpa` also RESCALES the entire sigma-level pressure grid
    (`atm.p_layers`), which shifts which physical layers count as
    "above" the aerosol's own FIXED `height_aerosol` threshold
    (`above_mask = atm.p_layers < height_aerosol` inside `ForwardModel.
    run` itself, upstream of anything `K_mol_lay`/`K_tau_lay` can see) --
    a geometric effect the analytic composition has no way to know
    about, measured directly as a real (not FD-noise) ~6-10% relative
    Jacobian error once aerosol coexists (cosine ~0.9997 -- direction
    right, magnitude wrong). `gert` doesn't expose `tau_abv`/`I_scatter`/
    the airmass factor as separate `ForwardResult` fields, so -- mirroring
    `height_aerosol_dI_dparam`'s own accepted cost tradeoff -- this
    falls back to 2 extra full RT calls rather than trying to patch in
    just the missing piece analytically.
    """
    if tau_aer is None:
        return _p_surface_dI_dparam_analytic(res, params, window=window, h_rel=h_rel)

    from gert.forward_model import ForwardModel
    from gert.rt_solver import SingleScatterSolver
    from . import along_slit_scene as als

    p_sfc_hpa = float(params["p_surface_hpa"])
    n_wn = len(wide_inst.windows[0].wn_hires)
    h = h_rel_rt * p_sfc_hpa
    p_aer_val = als.aerosol_phase_hg(als.AEROSOL_G, np.cos(geo.scattering_angle))

    def _I(p_sfc):
        atm = als.atmosphere_from_params(**{**params, "p_surface_hpa": p_sfc})
        fm = ForwardModel(atm, absco, wide_inst, geo, solver=SingleScatterSolver(),
                          solar_spectrum=solar)
        res_ = fm.run(albedo=np.array([alb]), albedo_slope=np.array([slope]),
                     tau_aerosol=tau_aer, height_aerosol=height_aer,
                     aerosol_profile_shape="gaussian",
                     thickness_aerosol=als.AEROSOL_THICKNESS_PA,
                     ssa_aerosol=[np.full(n_wn, als.AEROSOL_SSA)],
                     g_aerosol=[als.AEROSOL_G],
                     qext_aerosol=[np.full(n_wn, als.AEROSOL_QEXT_NORM)],
                     P_aerosol=[np.full(n_wn, p_aer_val)])
        return np.asarray(res_.I_hires[0], dtype=float)

    return (_I(p_sfc_hpa + h) - _I(p_sfc_hpa - h)) / (2.0 * h)


def _p_surface_dI_dparam_analytic(res, params, window: int = 0, h_rel: float = 1e-6):
    """``dI_hires/d(p_surface_hpa)`` -- the composite chain rule.

    Surface pressure is the only row that is not a simple scaling of one
    quantity. `atmosphere_from_params` rebuilds the ENTIRE profile from it::

        p_l  = sigma_l * 100*p_sfc                    (geocarb_gert.levels, pure sigma)
        z_l  = pressure_to_alt_std_atm(p_l)
        T_l  = _std_temperature(z_l)
        h2o_l = h2o_surface_vmr * exp(-z_l / H)

    so three separate paths reach the optical depth: pressure directly
    (layer amount and pressure broadening), temperature, and the H2O volume
    mixing ratio. Measured earlier, ignoring the T and H2O paths -- i.e.
    using gert's own ``p_scale`` Jacobian as-is -- gives a derivative 38%
    too small and ~32 degrees off (cosine 0.848), so none of them is
    optional.

    Everything here is exact
    ------------------------
    * ``dp_l/dp_sfc = sigma_l = p_l/p_sfc`` -- exactly parallel to the
      uniform-scaling direction gert linearised around, because this scene
      uses PURE SIGMA levels (`geocarb_gert.levels`) rather than gert's
      Laprise eta. Both the layer-thickness and the pressure-broadening term
      inside ``dtau_mol_dpscale_lay`` therefore pick up the same
      ``1/p_sfc``, so that combined array converts with a single constant and
      never has to be split.

      This was NOT true before 2026-08-18. Under Laprise eta the true
      transform weights those two terms differently, the split cannot be
      recovered from the sum, and this Jacobian floored at ~4e-4 relative
      (cosine 0.9999999) instead of reaching roundoff. Changing the scene's
      vertical coordinate removed the approximation at its source rather
      than bounding it -- see `geocarb_gert.levels` for the argument and for
      what the level change costs (nothing at standard surface pressure,
      <=26 Pa elsewhere, that maximum at the model top where nothing
      absorbs).
    * The Rayleigh term is exact for the same reason: a layer's air column
      is proportional to its pressure thickness, hence to ``p_sfc``, so
      ``d(tau_ray_l)/d(p_sfc) = tau_ray_l / p_sfc``.
    * ``dT_l/dp_sfc`` and ``dh2o_l/dp_sfc`` are central-differenced on
      `atmosphere_from_params` ALONE. That map is pure algebra -- no RT, no
      absorption coefficients, microseconds to evaluate -- so the finite
      difference is cheap and clean, and it handles `_std_temperature`'s
      piecewise-linear kinks and the barometric layer boundaries in
      `pressure_to_alt_std_atm` without this module reimplementing either.
      The radiative transfer is never finite-differenced.
    """
    from . import along_slit_scene as als

    K_mol_lay = res.K_mol_lay_hires[window]
    dtau_dp = res.dtau_mol_dpscale_lay_hires[window]
    dtau_dT = res.dtau_mol_dT_lay_hires[window]
    tau_lay = res.tau_gas_layer_hires[window]
    n_hires = np.shape(res.I_hires[window])[0]

    p_sfc_hpa = float(params["p_surface_hpa"])
    # pure sigma: p_l = sigma_l * P, so d/d(p_sfc) = (d/d[uniform scale]) / P
    # exactly, with no p_top offset to drop. See geocarb_gert.levels.
    P = p_sfc_hpa * 100.0                       # Pa
    denom = P

    # -- cheap central differences on the algebra-only profile construction --
    h = h_rel * p_sfc_hpa
    a_p = als.atmosphere_from_params(**{**params, "p_surface_hpa": p_sfc_hpa + h})
    a_m = als.atmosphere_from_params(**{**params, "p_surface_hpa": p_sfc_hpa - h})
    a_0 = als.atmosphere_from_params(**params)
    dT_lay = _layer_mid((a_p.T_levels - a_m.T_levels) / (2.0 * h))
    dvmr_lay = {m: _layer_mid((a_p.gases[m] - a_m.gases[m]) / (2.0 * h))
                for m in a_0.gases}
    vmr_lay = {m: _layer_mid(a_0.gases[m]) for m in a_0.gases}

    out = np.zeros(n_hires)
    for mol, K in K_mol_lay.items():                      # K: (n_lay, n_wn)
        # (a) pressure path -- exact under pure sigma
        if mol in dtau_dp:
            out += np.einsum("lw,wl->w", K, dtau_dp[mol]) / denom * 100.0
        # (b) temperature path -- exact per-layer derivative from gert
        if mol in dtau_dT:
            out += np.einsum("lw,wl->w", K, dtau_dT[mol] * dT_lay[None, :])
        # (c) VMR path -- only H2O's profile moves with p_surface (via z);
        #     tau is linear in vmr, so dtau/dvmr = tau/vmr
        if mol in tau_lay and mol in dvmr_lay:
            dv = dvmr_lay[mol]
            if np.any(dv != 0.0):
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(vmr_lay[mol] > 0, dv / vmr_lay[mol], 0.0)
                out += np.einsum("lw,wl->w", K, tau_lay[mol] * ratio[None, :])

    # (d) Rayleigh: tau_ray is proportional to the layer's own air column, so
    #     d(tau_ray_l)/d(p_sfc) = tau_ray_l / P exactly.
    K_ray = getattr(res, "K_ray_lay_hires", None)
    tau_ray = getattr(res, "tau_ray_lay_hires", None)
    if K_ray is not None and tau_ray is not None:
        out += np.einsum("lw,l->w", K_ray[window],
                         np.asarray(tau_ray[window], dtype=float)) / denom * 100.0
    return out


def t_offset_dI_dparam(res, window: int = 0):
    """``dI_hires/d(t_offset_k)`` -- exact, no finite differences.

    Unlike `p_surface_dI_dparam`'s temperature-path term, a uniform
    additive shift moves every layer's temperature by exactly 1K per 1K of
    offset with no altitude/VMR feedback (see `along_slit_scene.
    atmosphere_from_params`'s own `t_offset_k` docstring) -- so this is
    nothing but gert's own ``dtau_mol_dT_lay_hires`` summed against
    ``K_mol_lay_hires``, the SAME per-layer arrays `p_surface_dI_dparam`
    already reads for its own temperature path, just without the
    `dT_lay` chain-rule factor (here it's implicitly 1 everywhere, so it
    drops out of the einsum entirely). No Rayleigh term either -- Rayleigh
    optical depth depends on the layer's air column (pressure), not
    temperature, under this scene's pure-sigma levels.
    """
    K_mol_lay = res.K_mol_lay_hires[window]
    dtau_dT = res.dtau_mol_dT_lay_hires[window]
    n_hires = np.shape(res.I_hires[window])[0]
    out = np.zeros(n_hires)
    for mol, K in K_mol_lay.items():
        if mol in dtau_dT:
            out += np.einsum("lw,wl->w", K, dtau_dT[mol])
    return out


def height_aerosol_dI_dparam(res, atm, height_aerosol_val, window: int = 0,
                             h_rel: float = 1e-2, thickness_aerosol=None):
    """``dI_hires/d(height_aerosol)`` -- analytic composition through
    `tau_abv`, using `ForwardResult.I_scatter_hires`/`airmass_hires`
    (2026-09-09, `gert` now exposes both). Costs ZERO extra RT calls --
    a real fix, not just a cheaper approximation, superseding this
    function's earlier RT-level-finite-difference form (kept below as
    `height_aerosol_dI_dparam_fd` for cross-checking).

    **The chain rule.** `SingleScatterSolver`'s single-scatter term is
    `I_scatter(nu) = A(nu) * exp(-m * tau_abv(nu))`, so
    `d(I_scatter)/d(tau_abv) = -m * I_scatter` exactly -- no
    approximation, straight from the RT formula (`rt_solver.py`'s own
    docstring), using `res.airmass_hires[window]`/`res.I_scatter_hires
    [window]` directly instead of reconstructing them from `I_hires`.

    `tau_abv = tau_mol_above + tau_ray_above` is a sum, over layers with
    `atm.p_layers < height_aerosol`, of PER-LAYER optical depths that do
    NOT themselves depend on `height_aerosol` -- only the mask does. So
    `tau_abv`'s own derivative can be finite-differenced with NO RT at
    all: just re-sum the already-computed `res.tau_gas_layer_hires`/
    `res.tau_ray_lay_hires` (populated by the SAME `jacobians=True` call
    already made for the nominal state) under the mask evaluated at
    `height +/- h`. This mirrors `p_surface_dI_dparam`'s own
    `dT_lay`/`dvmr_lay` pattern: cheap, algebra-only, no RT.

    **The non-smoothness caveat is UNCHANGED.** `above_mask = atm.
    p_layers < height_aerosol` (`forward_model.py`) is still a hard
    boolean threshold -- this function computes the true local
    derivative of that (still discontinuous) function, which removes
    the "which FD step size am I even sampling" ambiguity the old
    RT-level FD had, but does not smooth the underlying mask (a
    separate, larger, deliberately out-of-scope change -- see
    `docs/PROJECT_STATUS.md` Sec.11's accepted caveat).

    Why `K_aer_lay_hires`-based chain rule alone still doesn't work: see
    `height_aerosol_dI_dparam_fd`'s docstring below (unchanged finding).
    """
    tau_gas_layer = res.tau_gas_layer_hires[window] if res.tau_gas_layer_hires else None
    tau_ray_lay = res.tau_ray_lay_hires[window] if res.tau_ray_lay_hires else None
    I_scatter = res.I_scatter_hires[window] if res.I_scatter_hires is not None else None
    m = res.airmass_hires[window] if res.airmass_hires is not None else None
    n_hires = np.shape(res.I_hires[window])[0]
    if not tau_gas_layer or tau_ray_lay is None or I_scatter is None or m is None:
        return np.zeros(n_hires)

    p_layers = np.asarray(atm.p_layers, dtype=float)
    tau_ray_lay = np.asarray(tau_ray_lay, dtype=float)
    sigma = max(float(thickness_aerosol), 100.0) if thickness_aerosol is not None else 1.0e4
    h = h_rel * sigma

    def tau_abv_at(height):
        above_mask = p_layers < height
        out = np.zeros(n_hires)
        for tau_lay_wn in tau_gas_layer.values():          # (n_wn, n_lay)
            out += tau_lay_wn[:, above_mask].sum(axis=1)
        out += float(tau_ray_lay[above_mask].sum())
        return out

    dtau_abv_dheight = (tau_abv_at(height_aerosol_val + h)
                        - tau_abv_at(height_aerosol_val - h)) / (2.0 * h)
    return -m * I_scatter * dtau_abv_dheight


def height_aerosol_dI_dparam_fd(absco, wide_inst, geo, solar, atm, alb, slope,
                                tau_aer, height_aerosol_val, h_rel: float = 1e-2):
    """``dI_hires/d(height_aerosol)`` -- a genuine RT-level finite
    difference. Superseded by `height_aerosol_dI_dparam` above (2026-09-09,
    `gert` now exposes `I_scatter_hires`/`airmass_hires`); kept here only
    as a cross-check reference for the analytic version.

    **Why this row didn't get a cheap analytic composition originally.**
    `SingleScatterSolver`'s own single-scatter term is
    `I_scatter = A(nu) * exp(-m * tau_abv(nu))`, where `tau_abv` is the
    gas+Rayleigh optical depth ABOVE the aerosol layer (`rt_solver.py`'s
    own docstring) -- THIS is the term that genuinely depends on
    `height_aerosol` (a higher aerosol layer has more gas above it to
    attenuate the scattered light on the way up). `K_aer_lay_hires`
    (`dI/d(tau_aer_lay[l])`) does NOT capture this -- checked directly
    (2026-09-08): in a real single-window solve, `K_aer_lay[l, w]` is
    IDENTICAL across every layer l (the SS model doesn't distinguish
    which layer the scattering happens at). Combined with
    `_aer_class_layer_arrays`'s own `aer_frac` being normalized (sums to
    1 always, so `sum_l d(aer_frac[l])/d(height) = 0` exactly), the
    `K_aer_lay`-chain-rule composition is mathematically forced to ~0
    (measured: ~1e-19, floating-point noise) regardless of the real
    physical sensitivity through `tau_abv` -- a real, not a bug. Before
    `gert` exposed `I_scatter_hires`/`airmass_hires`/`tau_abv_hires`,
    there was no cheap, algebra-only quantity to finite-difference this
    way, so this fell back to 2 extra full RT calls.
    """
    from gert.forward_model import ForwardModel
    from gert.rt_solver import SingleScatterSolver
    from . import along_slit_scene as als

    n_wn = len(wide_inst.windows[0].wn_hires)
    h = h_rel * als.AEROSOL_THICKNESS_PA
    p_aer_val = als.aerosol_phase_hg(als.AEROSOL_G, np.cos(geo.scattering_angle))

    def _I(height):
        fm = ForwardModel(atm, absco, wide_inst, geo, solver=SingleScatterSolver(),
                          solar_spectrum=solar)
        res = fm.run(albedo=np.array([alb]), albedo_slope=np.array([slope]),
                     tau_aerosol=tau_aer, height_aerosol=height,
                     aerosol_profile_shape="gaussian",
                     thickness_aerosol=als.AEROSOL_THICKNESS_PA,
                     ssa_aerosol=[np.full(n_wn, als.AEROSOL_SSA)],
                     g_aerosol=[als.AEROSOL_G],
                     qext_aerosol=[np.full(n_wn, als.AEROSOL_QEXT_NORM)],
                     P_aerosol=[np.full(n_wn, p_aer_val)])
        return np.asarray(res.I_hires[0], dtype=float)

    return (_I(height_aerosol_val + h) - _I(height_aerosol_val - h)) / (2.0 * h)


def make_spectrum_jac(absco, wide_inst, geo, solar, albedo):
    """``spectrum_jac(params, rows) -> (S_hires, {row: dS/d(param)})``.

    Thin wrapper around `geocarb_gert.spectrum.spectrum_and_jacobian`
    (2026-09-09 consolidation -- see that module's docstring), kept under
    this name/signature for any caller still using it directly. The only
    thing this wrapper does that `spectrum_and_jacobian` doesn't is
    resolve the `albedo` fallback -- `(surface or {}).get("albedo",
    albedo)` -- since `spectrum_and_jacobian` deliberately has no
    module-scalar fallback of its own (see its docstring: albedo
    resolution is the caller's responsibility).
    """
    from .spectrum import spectrum_and_jacobian

    def spectrum_jac(params: dict, rows, surface: dict | None = None):
        surface = dict(surface or {})
        surface["albedo"] = float(surface.get("albedo", albedo))
        return spectrum_and_jacobian(params, rows, absco, wide_inst, geo, solar,
                                     surface=surface)

    return spectrum_jac


_ANCHOR_SPECTRA_G: dict = {}


def _anchor_spectra_one(g):
    """Module-level (picklable, fork-inherited -- same pattern as
    `joint_state._render_one_anchor`) so `anchor_spectra_and_derivs` can
    run its per-anchor `spectrum_jac` calls in parallel. 2026-09-02
    (user): "let's parallelize the linearize step as well" -- this is the
    RT+derivative half of that; `linearize`'s own `L()` calls (the
    detector-operator half, the other real cost driver at ~325ms/call
    per this module's own docstring) are parallelized separately, below."""
    G = _ANCHOR_SPECTRA_G
    surf = G["surface_at_anchor"][g] if G["surface_at_anchor"] else None
    s, d = G["spectrum_jac"](G["params_at_anchor"][g], G["rows_needed"], surf)
    return g, np.asarray(s, dtype=float), d


def anchor_spectra_and_derivs(spectrum_jac, params_at_anchor, rows_needed,
                              surface_at_anchor=None, n_workers: int = 1,
                              pool: "LinearizePool | None" = None):
    """Run the forward model once per anchor, returning radiance and the
    hi-res derivative of each requested row.

    `spectrum_jac(params, rows, surface) -> (S, {row_name: dS/d(param)})` is
    supplied by the caller (see :func:`make_spectrum_jac`) so this module
    stays independent of how the band was configured.

    `n_workers` (2026-09-02, user): anchors are independent RT calls --
    ``1`` (default, unchanged) runs them sequentially; a daemon Pool
    worker forces this back to 1 regardless (cannot nest pools), same
    guard `joint_state.build_forward_state`/`render_at_anchors` use.

    `pool` (2026-09-12, user): a `LinearizePool` spanning the whole GN
    solve this call is one iteration of -- when given, its long-lived
    anchor-pool workers are used instead of forking a fresh one here; see
    `LinearizePool`'s docstring for why. Falls through to the plain path
    below (unchanged) if `pool` is None or declines (e.g. inside a daemon
    worker, where nested pools aren't allowed).
    """
    if pool is not None:
        result = pool.run_anchor(params_at_anchor, rows_needed, surface_at_anchor)
        if result is not None:
            return result
    import multiprocessing as mp
    n_anchor = len(params_at_anchor)
    if mp.current_process().daemon:
        n_workers = 1
    if n_workers <= 1 or n_anchor < 8:
        S, dS = [], []
        for g, p in enumerate(params_at_anchor):
            surf = surface_at_anchor[g] if surface_at_anchor else None
            s, d = spectrum_jac(p, rows_needed, surf)
            S.append(np.asarray(s, dtype=float))
            dS.append(d)
        return S, dS

    _ANCHOR_SPECTRA_G.update(dict(spectrum_jac=spectrum_jac, params_at_anchor=params_at_anchor,
                                  rows_needed=rows_needed, surface_at_anchor=surface_at_anchor))
    S: list = [None] * n_anchor
    dS: list = [None] * n_anchor
    ctx = mp.get_context("fork")
    with ctx.Pool(min(n_workers, n_anchor)) as pool:
        for g, s, d in pool.imap_unordered(_anchor_spectra_one, range(n_anchor), chunksize=2):
            S[g], dS[g] = s, d
    return S, dS


def _g_anomaly_sensitivity(p: ParamSpec, scene_etas, W, state_interp):
    """``d(value)/d(g_value_m)`` for a row's `sub_bin_anomaly`, at every
    `scene_etas` position -- a plain LINEAR formula (no quotient rule,
    unlike the retired multiplicative scheme's `K_g`), since the additive
    correction ``g(eta) - plain_interp_of_g(eta)`` never divides by
    anything:

        d(value)/d(g_value_m) = Wg[eta,m] - sum_k(W[eta,k]*Wg[bin_center_k,m])

    `W` is the SAME `interp_weights` matrix `linearize`'s own loop already
    computed for this row's state Jacobian `K` -- passed in rather than
    recomputed. `Wg` is `g`'s own `interp_weights`-style matrix (a
    throwaway single-row `StateSpec` on `sub_bin_anomaly.g_positions`, the
    same reusable-throwaway-spec trick used elsewhere in this codebase);
    `Wg[bin_center_k,:]` (`sub_bin_anomaly.Wg_at_centers`) was precomputed
    once at `state_spec_from_scene` construction time, since it depends
    only on positions, never on the retrieved state.
    """
    sba = p.sub_bin_anomaly
    kind = p.state_interp if p.state_interp is not None else state_interp
    g_throwaway = ParamSpec(name="_gpos", positions=sba.g_positions,
                            prior=sba.g_positions, sigma=1.0, free=True, kind="scale")
    Wg_scene = StateSpec([g_throwaway]).interp_weights(scene_etas, "_gpos", kind)
    return Wg_scene - W @ sba.Wg_at_centers


_LINEARIZE_L_G: dict = {}


def _L_worker(idx):
    """Module-level (picklable, fork-inherited) counterpart of
    `linearize`'s own local `L(field)` closure -- the detector-operator
    half of `linearize`'s cost, parallelized across every K/K_g column at
    once (2026-09-02, user).

    2026-09-11 (user: "are there duplicates in memory that could be
    avoided" -- the impprior_ws OOMs): this used to take the whole
    `(dest, field)` job as its argument, which `pool.map` pickles into
    a pipe to reach the worker -- a real copy of every (potentially
    large, one per free state SCALAR) `field` array, on top of the copy
    already sitting in the parent's `jobs` list, and another on
    deserialization in the worker. `anchor_spectra_and_derivs` (this
    same module, just above) already avoids exactly this for its own
    per-anchor inputs by stashing them in a fork-inherited global before
    the pool starts, so workers see them via copy-on-write instead of
    IPC -- `jobs` itself now goes into `_LINEARIZE_L_G` the same way,
    and only a plain integer index (cheap to pickle) travels through
    `pool.map`. Only the (much smaller, `y.size`-length) result `col`
    still crosses the IPC boundary, same as before."""
    from . import gd_render
    from .focalplane import footprint_average_scene
    G = _LINEARIZE_L_G
    dest, field = G["jobs"][idx]
    col = gd_render.predict_neighborhood(
        G["fpa"], G["rows_win"], G["wn_hires"], footprint_average_scene(G["scene_etas"], field),
        G["ils"], pad=G["pad"], footprint=True).ravel()
    return dest, col


class LinearizePool:
    """A persistent worker pool spanning every GN iteration of ONE
    `gauss_newton_state` call (one stage -- coarse or hires -- of one
    window), replacing the plain (non-pool) path's fresh `ctx.Pool(...)`
    per `linearize()` call in both `anchor_spectra_and_derivs` and
    `linearize`'s own L()-loop.

    2026-09-12 (user: "let's make that change", following up on the
    impprior_ws OOMs Sec.17's `pool.map`-pickling fix didn't fully cure):
    forking a NEW Pool on every GN iteration (up to 15/window --
    `gauss_newton_state`'s own `max_iter`) forks from whatever the PARENT
    process's RSS has grown to by that iteration -- and Python's own
    refcounting touches (so copy-on-write duplicates) nearly every object
    header a worker looks at, not just the bytes it actually needs. A
    later iteration's fork therefore costs strictly more than an earlier
    one even when the real job data is an identical size -- exactly the
    "climbs with each iteration, no plateau" behavior
    `submit_impprior_ws.sbatch` already documents. The fix: fork ONCE, as
    early in the solve as possible (before iteration has had a chance to
    grow the parent), and never again.

    That breaks the fork-inherited-global trick used elsewhere in this
    module (stash data in a plain dict, then fork so children see it via
    copy-on-write) for anything that changes iteration to iteration --
    only a FRESH fork ever picks up a global's latest value, and this pool
    is deliberately never fresh after iteration 1. Two different fixes for
    two different job shapes, both below:

    * `run_anchor`'s per-iteration inputs (`params_at_anchor`/
      `surface_at_anchor`) are tiny (a handful of floats per anchor) --
      cheap to just pass through `pool.map`'s own pickling every call,
      same as any ordinary multiprocessing job. `spectrum_jac` (an
      unpicklable closure) is the only thing that does NOT change
      per-iteration, so it alone goes into the fork-inherited global, set
      once before this class's one anchor-pool fork.
    * `run_L`'s per-iteration inputs (the K/K_g `field` arrays) are
      exactly the large arrays Sec.17 found real pickling duplication in
      -- still true with a long-lived pool. They go into a genuine
      (name-addressed, OS-level) `multiprocessing.shared_memory` segment
      instead: workers attach to it ONCE, by name, the first time they're
      asked to do L-loop work, then keep re-reading its CURRENT bytes on
      every later call -- the parent overwrites that same segment before
      each `pool.map` without forking again. Only the segment's
      name/shape/dtype (set into the fork-inherited global before this
      pool's own lazy, one-time L-pool fork, once the first call reveals
      the window's fixed job shape) and integer indices still cross
      through `pool.map`'s pickling.
    """

    def __init__(self, n_workers: int, spectrum_jac):
        import multiprocessing as mp
        self.n_workers = 1 if mp.current_process().daemon else n_workers
        self._anchor_pool = None
        self._L_pool = None
        self._shm = None
        self._shm_shape = None
        self._shm_dtype = None
        if self.n_workers > 1:
            _ANCHOR_SPECTRA_G["spectrum_jac"] = spectrum_jac
            ctx = mp.get_context("fork")
            self._anchor_pool = ctx.Pool(self.n_workers)

    def run_anchor(self, params_at_anchor, rows_needed, surface_at_anchor=None):
        """`None` return tells the caller to fall back to its own serial
        path -- mirrors `anchor_spectra_and_derivs`'s own `n_workers<=1`
        guard."""
        if self._anchor_pool is None:
            return None
        n = len(params_at_anchor)
        args = [(g, params_at_anchor[g],
                surface_at_anchor[g] if surface_at_anchor else None, rows_needed)
               for g in range(n)]
        S: list = [None] * n
        dS: list = [None] * n
        for g, s, d in self._anchor_pool.imap_unordered(_anchor_spectra_one_persistent, args, chunksize=2):
            S[g], dS[g] = s, d
        return S, dS

    def run_L(self, jobs, *, fpa, rows_win, wn_hires, scene_etas, ils, pad):
        """`None` return tells the caller to fall back (same contract as
        `run_anchor`).

        2026-09-12 (user's own stress test, c5 task 44 -- OOM'd in under 2
        minutes, far WORSE than the fresh-Pool-per-call code this was
        replacing): the first version of this method built `fields =
        np.stack(...)` -- a full second copy of every job's field array --
        THEN copied that into the shared buffer -- a third. Three
        same-sized copies of what could already be tens of GB for one L()
        batch on a wide/aerosol window swamped a 128G budget almost
        instantly, instead of the intended one-shared-buffer footprint.
        Fixed by writing each field into the shared buffer AS `jobs` IS
        WALKED, immediately dropping this method's own reference to it
        (`jobs[i] = None`) so nothing outside `jobs` itself (already built
        by `linearize`'s own loop, unavoidable, and identical in size to
        what the old fresh-Pool-per-call path also held) keeps it alive --
        peak extra memory here is one shared buffer plus, transiently, one
        field, not two extra full copies of the whole batch.
        """
        if self.n_workers <= 1 or len(jobs) < 4:
            return None
        n = len(jobs)
        first_field = np.asarray(jobs[0][1], dtype=np.float64)
        shape = (n,) + first_field.shape
        if self._L_pool is None:
            # First L() batch this window has ever needed -- only now do we
            # know its fixed job shape, so allocate the shared buffer and
            # fork this stage's L-pool, IN THAT ORDER (the manifest below
            # must already be in the global before the fork that lets
            # workers see it via copy-on-write).
            import multiprocessing as mp
            from multiprocessing import shared_memory
            nbytes = int(np.prod(shape)) * first_field.itemsize
            self._shm = shared_memory.SharedMemory(create=True, size=nbytes)
            self._shm_shape = shape
            self._shm_dtype = first_field.dtype
            _LINEARIZE_L_G.update(dict(fpa=fpa, rows_win=rows_win, wn_hires=wn_hires,
                                       scene_etas=scene_etas, ils=ils, pad=pad,
                                       shm_name=self._shm.name, shm_shape=self._shm_shape,
                                       shm_dtype=self._shm_dtype))
            ctx = mp.get_context("fork")
            self._L_pool = ctx.Pool(self.n_workers)
        elif shape != self._shm_shape:
            raise RuntimeError(
                f"LinearizePool.run_L: job shape changed mid-window "
                f"({self._shm_shape} -> {shape}) -- one instance is "
                f"scoped to exactly one gauss_newton_state call, whose "
                f"free-parameter structure never changes between GN "
                f"iterations")
        buf = np.ndarray(self._shm_shape, dtype=self._shm_dtype, buffer=self._shm.buf)
        dest_list: list = [None] * n
        for i in range(n):
            dest, field = jobs[i]
            buf[i] = field
            dest_list[i] = dest
            jobs[i] = None   # drop this field the instant it's copied in,
                              # rather than holding the whole batch AND the
                              # shared buffer resident at once
        cols = self._L_pool.map(_L_worker_shm, range(n), chunksize=1)
        return list(zip(dest_list, cols))

    def close(self):
        if self._anchor_pool is not None:
            self._anchor_pool.terminate()
            self._anchor_pool.join()
            self._anchor_pool = None
        if self._L_pool is not None:
            self._L_pool.terminate()
            self._L_pool.join()
            self._L_pool = None
        if self._shm is not None:
            self._shm.close()
            self._shm.unlink()
            self._shm = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _anchor_spectra_one_persistent(arg):
    """`LinearizePool.run_anchor`'s worker -- everything needed travels in
    `arg` except the unpicklable `spectrum_jac` closure, which came in via
    `_ANCHOR_SPECTRA_G` before this (long-lived) worker was forked."""
    g, p, surf, rows_needed = arg
    spectrum_jac = _ANCHOR_SPECTRA_G["spectrum_jac"]
    s, d = spectrum_jac(p, rows_needed, surf)
    return g, np.asarray(s, dtype=float), d


_L_WORKER_SHM_CACHE: dict = {}


def _L_worker_shm(idx):
    """`LinearizePool.run_L`'s worker. Attaches to the shared field buffer
    by name ONCE per worker PROCESS (cached across every task this
    long-lived worker ever handles, not just this call) -- see
    `LinearizePool`'s own docstring for why this replaces `_L_worker`'s
    fork-inherited-global `jobs` list."""
    from . import gd_render
    from .focalplane import footprint_average_scene
    from multiprocessing import shared_memory
    G = _LINEARIZE_L_G
    name = G["shm_name"]
    shm = _L_WORKER_SHM_CACHE.get(name)
    if shm is None:
        shm = shared_memory.SharedMemory(name=name)
        _L_WORKER_SHM_CACHE[name] = shm
    fields = np.ndarray(G["shm_shape"], dtype=G["shm_dtype"], buffer=shm.buf)
    return gd_render.predict_neighborhood(
        G["fpa"], G["rows_win"], G["wn_hires"],
        footprint_average_scene(G["scene_etas"], fields[idx]),
        G["ils"], pad=G["pad"], footprint=True).ravel()


def linearize(fpa, rows_win, scene_etas, spec: StateSpec, spectrum_jac,
              wn_hires, ils, x, pad: int = 4, state_interp: str = "linear",
              n_workers: int = 1, pool: "LinearizePool | None" = None):
    """``(y, K, K_g)`` -- the predicted sub-image, its analytic Jacobian,
    and (only for rows with `sub_bin_anomaly.g_cov` set) the sensitivity
    of each anchor value to that row's `g` values.

    `K` has one column per free element, ordered exactly as
    :meth:`StateSpec.slices` packs them, so it drops straight into the
    Rodgers step in `gauss_newton_state` with no reordering.

    `state_interp` is the same `interp1d` `kind` string `build_forward_state`
    takes, and MUST match whatever the forward model this Jacobian is
    differentiating was built with -- passed to both `spec.interp_to` (the
    anchor values, below) and `spec.interp_weights` (the chain-rule weights,
    in the loop below), so the two can never silently disagree about which
    kind is in effect. There is no separate "exact truth" mode to reconcile
    here (retired from `build_forward_state` itself) -- every row, free or
    frozen, goes through the identical interpolation.

    Only `atmosphere` gas rows are handled so far; a free row this module
    cannot yet differentiate raises rather than silently contributing a zero
    column, which would look like a converged-but-unconstrained parameter
    instead of a missing feature.

    `pool` (2026-09-12, user): pass a `LinearizePool` opened once before the
    `gauss_newton_state` call this `linearize` is `jacobian_fn` for, so
    every GN iteration's Pool work reuses the SAME forked workers instead
    of forking fresh ones each time -- see `LinearizePool`'s docstring.
    `None` (default) is the old behavior: a fresh Pool per call, still
    exactly as before for any caller that doesn't opt in (e.g.
    `gd_jacobian_validate.py`, one-off callers with no iteration loop to
    amortize a persistent pool over).
    """
    scene_etas = np.asarray(scene_etas, dtype=float)
    order = np.argsort(scene_etas)
    scene_etas = scene_etas[order]

    free = spec.free_params
    supported = (set(GAS_ROW_MOLECULE) | set(SURFACE_ROW_JACOBIAN)
                | {"p_surface_hpa", "t_offset_k", "height_aerosol"})
    unsupported = [p.name for p in free if p.name not in supported]
    if unsupported:
        raise NotImplementedError(
            f"analytic Jacobian not implemented for {unsupported}. Implemented: "
            f"{sorted(supported)}. dispersion (instrument target) is the "
            f"remaining step -- until then run that row with finite differences.")
    for p in free:
        if p.kind not in ("scale", "absolute"):
            raise NotImplementedError(f"{p.name}: kind={p.kind!r} not yet wired here")

    atm_names = [p.name for p in spec.rows_for("atmosphere")]
    surf_names = [p.name for p in spec.rows_for("surface")]
    vals = spec.interp_to(scene_etas, x, state_interp=state_interp)
    params_at_anchor = [{n: float(vals[n][g]) for n in atm_names}
                        for g in range(scene_etas.size)]
    surf_at_anchor = [{n: float(vals[n][g]) for n in surf_names}
                      for g in range(scene_etas.size)]
    rows_needed = [p.name for p in free]
    S, dS = anchor_spectra_and_derivs(spectrum_jac, params_at_anchor, rows_needed,
                                      surf_at_anchor, n_workers=n_workers, pool=pool)

    def L(field):
        """The detector operator applied to a per-anchor hi-res field.

        2026-09-02 (user): matches `build_forward_state`'s own fix --
        `footprint_average_scene` (per-pixel footprint integration, never
        spectral interpolation) instead of `nearest_bin_scene` (a point
        sample). This is still exactly the same LINEAR composition the
        module docstring describes: `footprint_average_scene`'s own
        zone-overlap weights are purely geometric (independent of the
        `field` values), so pushing a derivative field `dS_k` through it
        gives dY/dx_k correctly, the same way pushing S through it gives
        y -- `nearest_bin_scene` was always the degenerate (single-anchor-
        weight) special case of this, not a different operator. Before
        this fix, `y`/`K` here diverged from `build_forward_state`'s own
        `forward(x)` (used for the GN objective's acceptance checks),
        confirmed directly: a window with a resid=0 prior (exactly
        representable, see docs/PROJECT_STATUS.md) still showed
        rms_resid=0.045 here -- a pure Jacobian-vs-forward-model
        inconsistency, not a real residual.
        """
        return gd_render.predict_neighborhood(
            fpa, rows_win, wn_hires, footprint_average_scene(scene_etas, field),
            ils, pad=pad, footprint=True).ravel()

    y = L(S)
    K = np.empty((y.size, spec.n_free))
    slices = spec.slices()
    K_g = {}   # row name -> (y.size, n_g_positions); empty unless a row declares
              # sub_bin_anomaly.g_cov (2026-08-28)

    # Every K/K_g column is an independent L(field) call -- ~325ms each
    # per this module's own docstring, the dominant per-iteration cost
    # for a window with many free elements (2026-09-02, user: "let's
    # parallelize the linearize step as well"). Collect every (dest,
    # field) pair across ALL free rows first, so one pool spans the
    # WHOLE batch (better load balancing than one pool per row), then
    # scatter results back by destination.
    import multiprocessing as mp
    n_workers_eff = 1 if mp.current_process().daemon else n_workers
    jobs: list = []          # (dest, field) -- dest is ("K", col_idx) or ("Kg", name, m)
    dval_dg_by_name: dict = {}
    for p in free:
        W = spec.interp_weights(scene_etas, p.name, state_interp=state_interp)  # (n_scene, p.n)
        dS_row = np.asarray([d[p.name] for d in dS])     # (n_scene, n_hires)
        sl = slices[p.name]
        for k in range(p.n):
            # d(param at anchor g)/dx_k = W[g,k] * prior[k]   (kind="scale",
            # the retrieved number multiplies the prior) or W[g,k] * 1.0
            # (kind="absolute", the retrieved number IS the physical value
            # -- ParamSpec.apply's own "xi if kind=='absolute'" branch,
            # mirrored here exactly so this module's d(value)/dx factor never
            # silently disagrees with what apply() actually computes).
            dval_dx = p.prior[k] if p.kind == "scale" else 1.0
            jobs.append((("K", sl.start + k), dS_row * (W[:, k] * dval_dx)[:, None]))
        if p.sub_bin_anomaly is not None and p.sub_bin_anomaly.g_cov is not None:
            dval_dg = _g_anomaly_sensitivity(p, scene_etas, W, state_interp)
            dval_dg_by_name[p.name] = dval_dg
            for m in range(dval_dg.shape[1]):
                jobs.append((("Kg", p.name, m), dS_row * dval_dg[:, m][:, None]))

    results = None
    if pool is not None and n_workers_eff > 1:
        results = pool.run_L(jobs, fpa=fpa, rows_win=rows_win, wn_hires=wn_hires,
                             scene_etas=scene_etas, ils=ils, pad=pad)
    if results is None:
        if n_workers_eff <= 1 or len(jobs) < 4:
            results = [(dest, L(field)) for dest, field in jobs]
        else:
            # `jobs` (every K/K_g column's full field array) goes into the
            # fork-inherited global too, not just the small metadata -- see
            # _L_worker's docstring. Workers pick up their own job by index
            # via copy-on-write instead of having it pickled to them.
            _LINEARIZE_L_G.update(dict(fpa=fpa, rows_win=rows_win, wn_hires=wn_hires,
                                       scene_etas=scene_etas, ils=ils, pad=pad, jobs=jobs))
            ctx = mp.get_context("fork")
            with ctx.Pool(min(n_workers_eff, len(jobs))) as one_shot_pool:
                results = one_shot_pool.map(_L_worker, range(len(jobs)), chunksize=1)

    kg_stacks: dict = {name: [None] * dval_dg_by_name[name].shape[1] for name in dval_dg_by_name}
    for dest, col in results:
        if dest[0] == "K":
            K[:, dest[1]] = col
        else:
            kg_stacks[dest[1]][dest[2]] = col
    for name, cols in kg_stacks.items():
        K_g[name] = np.stack(cols, axis=1)
    return y, K, K_g
