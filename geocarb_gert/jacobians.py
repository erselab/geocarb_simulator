"""Analytic Jacobians for the joint-block state vector.

Replaces `gauss_newton_state`'s finite differences with derivatives assembled
from `gert`'s own per-layer analytic arrays, composed with this package's
detector operator. **`gert` itself is not modified** -- everything here reads
what `ForwardModel.run(jacobians=True)` already returns.

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

    dY/dx_k = predict_neighborhood(..., radiance=nearest_bin_scene(etas, dS_k))

with `dS_k[g] = (dS/dparam_r)[g] * W[g, k] * prior_r[k]`, where `W` is
`StateSpec.interp_weights` (the state-space interpolation written as a
matrix) and the `prior_r[k]` factor is `d(value)/dx` for `kind="scale"`.

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
albedo             surface      gert `K_albedo_hires` -- exact, nothing to chain
albedo_slope       surface      gert `K_slope_hires`  -- exact (spectral slope)
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
from .focalplane import nearest_bin_scene
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


def p_surface_dI_dparam(res, params, window: int = 0, h_rel: float = 1e-6):
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


def make_spectrum_jac(absco, wide_inst, geo, solar, albedo):
    """``spectrum_jac(params, rows) -> (S_hires, {row: dS/d(param)})``.

    The analytic counterpart of the sweep's own `_make_state_spectrum`, and
    deliberately built the same way -- straight from
    `along_slit_scene.atmosphere_from_params`, not `StateVector.gas_scaling`,
    so no quantity is privileged. The only difference is
    ``SingleScatterSolver(jacobians=True)`` and ``run(jacobians=True)``, which
    is what populates the per-layer arrays :func:`gas_dI_dparam` reads.
    """
    from gert.forward_model import ForwardModel
    from gert.rt_solver import SingleScatterSolver

    from . import along_slit_scene as als

    def spectrum_jac(params: dict, rows, surface: dict | None = None):
        atm = als.atmosphere_from_params(**params)
        # a free albedo row overrides the fixed scalar this factory was built
        # with; without one, behaviour is identical to `_make_state_spectrum`
        alb = float((surface or {}).get("albedo", albedo))
        slope = float((surface or {}).get("albedo_slope", 0.0))
        fm = ForwardModel(atm, absco, wide_inst, geo,
                          solver=SingleScatterSolver(jacobians=True),
                          solar_spectrum=solar)
        res = fm.run(albedo=np.array([alb]), albedo_slope=np.array([slope]),
                     jacobians=True)
        S = np.asarray(res.I_hires[0], dtype=float)
        d = {}
        for row in rows:
            if row in SURFACE_ROW_JACOBIAN:
                d[row] = surface_dI_dparam(res, row)
            elif row == "p_surface_hpa":
                d[row] = p_surface_dI_dparam(res, params)
            elif row in GAS_ROW_MOLECULE:
                d[row] = gas_dI_dparam(res, GAS_ROW_MOLECULE[row], params[row])
            else:
                raise NotImplementedError(f"no analytic derivative for row {row!r}")
        return S, d

    return spectrum_jac


def anchor_spectra_and_derivs(spectrum_jac, params_at_anchor, rows_needed,
                              surface_at_anchor=None):
    """Run the forward model once per anchor, returning radiance and the
    hi-res derivative of each requested row.

    `spectrum_jac(params, rows, surface) -> (S, {row_name: dS/d(param)})` is
    supplied by the caller (see :func:`make_spectrum_jac`) so this module
    stays independent of how the band was configured.
    """
    S, dS = [], []
    for g, p in enumerate(params_at_anchor):
        surf = surface_at_anchor[g] if surface_at_anchor else None
        s, d = spectrum_jac(p, rows_needed, surf)
        S.append(np.asarray(s, dtype=float))
        dS.append(d)
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


def linearize(fpa, rows_win, scene_etas, spec: StateSpec, spectrum_jac,
              wn_hires, ils, x, pad: int = 4, state_interp: str = "linear"):
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
    """
    scene_etas = np.asarray(scene_etas, dtype=float)
    order = np.argsort(scene_etas)
    scene_etas = scene_etas[order]

    free = spec.free_params
    supported = set(GAS_ROW_MOLECULE) | set(SURFACE_ROW_JACOBIAN) | {"p_surface_hpa"}
    unsupported = [p.name for p in free if p.name not in supported]
    if unsupported:
        raise NotImplementedError(
            f"analytic Jacobian not implemented for {unsupported}. Implemented: "
            f"{sorted(supported)}. dispersion (instrument target) is the "
            f"remaining step -- until then run that row with finite differences.")
    for p in free:
        if p.kind != "scale":
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
                                      surf_at_anchor)

    def L(field):
        """The detector operator applied to a per-anchor hi-res field."""
        return gd_render.predict_neighborhood(
            fpa, rows_win, wn_hires, nearest_bin_scene(scene_etas, field),
            ils, pad=pad).ravel()

    y = L(S)
    K = np.empty((y.size, spec.n_free))
    slices = spec.slices()
    K_g = {}   # row name -> (y.size, n_g_positions); empty unless a row declares
              # sub_bin_anomaly.g_cov (2026-08-28)
    for p in free:
        W = spec.interp_weights(scene_etas, p.name, state_interp=state_interp)  # (n_scene, p.n)
        dS_row = np.asarray([d[p.name] for d in dS])     # (n_scene, n_hires)
        sl = slices[p.name]
        for k in range(p.n):
            # d(param at anchor g)/dx_k = W[g,k] * prior[k]   (kind="scale")
            K[:, sl.start + k] = L(dS_row * (W[:, k] * p.prior[k])[:, None])
        if p.sub_bin_anomaly is not None and p.sub_bin_anomaly.g_cov is not None:
            dval_dg = _g_anomaly_sensitivity(p, scene_etas, W, state_interp)
            K_g[p.name] = np.stack([L(dS_row * dval_dg[:, m][:, None])
                                    for m in range(dval_dg.shape[1])], axis=1)
    return y, K, K_g
