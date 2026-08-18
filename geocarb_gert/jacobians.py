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
p_surface_hpa      atmosphere   NOT YET -- composite chain rule, next step
albedo             surface      NOT YET -- gert `K_albedo_hires`
dispersion         instrument   NOT YET -- `_diagonal_ils_convolve_dnu`
=================  ===========  ==================================================

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
from .joint_state import StateSpec

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


def linearize(fpa, rows_win, scene_etas, spec: StateSpec, spectrum_jac,
              wn_hires, ils, x, pad: int = 4):
    """``(y, K)`` -- the predicted sub-image and its analytic Jacobian.

    `K` has one column per free element, ordered exactly as
    :meth:`StateSpec.slices` packs them, so it drops straight into the
    Rodgers step in `gauss_newton_state` with no reordering.

    Only `atmosphere` gas rows are handled so far; a free row this module
    cannot yet differentiate raises rather than silently contributing a zero
    column, which would look like a converged-but-unconstrained parameter
    instead of a missing feature.
    """
    scene_etas = np.asarray(scene_etas, dtype=float)
    order = np.argsort(scene_etas)
    scene_etas = scene_etas[order]

    free = spec.free_params
    supported = set(GAS_ROW_MOLECULE) | set(SURFACE_ROW_JACOBIAN)
    unsupported = [p.name for p in free if p.name not in supported]
    if unsupported:
        raise NotImplementedError(
            f"analytic Jacobian not implemented for {unsupported}. Implemented: "
            f"{sorted(supported)}. p_surface_hpa (composite chain rule) and "
            f"dispersion (instrument target) are the remaining steps -- until "
            f"then run those rows with finite differences.")
    for p in free:
        if p.kind != "scale":
            raise NotImplementedError(f"{p.name}: kind={p.kind!r} not yet wired here")

    atm_names = [p.name for p in spec.rows_for("atmosphere")]
    surf_names = [p.name for p in spec.rows_for("surface")]
    vals = spec.interp_to(scene_etas, x)
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
    for p in free:
        W = spec.interp_weights(scene_etas, p.name)      # (n_scene, p.n)
        dS_row = np.asarray([d[p.name] for d in dS])     # (n_scene, n_hires)
        sl = slices[p.name]
        for k in range(p.n):
            # d(param at anchor g)/dx_k = W[g,k] * prior[k]   (kind="scale")
            K[:, sl.start + k] = L(dS_row * (W[:, k] * p.prior[k])[:, None])
    return y, K
