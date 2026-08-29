"""A freezable, per-element state-vector spec for the joint bin retrieval.

Replaces the joint block's hard-coded "only ``co2_scale`` is free, everything
else is pinned to local truth" arrangement with one generic mechanism in
which **no parameter is special**. CO2 is a row in a table exactly like
surface pressure or H2O; making it frozen and H2O free is a flag, not a code
change.

This is ``JOINT_BLOCK_MIGRATION_PLAN.md`` Sec.4.0's "regularization as a
per-element property, not per-group code", extended to cover freeze/unfreeze
as well as the prior: each row carries its own positions, prior, sigma and
correlation length, and one generic assembly builds the packed vector, the
prior covariance, and the per-anchor state dict from those rows.

Why it matters beyond tidiness. Measured on the realistic along-slit scene,
the joint block's post-fit residual is dominated not by CO2 but by the
quantities held fixed: per-row residual RMS correlates +0.94 with surface
pressure's own nearest-anchor representation error across the topographic
depression, and +0.89 with H2O's globally, versus +0.59 for CO2 and +0.38
for keystone. Any honest next step needs those quantities to be unfreezable
one at a time, which is exactly what this provides.

Design notes
------------
* **Positions are per row.** A row's ``positions`` are its own eta bin
  centres; different rows may use different bin densities (Sec.4.1's point
  that albedo probably wants denser bins and a shorter correlation length
  than a well-mixed gas). Nothing assumes a shared grid.
* **Scale vs. absolute.** ``kind="scale"`` means the retrieved number
  multiplies the prior (today's ``co2_scale`` convention); ``kind="absolute"``
  means it replaces it. Surface pressure and albedo are more natural as
  absolute, gases as scale -- but that is a per-row choice, not a global one.
* **Prior covariance.** Built per row as
  ``Sa[i,j] = sigma^2 * exp(-|eta_i - eta_j| / corr_length)`` -- the
  exponential-correlation Gauss-Markov form Sec.4.0 showed today's
  ``gamma*(L^T L) + I/sigma_abs^2`` is already an implicit, awkwardly
  parameterized special case of. ``corr_length`` is in eta, so it is
  bin-spacing independent, unlike the current first-difference operator.
* **Multi-FPA ready.** Nothing here references an FPA or a wavelength: rows
  live in eta, which is the band-independent along-slit coordinate. A
  per-FPA quantity (dispersion, or Sec.5's per-band albedo) is a row whose
  ``positions`` are per-FPA points, or several rows, without changing this
  code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d


#: Default prior correlation length per state row, in eta (1 eta = 1400 km,
#: 1 detector row ~ 0.00195 eta = 2.73 km). Each is set from the PHYSICAL
#: scale of that quantity's own structure in `along_slit_scene`, not from a
#: bin count -- which is the whole point of measuring distance in eta rather
#: than bin index: these numbers stay correct when the bin density changes.
#:
#: For scale, the prior these replace (`prior_form="tikhonov"` with
#: gamma=3.0, sigma=0.10) had an implied decorrelation length of ~0.28 bins,
#: i.e. 0.77 km at one bin per row -- BELOW the 2.73 km ground sample
#: distance. It imposed essentially no spatial correlation at all, which is
#: the real explanation for Sec.9's puzzling null result that changing gamma
#: by 100x moved the bias under 2%: the smoothness spring was never actually
#: engaged, only the absolute one.
#:
#: CAVEAT on p_surface (user, 2026-08-17): this value is taken from the
#: scene's own topographic depression (`p_surface_hpa`'s `mountain`, width
#: 140 km), because in THIS synthetic scene surface pressure varies only
#: with topography. In real retrievals the dominant source of
#: surface-pressure-related bias is SCATTERING HEIGHT uncertainty (aerosol
#: and cloud layers), not genuine surface pressure variability -- and that
#: has its own, generally shorter and more variable, spatial scale. So this
#: default is right for the current scene and should NOT be carried over
#: uncritically to real data or to a scene with aerosols.
# Sourced from input/retrieval_defaults.yml's correlation_length_eta: block
# at import time (Phase C of the config-consolidation plan) -- was a
# typed-inline dict. This is a retrieval-methodology default (how much
# structure each row's prior assumes), not instrument physics, hence
# RetrievalDefaults rather than GeoCarbInstrumentConfig -- kept as a module
# constant since it's read bare below and would otherwise need every
# caller updated. Per-row rationale (co2_ppm ~10km hot-spot scale,
# p_surface_hpa ~140km topographic depression, albedo ~30km
# ALBEDO_CORR_KM/SLIT_HALF_KM, etc.) is unchanged and lives in
# retrieval_defaults.yml's own comments now.
from .mission_config import RetrievalDefaults as _RetrievalDefaults  # noqa: E402
DEFAULT_CORR_LENGTH_ETA = _RetrievalDefaults.from_yaml().corr_length_eta


#: Where a state row enters the forward model. This is not cosmetic
#: bookkeeping: it decides which code consumes the row, and where its
#: Jacobian comes from.
#:
#: ``atmosphere``  co2/ch4/co/h2o/p_surface -- interpolated onto the anchor
#:                 etas, fed to `along_slit_scene.atmosphere_from_params`,
#:                 and into the RT. Jacobian from gert's per-layer arrays
#:                 (`K_mol_lay_hires`, `dtau_mol_dT_lay_hires`,
#:                 `dtau_mol_dpscale_lay_hires`, `K_ray_lay_hires`).
#: ``surface``     albedo and its spectral slope -- also eta-positioned, but
#:                 NOT part of `AtmosphericProfile`; they are passed
#:                 separately to `ForwardModel.run`. Jacobian from gert's
#:                 `K_albedo_hires` / `K_slope_hires`, which are exact and
#:                 come free with `jacobians=True`.
#: ``instrument``  dispersion -- touches neither the atmosphere nor the
#:                 scene. It moves each detector column's ILS centre, so it
#:                 is NOT eta-positioned (its positions are per-FPA, not
#:                 along-slit) and must never be interpolated onto anchors.
#:                 Its Jacobian is the only one gert cannot supply; see
#:                 `gd_render._diagonal_ils_convolve_dnu`.
TARGETS = ("atmosphere", "surface", "instrument")

#: Targets whose rows live in eta and are interpolated onto scene positions.
#: `instrument` is deliberately absent -- see TARGETS.
ETA_TARGETS = ("atmosphere", "surface")


def _row_interp1d(positions, values, kind: str, axis: int = -1):
    """One row's own `scipy.interpolate.interp1d`, `kind`-agnostic, clamped
    at the ends (matches `np.interp`'s own behaviour there, unlike
    `interp1d`'s default of raising outside the data range). Shared by
    :meth:`StateSpec.interp_to` and :meth:`StateSpec.interp_weights` so the
    two can never disagree about what a given `kind` does -- the weights
    method calls this on the identity matrix instead of real values, not on
    a separately-derived formula.
    """
    values = np.asarray(values)
    lo = values[0] if axis == -1 else values[0, ...]
    hi = values[-1] if axis == -1 else values[-1, ...]
    return interp1d(positions, values, kind=kind, axis=axis,
                    bounds_error=False, fill_value=(lo, hi))


@dataclass
class SubBinAnomaly:
    """A row's OPT-IN higher-resolution reference profile `g`, riding
    additively on top of the plain piecewise-linear bin interpolant
    (2026-08-28; see `StateSpec.interp_to`'s own docstring for the exact
    formula and why it is additive, not multiplicative).

    `g_at_centers` (`g_fn` evaluated once at this row's own bin `positions`,
    precomputed at construction, not per call) is what makes the anomaly
    correction vanish exactly wherever `g` is itself piecewise-linear on
    this same grid -- the property this whole mechanism exists for.

    `g_positions`/`g_cov` are optional and only needed to propagate `g`'s
    OWN uncertainty into the retrieved posterior (`gauss_newton_state`'s
    `g_cov` parameter) -- `g_cov=None` (default) means "no uncertainty
    machinery for this row at all", a strict no-op. `Wg_at_centers` (this
    row's bin centers expressed as `g_positions`' own `interp_weights`
    matrix) is precomputed alongside `g_positions` since it's needed by
    `jacobians._g_anomaly_sensitivity` and never depends on the retrieved
    state.
    """

    g_fn: Callable
    g_at_centers: np.ndarray
    g_positions: np.ndarray | None = None
    g_cov: np.ndarray | None = None
    Wg_at_centers: np.ndarray | None = None


@dataclass
class ParamSpec:
    """One state-vector element: where it lives, its prior, and whether it
    is currently being retrieved."""

    name: str
    positions: np.ndarray          # eta bin centres for this parameter
    prior: np.ndarray              # prior value at each position
    sigma: float                   # prior 1-sigma (in the parameter's own units)
    corr_length: float = 0.05      # correlation length in eta ("exponential" prior only)
    free: bool = True              # False -> frozen at `prior`, contributes no elements
    kind: str = "scale"            # "scale" (multiplies prior) or "absolute"
    prior_form: str = "exponential"  # "exponential" (default) or "tikhonov"
    gamma: float = 3.0             # tikhonov: first-difference smoothness strength
    target: str = "atmosphere"     # where this row enters the forward model; see TARGETS
    state_interp: str | None = None  # this row's OWN interp1d kind override (2026-08-25);
                                     # None -> fall back to whatever the caller passes to
                                     # interp_to/interp_weights (today's behaviour,
                                     # unchanged for every row that never sets this). Lets
                                     # one row use "nearest" (a hard-edged feature, e.g. an
                                     # albedo patch boundary) while another shares the same
                                     # bin positions but stays "linear" (a smooth feature,
                                     # e.g. a CO2 hot spot) -- see StateSpec.interp_to/
                                     # interp_weights for the resolution rule.
    sub_bin_anomaly: SubBinAnomaly | None = None  # this row's opt-in additive
                                     # high-resolution reference profile (2026-08-28);
                                     # None (every row, until one opts in) is a
                                     # byte-identical no-op. See SubBinAnomaly.

    def __post_init__(self):
        self.positions = np.atleast_1d(np.asarray(self.positions, dtype=float))
        self.prior = np.atleast_1d(np.asarray(self.prior, dtype=float))
        if self.prior.size == 1 and self.positions.size > 1:
            self.prior = np.full(self.positions.size, float(self.prior[0]))
        if self.prior.shape != self.positions.shape:
            raise ValueError(f"{self.name}: prior {self.prior.shape} does not match "
                             f"positions {self.positions.shape}")
        if self.kind not in ("scale", "absolute"):
            raise ValueError(f"{self.name}: kind must be 'scale' or 'absolute', got {self.kind!r}")
        if self.target not in TARGETS:
            raise ValueError(f"{self.name}: target must be one of {TARGETS}, got {self.target!r}")

    @property
    def is_eta_positioned(self) -> bool:
        """True when this row's `positions` are along-slit eta, so it can be
        interpolated onto scene anchors. False for `instrument` rows."""
        return self.target in ETA_TARGETS

    @property
    def n(self) -> int:
        return int(self.positions.size)

    def x0(self) -> np.ndarray:
        """This row's own starting/identity vector: ones for a scale
        parameter, the prior itself for an absolute one."""
        return np.ones(self.n) if self.kind == "scale" else self.prior.copy()

    def apply(self, xi: np.ndarray) -> np.ndarray:
        """Parameter values at `positions` given this row's own sub-vector."""
        xi = np.asarray(xi, dtype=float)
        return self.prior * xi if self.kind == "scale" else xi

    def Sa_block(self) -> np.ndarray:
        """Prior COVARIANCE for this row alone (exponential form only)."""
        d = np.abs(self.positions[:, None] - self.positions[None, :])
        return self.sigma ** 2 * np.exp(-d / max(self.corr_length, 1e-12))

    def Sa_inv_block(self) -> np.ndarray:
        """Prior PRECISION for this row alone.

        Two forms, and they are NOT interchangeable despite Sec.4.0 showing
        they belong to the same family:

        ``tikhonov`` (default) reproduces the original
        ``gauss_newton_regularized`` exactly:
        ``gamma*(L^T L) + I/sigma**2`` with ``L`` the first-difference
        operator. Distance is measured in BIN INDEX, so every adjacent pair
        couples equally regardless of eta separation, and the edge bins get
        the free/natural boundary condition (one neighbour instead of two)
        that Sec.9 measured -- an edge/centre marginal-std ratio of 1.014 at
        sigma=0.10, rising to 1.251 as sigma loosens.

        ``exponential`` builds the covariance ``sigma^2 exp(-|eta_i-eta_j|/
        corr_length)`` and inverts it. Distance is physical eta, so
        non-uniform bin spacing (which `pixel_density_bin_centers` always
        produces) is handled correctly, and sigma/corr_length are
        independent knobs. But inverting a stationary exponential covariance
        yields compensating corner terms, so it has NO edge weakening --
        a real behavioural difference at every window boundary, since
        production windows tile non-overlapping.

        Defaulted to ``tikhonov`` through 2026-08-17 for continuity with
        every result computed before then. As of the 2026-08-20 config
        consolidation, ``exponential`` (with per-row correlation lengths
        from `input/retrieval_defaults.yml`'s `correlation_length_eta`,
        e.g. ~10 km for `co2_ppm`) is the actual default -- this is also
        the prior form whose lack of edge-weakening produces the
        window-boundary overshoot/undershoot bias documented in
        docs/PROJECT_STATUS.md Sec.6.
        """
        n = self.n
        if self.prior_form == "tikhonov":
            if n == 1:
                return np.eye(1) / self.sigma ** 2
            L = np.zeros((n - 1, n))
            for k in range(n - 1):
                L[k, k] = -1.0
                L[k, k + 1] = 1.0
            return self.gamma * (L.T @ L) + np.eye(n) / self.sigma ** 2
        if self.prior_form == "exponential":
            return np.linalg.inv(self.Sa_block())
        raise ValueError(f"{self.name}: unknown prior_form {self.prior_form!r} "
                         f"(expected 'tikhonov' or 'exponential')")


class StateSpec:
    """An ordered collection of :class:`ParamSpec` rows, with packing that
    includes only the currently-free ones.

    Freezing a parameter shrinks the retrieved vector and leaves its value
    at the prior; unfreezing grows it back. No other code needs to know
    which rows are free.
    """

    def __init__(self, params):
        self.params = list(params)
        names = [p.name for p in self.params]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate parameter names: {names}")

    # -- freezing ---------------------------------------------------------
    def __getitem__(self, name: str) -> ParamSpec:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(f"{name!r} not in state ({[p.name for p in self.params]})")

    def freeze(self, *names: str) -> "StateSpec":
        for n in names:
            self[n].free = False
        return self

    def unfreeze(self, *names: str) -> "StateSpec":
        for n in names:
            self[n].free = True
        return self

    def set_free(self, **flags) -> "StateSpec":
        """`spec.set_free(co2_ppm=False, p_surface_hpa=True)`."""
        for n, v in flags.items():
            self[n].free = bool(v)
        return self

    @property
    def free_params(self):
        return [p for p in self.params if p.free]

    @property
    def free_names(self):
        return [p.name for p in self.free_params]

    @property
    def n_free(self) -> int:
        return sum(p.n for p in self.free_params)

    # -- packing ----------------------------------------------------------
    def x0(self) -> np.ndarray:
        """Starting vector over the free rows only."""
        blocks = [p.x0() for p in self.free_params]
        return np.concatenate(blocks) if blocks else np.zeros(0)

    def slices(self) -> dict:
        """`{name: slice}` into the packed vector, free rows only."""
        out, i = {}, 0
        for p in self.free_params:
            out[p.name] = slice(i, i + p.n)
            i += p.n
        return out

    def unpack(self, x) -> dict:
        """`{name: values at that row's own positions}` for EVERY row --
        free rows from `x`, frozen rows from their prior."""
        x = np.asarray(x, dtype=float)
        if x.size != self.n_free:
            raise ValueError(f"expected {self.n_free} free elements, got {x.size}")
        sl = self.slices()
        return {p.name: (p.apply(x[sl[p.name]]) if p.free else p.apply(p.x0()))
                for p in self.params}

    def Sa_inv(self) -> np.ndarray:
        """Block-diagonal prior precision over the free rows only."""
        blocks = [p.Sa_inv_block() for p in self.free_params]
        if not blocks:
            return np.zeros((0, 0))
        n = sum(b.shape[0] for b in blocks)
        out, i = np.zeros((n, n)), 0
        for b in blocks:
            k = b.shape[0]
            out[i:i + k, i:i + k] = b
            i += k
        return out

    # -- consumption by a forward model -----------------------------------
    def rows_for(self, *targets):
        """The rows belonging to the given targets, in state order."""
        return [p for p in self.params if p.target in targets]

    def interp_to(self, etas, x, targets=ETA_TARGETS, state_interp: str = "linear") -> dict:
        """`{name: value}` interpolated onto `etas`, for the eta-positioned
        rows -- the state-space interpolation a forward model needs to build
        one atmosphere per anchor.

        `state_interp` names a `scipy.interpolate.interp1d` `kind` --
        `"linear"` (default, today's behaviour) or `"nearest"` (piecewise-
        constant: each anchor takes its single nearest row-position's own
        value, no blending). Any other `interp1d`-supported kind (e.g.
        `"quadratic"`, `"cubic"`) works too with no code change here -- see
        :func:`_row_interp1d`. This is the FALLBACK kind, used for any row
        whose own `ParamSpec.state_interp` is `None` (every row, until one
        opts in) -- a row with its own `state_interp` set always uses that
        instead, letting different rows share the same bin positions with
        genuinely different interpolation behaviour (2026-08-25).

        `instrument` rows are EXCLUDED by default and must be: their
        `positions` are not along-slit coordinates, so interpolating them
        against `etas` would be meaningless arithmetic on mismatched axes
        rather than a harmless no-op. Use :meth:`values_for` to read them.

        A row with `ParamSpec.sub_bin_anomaly` set (2026-08-28) additionally
        picks up `g`'s own deviation from ITS OWN plain interpolant on this
        row's positions: ``value(eta) = plain_interp(eta) + (g(eta) -
        plain_interp_of_g(eta))``. This vanishes identically (not just at
        the bin centers) whenever `g` is itself piecewise-linear on this
        row's grid -- the exact-reduction property the mechanism is built
        around -- and always leaves `value(bin_center_k) == bin_value_k`
        exactly, for any `g`, since both interpolants agree exactly at
        their own nodes. `None` (every row, until one opts in) is a
        byte-identical no-op.
        """
        etas = np.asarray(etas, dtype=float)
        vals = self.unpack(x)
        out = {}
        for p in self.params:
            if p.target not in targets:
                continue
            v = vals[p.name]
            kind = p.state_interp if p.state_interp is not None else state_interp
            plain = (np.full(etas.shape, v[0]) if p.n == 1
                     else _row_interp1d(p.positions, v, kind)(etas))
            sba = p.sub_bin_anomaly
            if sba is not None:
                g_at_etas = np.asarray(sba.g_fn(etas), dtype=float)
                g_plain = (np.full(etas.shape, sba.g_at_centers[0]) if p.n == 1
                          else _row_interp1d(p.positions, sba.g_at_centers, kind)(etas))
                plain = plain + (g_at_etas - g_plain)
            out[p.name] = plain
        return out

    def values_for(self, x, *targets) -> dict:
        """`{name: values at that row's own positions}`, uninterpolated.

        The read path for `instrument` rows, whose positions are not eta.
        """
        vals = self.unpack(x)
        return {p.name: vals[p.name] for p in self.params if p.target in targets}

    def interp_weights(self, etas, name, state_interp: str = "linear") -> np.ndarray:
        """`(len(etas), n)` matrix `W` with `W[g, k] = d(param at etas[g]) /
        d(this row's value at position k)`.

        The interpolation operator written out as a matrix, built by
        interpolating the basis vectors (the `n x n` identity, one column per
        position) in a single vectorized `interp1d` call rather than
        rederiving each kind's own weight formula -- so it cannot drift from
        what :meth:`interp_to` actually does, including at and beyond the end
        points (clamped, not extrapolated, matching `interp_to`), and
        including non-local kinds: `interp1d` is linear in its `y` data for
        any fixed `kind`, so interpolating the identity matrix always gives
        the exact weight matrix, whether that kind's support is 2 points
        (linear/nearest) or the whole row (a spline).

        This is the whole chain rule from a row's own elements to the anchor
        values a forward model sees. Combined with `d(value)/dx = prior` for
        `kind="scale"`, it gives `d(param at anchor g)/dx_k` exactly.

        `state_interp` is the FALLBACK kind, overridden by this row's own
        `ParamSpec.state_interp` when set -- same resolution rule as
        `interp_to`, so the two can never disagree about which kind is
        actually in effect for a given row.

        Deliberately UNAFFECTED by `ParamSpec.sub_bin_anomaly` (2026-08-28):
        that mechanism adds `g(eta) - plain_interp_of_g(eta)`, a term that
        depends only on `g` and this row's own fixed positions, never on
        the retrieved bin values themselves -- so `d(value)/d(bin_value_k)`
        is exactly this same `W` whether or not a row has an anomaly set.
        `jacobians.linearize`'s state Jacobian `K` needs no change either,
        for the same reason.
        """
        p = self[name]
        etas = np.asarray(etas, dtype=float)
        if p.n == 1:
            return np.ones((etas.size, 1))
        kind = p.state_interp if p.state_interp is not None else state_interp
        basis = np.eye(p.n)
        return _row_interp1d(p.positions, basis, kind, axis=0)(etas)

    def cov_for(self, name: str, S_ret_scale: np.ndarray) -> np.ndarray:
        """One row's own ``(n, n)`` posterior covariance sub-matrix, in
        physical units -- sliced out of `gauss_newton_state(...,
        return_cov=True)`'s own ``S_ret_scale`` (packed over ALL free
        rows, in scale units -- see `slices()`) and converted with this
        row's own `prior`: ``kind="scale"`` -> ``Cov_phys[i,j] =
        prior[i]*prior[j]*S_ret_scale[i,j]`` (the delta-method Jacobian of
        ``value = prior * x`` is just `prior` itself, elementwise);
        ``kind="absolute"`` -> already physical, no conversion.

        A frozen row has no posterior uncertainty by construction (it
        never moved from its prior, `Sa`/`Sy` were never consulted for
        it) -- returns a zero matrix of the right shape rather than
        raising, so callers iterating over every row (frozen and free
        alike, e.g. building a full snapshot) don't need to special-case
        frozen ones.
        """
        p = self[name]
        if not p.free:
            return np.zeros((p.n, p.n))
        sl = self.slices()[name]
        block = np.asarray(S_ret_scale, dtype=float)[sl, sl]
        if p.kind == "scale":
            return block * np.outer(p.prior, p.prior)
        return block

    def project_cov(self, etas, name: str, S_ret_scale: np.ndarray,
                    state_interp: str = "linear") -> np.ndarray:
        """This row's posterior covariance projected onto arbitrary eta
        positions, through the SAME linear interpolation `interp_to`/
        `interp_weights` use for the point estimate: ``Var(x_hat(eta)) =
        W @ Cov_row @ W.T`` -- exact, not an approximation, since
        `interp_to` is itself exactly ``value = W @ row_values`` for a
        fixed `state_interp` kind (see `interp_weights`'s own docstring).

        Returns the FULL ``(len(etas), len(etas))`` covariance, including
        off-diagonal terms between the query points -- e.g. needed to
        propagate uncertainty correctly through a further downstream
        linear combination (a window-boundary blend, say). Take
        ``.diagonal()`` for just the per-point variance.
        """
        W = self.interp_weights(etas, name, state_interp)
        cov_row = self.cov_for(name, S_ret_scale)
        return W @ cov_row @ W.T

    def snapshot(self, x, **extra) -> dict:
        """A complete, self-describing record of this solve's state vector.

        Standing project rule: every run saves the ENTIRE state vector and
        all residuals, not just the quantity of interest. That includes the
        FROZEN rows -- "what was held fixed, and at what value" is exactly
        what is needed to interpret a result months later, and it is not
        recoverable from the free rows alone. It is also what makes two runs
        with different free/frozen splits comparable at all.

        Every row records its positions, prior, retrieved values, whether it
        was free, its kind, and its prior sigma/correlation length, so the
        record reproduces the solve without reference to the code that
        produced it.
        """
        x = np.asarray(x, dtype=float)
        vals = self.unpack(x)
        return dict(
            params={p.name: dict(positions=p.positions.copy(), prior=p.prior.copy(),
                                 values=vals[p.name].copy(), free=bool(p.free),
                                 kind=p.kind, sigma=float(p.sigma),
                                 corr_length=float(p.corr_length))
                    for p in self.params},
            x=x.copy(), free_names=list(self.free_names), n_free=int(self.n_free),
            slices={k: (v.start, v.stop) for k, v in self.slices().items()},
            **extra)

    def __repr__(self) -> str:
        rows = ", ".join(f"{p.name}[{p.n}]{'' if p.free else ' FROZEN'}" for p in self.params)
        return f"StateSpec({rows}; n_free={self.n_free})"


def albedo_positions_for(bin_centers, density: int = 3) -> np.ndarray:
    """A denser bin grid for the surface rows, spanning the same eta range.

    Surface albedo is the one state row whose truth structure is SHORTER
    than the gas features (`DEFAULT_CORR_LENGTH_ETA["albedo"]` ~ 30 km,
    against 100-500 km for the gases and surface pressure), so sharing the
    gas grid would under-resolve it by construction. `density` bins per gas
    bin, uniformly spaced across the same span -- uniform rather than
    pixel-density-weighted, because albedo structure is a property of the
    GROUND, with no reason to cluster where the detector's own keystone
    happens to put more pixels.

    Note this raises the free-element count: `density=3` on a 31-bin window
    adds 91 albedo elements to 31 CO2 ones. That is the honest cost of
    resolving a short-correlation-length quantity, but it is worth knowing
    before freeing albedo on a wide window.
    """
    bc = np.atleast_1d(np.asarray(bin_centers, dtype=float))
    if bc.size == 1 or density <= 1:
        return bc.copy()
    return np.linspace(bc.min(), bc.max(), (bc.size - 1) * int(density) + 1)


def information_weighted_bin_centers(eta_lo: float, eta_hi: float, G: int,
                                     weight_fn, n_grid: int = 2000) -> np.ndarray:
    """`G` bin centers between `eta_lo`/`eta_hi`, at equal quantiles of
    `weight_fn`'s own cumulative density -- the same equal-population
    placement principle `gd_joint_block_diagnostics.pixel_density_bin_
    centers` uses for raw pixel density, generalized to an arbitrary
    weight FUNCTION instead of a sample array (2026-08-25).

    `pixel_density_bin_centers` calls `np.quantile` directly on real pixel
    eta samples -- correct there because "one sample = one unit of desired
    density" already holds. `np.quantile` has no weight argument, so a
    density *function* (rather than samples already drawn proportional to
    it) needs its own inverse-CDF construction: evaluate `weight_fn` on a
    fine grid, build its empirical CDF, then invert (interpolate) at `G`
    equally-spaced quantile levels. A constant `weight_fn` reproduces a
    uniform grid; this deliberately does NOT reproduce `pixel_density_bin_
    centers` bit-for-bit even when fed something proportional to pixel
    density, since it is quantizing a continuous density estimate, not
    resampling the original discrete pixels.

    `ParamSpec.positions` (and everything downstream -- `Sa_block`/`Sa_inv_
    block`, `interp_weights`/`_row_interp1d`) already handles arbitrary
    irregular positions under the `exponential` prior form (today's
    default), so bins placed this way need no other change to use.
    """
    eta_grid = np.linspace(float(eta_lo), float(eta_hi), int(n_grid))
    w = np.maximum(np.asarray(weight_fn(eta_grid), dtype=float), 1e-12)
    cdf = np.cumsum(w)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    return np.interp(np.linspace(0.0, 1.0, int(G)), cdf, eta_grid)


def combined_information_weighted_bin_centers(eta_lo: float, eta_hi: float, G: int,
                                               weight_fns: dict, n_grid: int = 2000) -> np.ndarray:
    """`G` bin centers shared across every row named in `weight_fns`,
    placed by the ELEMENTWISE MAX across their own individual weight
    functions (2026-08-25) -- the shared grid refines wherever ANY row
    wants it (a CO2 hot spot here, an albedo patch boundary there),
    instead of one row's own need being diluted by averaging with a row
    that doesn't care about that location. `weight_fns` values are only
    used for their VALUES, not their keys -- the dict is keyed by row name
    purely so a caller's own weight-function collection stays
    self-documenting at the call site (e.g. `{"albedo": als.albedo_info_
    density, "co2_ppm": some_hotspot_proxy}`).

    Thin wrapper around `information_weighted_bin_centers` -- no duplicated
    inverse-CDF logic, this only builds its `weight_fn` argument.
    """
    def combined(eta):
        return np.maximum.reduce([np.asarray(fn(eta), dtype=float) for fn in weight_fns.values()])
    return information_weighted_bin_centers(eta_lo, eta_hi, G, combined, n_grid=n_grid)


def state_spec_from_scene(bin_centers, fields=None, free=("co2_ppm",),
                          sigmas=None, corr_length=None, kinds=None,
                          prior_form="exponential", gamma=3.0,
                          band_label=None, surface_positions=None,
                          surface_density: int = 3, uniform: bool = False,
                          prior_anchor_density: float | None = None,
                          surface_fields=None,
                          row_state_interp: dict | None = None,
                          row_sub_bin_anomaly: dict | None = None) -> StateSpec:
    """Build a :class:`StateSpec` whose priors are the truth scene's own
    values at `bin_centers` -- the joint block's existing "local-truth
    nuisance idealization", but now with every quantity present as a real,
    unfreezable row instead of being pinned invisibly inside the forward
    model.

    `free` names the rows that start free; everything else starts frozen, so
    the default `free=("co2_ppm",)` reproduces today's CO2-only retrieval
    exactly while making the other four one flag away from being retrieved.

    `corr_length` may be a float (applied to every row), a dict keyed by row
    name, or None (the default) to use :data:`DEFAULT_CORR_LENGTH_ETA` --
    each row's own physical scale. Passing a single float is what the
    original bin-index prior effectively did and is usually NOT what you
    want, since surface pressure and a CO2 hot spot do not share a scale.

    `band_label` (a `SpectralWindow.label`, e.g. "CO2_strong") adds the
    `surface`-target rows from `surface_fields` (default `along_slit_scene.
    SURFACE_FIELDS`, i.e. the exact truth -- pass `along_slit_scene.
    SURFACE_PRIOR_FIELD_SETS[key]` for an imperfect surface prior, the same
    way `fields` selects an imperfect atmosphere prior via `PRIOR_FIELD_SETS`;
    added 2026-08-25, previously hardcoded to `SURFACE_FIELDS`). Omitting
    `band_label` reproduces the pre-2026-08-18 atmosphere-only state
    exactly, so every existing caller is unaffected.

    Surface rows now share `bin_centers` -- the SAME positions the
    atmosphere rows use -- by default (changed 2026-08-25; was `albedo_
    positions_for(bin_centers, surface_density)`, a separate, uniformly
    3x-denser grid). Pass `surface_positions` explicitly for a genuinely
    different grid (an information-weighted one via `joint_state.
    information_weighted_bin_centers`/`combined_information_weighted_bin_
    centers`, or the old flat-oversampling behaviour via `albedo_positions_
    for(bin_centers, surface_density)` computed by the caller) -- `surface_
    density` itself is now IGNORED unless the caller uses it to build that
    override. This is a real, deliberate behaviour change for any caller
    that frees a surface row and does NOT pass `surface_positions`
    (`scripts/gd_joint_block_whole_slit_sweep.py`'s own production path
    included) -- previously-banked results with a free albedo row
    (`docs/PROJECT_STATUS.md` Sec.7) used the old separate-grid behaviour
    and are unaffected (already saved), but a fresh rerun of the same
    command will now place albedo on the shared grid instead.

    `row_state_interp` (default None) sets `ParamSpec.state_interp` on
    matching rows at construction time -- e.g. `{"albedo": "nearest"}`
    lets one row use piecewise-constant interpolation on the SAME shared
    grid another row (still `None`, falling back to whatever `state_interp`
    the solve itself is called with) interpolates linearly. See `StateSpec.
    interp_to`/`interp_weights`'s own docstrings for the resolution rule.

    `row_sub_bin_anomaly` (default None, 2026-08-28) sets `ParamSpec.
    sub_bin_anomaly` on matching rows at construction time -- a per-row
    dict `{"g_fn": callable, "g_positions": array (optional), "g_cov":
    array (optional)}`. `g_fn(eta_array) -> values` is a higher-resolution
    reference profile that rides ADDITIVELY on top of this row's own plain
    interpolant (see `StateSpec.interp_to`'s docstring for the exact
    formula); `g_positions`/`g_cov` are only needed to propagate `g`'s own
    uncertainty into the posterior via `gauss_newton_state`'s `g_cov`
    parameter, and are `None` by default (no uncertainty machinery at
    all, not "zero uncertainty").

    `uniform=True` evaluates every field at a single fixed position
    (`x_km=0.0`, matching `along_slit_scene.atmosphere_at(0.0)`'s own
    convention for "the" center atmosphere) instead of each bin's own
    `x_km`, so every bin gets the SAME prior -- the correct state-side
    counterpart to a truth scene whose composition genuinely does not vary
    along the slit (`--uniform` / `--barcode` on
    `gd_joint_block_whole_slit_sweep.py`).

    Found 2026-08-19: before this parameter existed, nothing in this
    function read `uniform` at all -- the sweep script's own `--uniform`
    flag only reached a `prior_atms`/`prior_co2_ppm_bins` pair that fed
    NOTHING (`state_spec_from_scene` was always called with no `uniform`
    argument), so every StateSpec-era `--uniform` run was silently
    regularized toward the real, non-uniform along-slit truth while
    rendering a genuinely flat detector image -- exactly the kind of
    prior/truth mismatch a `--uniform` run exists to rule out. Confirmed
    directly: `state_spec_from_scene(bc, free=('co2_ppm',))` at six
    different bin centres returned six different priors before this fix.

    `prior_anchor_density` (default None) controls the RESOLUTION at which
    `fields` gets sampled to build the prior, independent of `fields` itself
    -- a way to study sensitivity to prior quality directly. None (the
    default) reproduces today's exact mechanism: every bin's prior is `fn`
    evaluated at that bin's own position. Any float instead builds the prior
    from `round(prior_anchor_density * len(positions))` anchor points spread
    evenly across `positions`' own span, sampling `fields` only there and
    linearly interpolating onto the actual bin positions -- 1.0 lands on
    exactly `len(positions)` anchors and is special-cased to reuse
    `positions` itself, so it reproduces the default exactly ("just bin
    centers"); <1.0 is a coarser prior that smooths away sub-anchor-spacing
    structure (hot spots included, even if `fields` is the exact truth);
    >1.0 oversamples, giving the prior more along-slit structure than one
    value per bin would on its own. Ignored when `uniform=True` (that
    already collapses every bin to a single shared sample).
    """
    from . import along_slit_scene as als

    fields = als.STATE_FIELDS if fields is None else fields
    surface_fields = als.SURFACE_FIELDS if surface_fields is None else surface_fields
    bin_centers = np.atleast_1d(np.asarray(bin_centers, dtype=float))
    default_sigma = {"co2_ppm": 0.10, "ch4_ppb": 0.10, "co_ppb": 0.20,
                     "h2o_surface_vmr": 0.25, "p_surface_hpa": 0.02,
                     "albedo": 0.20}
    sigmas = {**default_sigma, **(sigmas or {})}
    kinds = kinds or {}

    def _corr_for(name, cl):
        if cl is None:
            return DEFAULT_CORR_LENGTH_ETA.get(name, 0.05)
        if isinstance(cl, dict):
            return cl.get(name, DEFAULT_CORR_LENGTH_ETA.get(name, 0.05))
        return cl

    def _row(name, fn, positions, target):
        def _eval(xk):
            return fn(xk, band_label) if target == "surface" else fn(xk)

        if uniform:
            # every bin gets x_km=0.0 -- prior_anchor_density is moot, this
            # already collapses every bin to one shared sample
            prior = _eval(np.zeros_like(positions))
        elif prior_anchor_density is None:
            # today's exact mechanism: fn evaluated at each bin's own position
            prior = _eval(positions * als.SLIT_HALF_KM)
        else:
            n_anchor = max(2, int(round(prior_anchor_density * len(positions))))
            # density landing on exactly len(positions) anchors reuses
            # `positions` itself (not a uniform re-grid of the same span),
            # so it reproduces the exact-mechanism prior bit-for-bit
            anchor_pos = (positions if n_anchor == len(positions)
                         else np.linspace(positions.min(), positions.max(), n_anchor))
            anchor_vals = _eval(anchor_pos * als.SLIT_HALF_KM)
            prior = np.interp(positions, anchor_pos, anchor_vals)

        row_kind = (row_state_interp or {}).get(name)
        anomaly_opts = (row_sub_bin_anomaly or {}).get(name)
        anomaly = None
        if anomaly_opts is not None:
            opts = dict(anomaly_opts)
            g_cov = opts.pop("g_cov", None)
            g_fn = opts["g_fn"]
            g_positions = opts.get("g_positions")
            g_at_centers = np.asarray(g_fn(positions), dtype=float)
            Wg_at_centers = None
            if g_positions is not None:
                g_positions = np.asarray(g_positions, dtype=float)
                g_throwaway = ParamSpec(name="_gpos", positions=g_positions,
                                        prior=g_positions, sigma=1.0, free=True, kind="scale")
                Wg_at_centers = StateSpec([g_throwaway]).interp_weights(
                    positions, "_gpos", row_kind or "linear")
            anomaly = SubBinAnomaly(g_fn=g_fn, g_at_centers=g_at_centers,
                                    g_positions=g_positions, g_cov=g_cov,
                                    Wg_at_centers=Wg_at_centers)

        return ParamSpec(name=name, positions=positions,
                         prior=np.asarray(prior, dtype=float),
                         sigma=float(sigmas.get(name, 0.10)),
                         corr_length=float(_corr_for(name, corr_length)),
                         free=(name in free), kind=kinds.get(name, "scale"),
                         prior_form=prior_form, gamma=float(gamma), target=target,
                         state_interp=row_kind, sub_bin_anomaly=anomaly)

    rows = [_row(name, fn, bin_centers, "atmosphere") for name, fn in fields.items()]
    if band_label is not None:
        # Shared grid by default (2026-08-25) -- surface rows use the SAME
        # bin_centers atmosphere rows use, unless surface_positions is
        # explicitly given. See this function's own docstring for why this
        # is a deliberate behaviour change, not an oversight.
        pos = (bin_centers if surface_positions is None
               else np.atleast_1d(np.asarray(surface_positions, dtype=float)))
        rows += [_row(name, fn, pos, "surface")
                 for name, fn in surface_fields.items()]
    return StateSpec(rows)


def gauss_newton_state(forward, y_true, spec: StateSpec, Sy_inv_diag,
                       step: float = 1e-3, max_iter: int = 15, tol: float = 1e-5,
                       label: str = "", verbose: bool = True, jacobian_fn=None,
                       return_cov: bool = False, return_avk: bool = False,
                       g_cov: dict | None = None):
    """Regularized Gauss-Newton over whatever :class:`StateSpec` says is free.

    The generic counterpart of `gd_joint_block_retrieve.gauss_newton_
    regularized`, which hardwired a single flat CO2 vector and a
    `gamma*(L^T L) + I/sigma_abs**2` prior. Here the free parameters, their
    ordering, and the prior precision all come from `spec`, so freeing
    surface pressure or freezing CO2 is a flag rather than a code change.

    Rodgers form, unchanged:
        dx = (K^T Sy^-1 K + Sa^-1)^-1 (K^T Sy^-1 resid - Sa^-1 (x - x_a))

    `Sa_inv` is `spec.Sa_inv()` -- block diagonal, one exponential-correlation
    block per free row, each with its own sigma and correlation length in
    eta. JOINT_BLOCK_MIGRATION_PLAN.md Sec.4.0 showed the old two-constant
    form is an implicit, awkwardly parameterized special case of exactly
    this: there, the decorrelation length and the marginal prior std were
    both set jointly by the ratio (1/sigma_abs^2)/gamma and could only be
    recovered by inverting the matrix. Here they are independent, named, and
    in physical units.

    One finite-difference step serves every parameter because all rows use
    `kind="scale"`: each element is a multiplier on its own prior, so they
    are all O(1) regardless of whether the underlying quantity is 416 ppm or
    1013 hPa. A row switched to `kind="absolute"` would need its own step,
    which is why this asserts rather than silently mis-scaling.

    Passing `jacobian_fn` switches to ANALYTIC derivatives: a callable
    ``jacobian_fn(x) -> (y, K)``, normally
    `geocarb_gert.jacobians.linearize` bound to this window. It returns the
    forward value alongside the Jacobian, so an iteration costs one
    evaluation rather than ``n_free + 1``, and `forward` is then unused. The
    `kind="scale"` restriction is lifted in that mode, because analytic
    columns carry their own units and there is no shared step to mis-scale.

    ``return_cov=True`` additionally returns the posterior covariance
    ``S_ret = A^-1`` (Rodgers ``(K^T Sy^-1 K + Sa^-1)^-1``), evaluated once
    at ``A``'s own final-iteration value -- no extra forward/Jacobian
    evaluation, since ``A`` is already built for that last `np.linalg.
    solve`. This is the FULL covariance over every free element in `spec`'s
    own packed order (`spec.slices()`), in "scale" units (same units `x`
    itself is in) -- includes off-diagonal covariance between every pair of
    bins, both within one row (neighboring bins) and, when more than one
    row is free at once (e.g. co2_ppm + p_surface_hpa), across rows. Use
    `StateSpec.cov_for`/`StateSpec.project_cov` to slice out one row's own
    block, convert to physical units, and (optionally) project onto
    arbitrary eta positions through the same interpolation the point
    estimate itself uses.

    ``return_avk=True`` additionally returns the averaging kernel ``AVK =
    Gain @ K``, ``Gain = S_ret @ K^T @ Sy_inv`` -- both already available
    at the final iteration (``KtSyinv`` is reused, ``S_ret`` computed once
    regardless of ``return_cov``), so this is one extra matmul, not a
    second solve. ``AVK`` is ``(n_free, n_free)`` in the SAME packed order
    as the covariance (`spec.slices()`); row ``k`` is the linear
    combination of the TRUE state (at every other free element's own
    position) the retrieved element ``k`` actually reflects -- ``x_hat =
    AVK @ x_true + (I - AVK) @ x_a`` (Rodgers). A well-constrained element's
    row is close to a delta function at its own position (the retrieved
    value behaves like a genuine point estimate of truth there); a poorly
    -constrained one spreads across neighbors and the prior. ``trace(AVK)``
    is the degrees-of-freedom-for-signal for this whole solve -- the
    rigorous "how many independent pieces of information" answer, cheap to
    derive from the returned matrix rather than something this function
    computes itself, since a caller may want it per-row, for a row subset,
    or as the one whole-solve scalar.

    ``g_cov`` (paired with `jacobian_fn` -- only meaningful in analytic
    mode, since it consumes the `K_g` dict `jac.linearize` now returns as
    a third tuple element) propagates a `sub_bin_anomaly` reference
    function's OWN uncertainty into the retrieved posterior, Rodgers-style:
    ``S_g_total = sum over rows( Gain @ K_g[name] @ g_cov[name] @
    K_g[name].T @ Gain.T )``, ``Gain = S_ret @ KtSyinv`` (the same `Gain`
    `AVK` is already built from, reused rather than rebuilt). This is a
    SEPARATE output, never folded into `S_ret` -- `S_ret` stays exactly the
    noise-driven posterior covariance it always was, so a caller can see
    the two contributions apart. Default `None`: every row's `K_g` is
    empty (`jac.linearize` returns an empty dict for any row that never
    declared `sub_bin_anomaly.g_cov`), and passing `g_cov=None` (or an
    empty dict) skips the whole computation outright -- no extra matmuls,
    no shape/None-handling risk to `S_ret`/`AVK`, for every caller that
    never uses this feature. `g_cov[name]` must line up with `K_g[name]`'s
    own columns (`g_positions` for that row); the two are only checked
    for shape agreement, not for what `g_positions` values they came from.

    Return shape depends on which of `return_cov`/`return_avk`/`g_cov` are
    set: ``x`` (neither cov flag), ``(x, S_ret)`` (cov only, unchanged from
    before this parameter existed), ``(x, AVK)`` (avk only), ``(x, S_ret,
    AVK)`` (both) -- and, whenever `g_cov` is given and nonempty, `S_g_
    total` is appended as the LAST element of whichever of those tuples
    would otherwise be returned (e.g. ``(x, S_ret, S_g_total)`` with only
    `return_cov`, ``(x, AVK, S_g_total)`` with only `return_avk`, ``(x,
    S_ret, AVK, S_g_total)`` with both). `g_cov` with neither `return_cov`
    nor `return_avk` still returns bare ``x`` -- `S_g_total` needs `S_ret`
    (via `Gain`), so requesting it without `return_cov`/`return_avk` is a
    no-op, not an implicit `return_cov=True`.
    """
    if jacobian_fn is None:
        for p in spec.free_params:
            if p.kind != "scale":
                raise NotImplementedError(
                    f"{p.name}: gauss_newton_state uses one shared finite-difference "
                    f"step, which assumes kind='scale' (elements are O(1) multipliers). "
                    f"kind='absolute' needs a per-row step -- pass jacobian_fn instead.")

    x = spec.x0()
    x_a = x.copy()
    n = x.size
    if n == 0:
        raise ValueError("no free parameters -- every row is frozen")
    Sa_inv = spec.Sa_inv()
    Sy_inv_diag = np.asarray(Sy_inv_diag, dtype=float)

    K_g_dict: dict = {}
    for it in range(max_iter):
        if jacobian_fn is not None:
            y0, K, K_g_dict = jacobian_fn(x)
            resid = y_true - y0
        else:
            y0 = forward(x)
            resid = y_true - y0
            K = np.empty((y0.size, n))
            for k in range(n):
                xp = x.copy()
                xp[k] += step
                K[:, k] = (forward(xp) - y0) / step
        KtSyinv = K.T * Sy_inv_diag[None, :]
        A = KtSyinv @ K + Sa_inv
        b = KtSyinv @ resid - Sa_inv @ (x - x_a)
        dx = np.linalg.solve(A, b)
        x = x + dx
        if verbose:
            rms = float(np.sqrt(np.mean(resid ** 2)))
            print(f"  [{label}] iter {it}: |dx|={np.linalg.norm(dx):.3e} "
                  f"rms_resid={rms:.4g}", flush=True)
        if np.linalg.norm(dx) < tol:
            break
    if not (return_cov or return_avk):
        return x
    S_ret = np.linalg.inv(A)

    def _s_g_total():
        # Strict no-op: skipped outright (not computed-and-zeroed) unless
        # g_cov is given and nonempty, per the plan's own no-op guarantee.
        if not g_cov:
            return None
        Gain = S_ret @ KtSyinv
        total = np.zeros((n, n))
        for name, cov in g_cov.items():
            Kg = K_g_dict.get(name)
            if Kg is None or Kg.size == 0:
                continue
            GKg = Gain @ Kg
            total = total + GKg @ cov @ GKg.T
        return total

    if not return_avk:
        s_g_total = _s_g_total()
        return (x, S_ret) if s_g_total is None else (x, S_ret, s_g_total)
    avk = (S_ret @ KtSyinv) @ K
    s_g_total = _s_g_total()
    if return_cov:
        return (x, S_ret, avk) if s_g_total is None else (x, S_ret, avk, s_g_total)
    return (x, avk) if s_g_total is None else (x, avk, s_g_total)


def build_forward_state(fpa, rows_win, scene_etas, spec: StateSpec, spectrum,
                        wn_hires, ils, pad: int = 4, state_interp: str = "linear"):
    """``forward(x) -> raveled sub-image`` for a :class:`StateSpec`, on an
    arbitrary set of scene positions.

    Unifies the joint block's "coarse" and "hi-res" forward models, which
    differ only in WHERE the scene is evaluated:

      coarse   ``scene_etas = bin_centers``  -- the state's own positions, so
               :meth:`StateSpec.interp_to` is the identity and no
               interpolation happens at all. (This is why ``state_interp``
               is meaningless for the coarse solve: there is nothing between
               the state and the scene to interpolate.)
      hi-res   ``scene_etas = anchor_etas``  -- a finer grid, so every row is
               interpolated from its own positions onto the anchors, by
               whatever `state_interp` kind names. State-space interpolation
               followed by a fresh RT run per anchor, exactly what
               ``nearest_bin_scene``'s docstring prescribes; no spectrum is
               ever blended with another.

    Pixels are then assigned to scene positions by nearest
    (``nearest_bin_scene``), unchanged. That hard assignment is the source of
    the first-order ``~(1/4)|f'|h`` stepping error the residual is dominated
    by, and it is a property of the SAMPLING, not of the state -- which is
    why adding free parameters barely dents it while halving ``h`` does.

    Each scene position's spectrum is cached on that position's full
    parameter vector, so an iteration re-runs RT only where the state
    actually moved.

    ``state_interp`` names a :func:`scipy.interpolate.interp1d` ``kind`` --
    ``"linear"`` (default) or ``"nearest"`` (piecewise-constant: each anchor
    takes its single nearest bin's own value). EVERY row, free or frozen,
    goes through the same interpolation, always -- there is deliberately no
    bypass to exact truth here. If a row needs to be as-good-as-truth, that
    is a PRIOR-construction decision (``fields=``, ``prior_anchor_density``,
    ``native_res=True`` in ``state_spec_from_scene``/the diagnostic scripts),
    never something the renderer silently substitutes. (Found 2026-08-19,
    user: an earlier version of this function had a ``state_interp=False``
    branch that overrode frozen rows to ``als.STATE_FIELDS`` truth directly,
    ignoring whatever ``fields=`` built the prior with -- retired outright,
    not renamed, because "the way the FPA is rendered should be independent
    of the prior".)

    Parameters
    ----------
    spectrum : callable
        ``spectrum(params: dict) -> hi-res radiance``, where ``params`` is
        keyed exactly as :data:`geocarb_gert.along_slit_scene.STATE_FIELDS`.

        When the spec carries `surface`-target rows, it is called instead as
        ``spectrum(params, surface: dict)`` with the second dict keyed as
        :data:`geocarb_gert.along_slit_scene.SURFACE_FIELDS` -- i.e. the
        second argument appears only when there is something to put in it, so
        every pre-2026-08-18 one-argument `spectrum` keeps working untouched
        rather than needing a signature change it has no use for.
    """
    from . import gd_render
    from .focalplane import nearest_bin_scene

    scene_etas = np.asarray(scene_etas, dtype=float)
    order = np.argsort(scene_etas)
    scene_etas = scene_etas[order]
    n_scene = scene_etas.size
    # atmosphere rows only: `spectrum`'s first dict builds an
    # AtmosphericProfile, so a `surface` row there would be an unexpected
    # keyword. `instrument` rows are consumed at the ILS centring, not here.
    names = [p.name for p in spec.rows_for("atmosphere")]
    surf_names = [p.name for p in spec.rows_for("surface")]
    # the cache key must span BOTH, or a move in albedo alone would reuse a
    # stale spectrum computed at the previous albedo
    cache_key = np.full((n_scene, len(names) + len(surf_names)), np.nan)
    cache_S = [None] * n_scene

    def forward(x):
        vals = spec.interp_to(scene_etas, x, state_interp=state_interp)
        key = np.column_stack([vals[n] for n in names + surf_names])
        for g in range(n_scene):
            if cache_S[g] is None or not np.array_equal(key[g], cache_key[g]):
                atm_p = {n: float(vals[n][g]) for n in names}
                cache_S[g] = (spectrum(atm_p, {n: float(vals[n][g]) for n in surf_names})
                              if surf_names else spectrum(atm_p))
                cache_key[g] = key[g]
        radiance = nearest_bin_scene(scene_etas, cache_S)
        return gd_render.predict_neighborhood(fpa, rows_win, wn_hires, radiance,
                                              ils, pad=pad).ravel()

    return forward
