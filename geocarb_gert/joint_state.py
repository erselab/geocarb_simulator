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

import numpy as np


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
DEFAULT_CORR_LENGTH_ETA = {
    "co2_ppm": 0.0071,            # ~10 km -- the CO2/CO hot-spot scale; longer would
                                  # wash out the point sources the study is about
    "ch4_ppb": 0.0714,            # ~100 km -- its own broad plume width
    "co_ppb": 0.0393,             # ~55 km  -- its own broad plume width
    "h2o_surface_vmr": 0.357,     # ~500 km -- the smooth climatological tanh gradient
    "p_surface_hpa": 0.100,       # ~140 km -- the topographic depression width
    # Surface albedo is the one row whose structure is SHORTER than the gas
    # features, not longer: `along_slit_scene.ALBEDO_CORR_KM` = 30 km of
    # within-patch variability, on top of land-cover boundaries only
    # ALBEDO_EDGE_KM ~ 3 km (about one detector row) wide. This is Sec.4.1's
    # point that albedo wants denser bins and a shorter correlation length
    # than a well-mixed gas -- see `albedo_positions_for` below.
    "albedo": 0.0214,             # ~30 km -- ALBEDO_CORR_KM / SLIT_HALF_KM
}


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

        Defaulting to ``tikhonov`` keeps continuity with every result
        computed before 2026-08-17; ``exponential`` is opt-in.
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

    def interp_to(self, etas, x, targets=ETA_TARGETS) -> dict:
        """`{name: value}` linearly interpolated onto `etas`, for the
        eta-positioned rows -- the state-space interpolation a forward model
        needs to build one atmosphere per anchor.

        `instrument` rows are EXCLUDED by default and must be: their
        `positions` are not along-slit coordinates, so interpolating them
        against `etas` would be meaningless arithmetic on mismatched axes
        rather than a harmless no-op. Use :meth:`values_for` to read them.
        """
        etas = np.asarray(etas, dtype=float)
        vals = self.unpack(x)
        out = {}
        for p in self.params:
            if p.target not in targets:
                continue
            v = vals[p.name]
            out[p.name] = (np.full(etas.shape, v[0]) if p.n == 1
                           else np.interp(etas, p.positions, v))
        return out

    def values_for(self, x, *targets) -> dict:
        """`{name: values at that row's own positions}`, uninterpolated.

        The read path for `instrument` rows, whose positions are not eta.
        """
        vals = self.unpack(x)
        return {p.name: vals[p.name] for p in self.params if p.target in targets}

    def interp_weights(self, etas, name) -> np.ndarray:
        """`(len(etas), n)` matrix `W` with `W[g, k] = d(param at etas[g]) /
        d(this row's value at position k)`.

        The linear-interpolation operator written out as a matrix, built by
        interpolating the basis vectors rather than by rederiving np.interp's
        weight formula -- so it cannot drift from what :meth:`interp_to`
        actually does, including at and beyond the end points, where np.interp
        clamps rather than extrapolating.

        This is the whole chain rule from a row's own elements to the anchor
        values a forward model sees. Combined with `d(value)/dx = prior` for
        `kind="scale"`, it gives `d(param at anchor g)/dx_k` exactly.
        """
        p = self[name]
        etas = np.asarray(etas, dtype=float)
        if p.n == 1:
            return np.ones((etas.size, 1))
        W = np.empty((etas.size, p.n))
        for k in range(p.n):
            e_k = np.zeros(p.n)
            e_k[k] = 1.0
            W[:, k] = np.interp(etas, p.positions, e_k)
        return W

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


def state_spec_from_scene(bin_centers, fields=None, free=("co2_ppm",),
                          sigmas=None, corr_length=None, kinds=None,
                          prior_form="exponential", gamma=3.0,
                          band_label=None, surface_positions=None,
                          surface_density: int = 3, uniform: bool = False) -> StateSpec:
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
    `surface`-target rows from `along_slit_scene.SURFACE_FIELDS`, on their
    own denser grid (`surface_positions`, or `albedo_positions_for(
    bin_centers, surface_density)`). Omitting it reproduces the pre-2026-08-18
    atmosphere-only state exactly, so every existing caller is unaffected.

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
    """
    from . import along_slit_scene as als

    fields = als.STATE_FIELDS if fields is None else fields
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
        # under uniform=True every bin gets x_km=0.0 -- see the `uniform`
        # branch above; `positions` (eta) still vary, only the physical
        # position fed to the truth function is pinned
        xk = (np.zeros_like(positions) if uniform else positions * als.SLIT_HALF_KM)
        prior = (fn(xk, band_label) if target == "surface" else fn(xk))
        return ParamSpec(name=name, positions=positions,
                         prior=np.asarray(prior, dtype=float),
                         sigma=float(sigmas.get(name, 0.10)),
                         corr_length=float(_corr_for(name, corr_length)),
                         free=(name in free), kind=kinds.get(name, "scale"),
                         prior_form=prior_form, gamma=float(gamma), target=target)

    rows = [_row(name, fn, bin_centers, "atmosphere") for name, fn in fields.items()]
    if band_label is not None:
        pos = (albedo_positions_for(bin_centers, surface_density)
               if surface_positions is None
               else np.atleast_1d(np.asarray(surface_positions, dtype=float)))
        rows += [_row(name, fn, pos, "surface")
                 for name, fn in als.SURFACE_FIELDS.items()]
    return StateSpec(rows)


def gauss_newton_state(forward, y_true, spec: StateSpec, Sy_inv_diag,
                       step: float = 1e-3, max_iter: int = 15, tol: float = 1e-5,
                       label: str = "", verbose: bool = True, jacobian_fn=None):
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

    for it in range(max_iter):
        if jacobian_fn is not None:
            y0, K = jacobian_fn(x)
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
    return x


def build_forward_state(fpa, rows_win, scene_etas, spec: StateSpec, spectrum,
                        wn_hires, ils, pad: int = 4, state_interp: bool = True):
    """``forward(x) -> raveled sub-image`` for a :class:`StateSpec`, on an
    arbitrary set of scene positions.

    Unifies the joint block's "coarse" and "hi-res" forward models, which
    differ only in WHERE the scene is evaluated:

      coarse   ``scene_etas = bin_centers``  -- the state's own positions, so
               :meth:`StateSpec.interp_to` is the identity and no
               interpolation happens at all. (This is why a ``--state-interp``
               switch is meaningless for the coarse solve: there is nothing
               between the state and the scene to interpolate.)
      hi-res   ``scene_etas = anchor_etas``  -- a finer grid, so every row is
               linearly interpolated from its own positions onto the anchors.
               State-space interpolation followed by a fresh RT run per
               anchor, exactly what ``nearest_bin_scene``'s docstring
               prescribes; no spectrum is ever blended with another.

    Pixels are then assigned to scene positions by nearest
    (``nearest_bin_scene``), unchanged. That hard assignment is the source of
    the first-order ``~(1/4)|f'|h`` stepping error the residual is dominated
    by, and it is a property of the SAMPLING, not of the state -- which is
    why adding free parameters barely dents it while halving ``h`` does.

    Each scene position's spectrum is cached on that position's full
    parameter vector, so an iteration re-runs RT only where the state
    actually moved.

    ``state_interp=False`` reproduces the pre-StateSpec hi-res behaviour:
    FROZEN rows are evaluated at their exact truth at each scene position
    (``als.state_at(scene_etas)``) instead of being interpolated from the bin
    grid, while free rows are still interpolated. That distinction is real
    and measurable -- on a smooth, low-keystone window, exact-at-anchor beats
    interpolated-from-bins by ~1.5x in residual RMS, because interpolation
    can only add error to a quantity that was already exactly known. With
    ``state_interp=True`` every row goes through the identical path, which is
    the honest configuration when the nuisance state is meant to be treated
    like any other (and the only sane one once those rows are actually
    free, since a free row has no "truth" to be exact about).

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

    exact = None
    if not state_interp:
        from . import along_slit_scene as _als
        frozen = [p.name for p in spec.params if not p.free]
        exact = {n: np.asarray(_als.STATE_FIELDS[n](scene_etas * _als.SLIT_HALF_KM),
                               dtype=float) for n in frozen if n in _als.STATE_FIELDS}

    def forward(x):
        vals = spec.interp_to(scene_etas, x)
        if exact:
            vals.update(exact)          # frozen rows at exact truth, not interpolated
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
