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


@dataclass
class ParamSpec:
    """One state-vector element: where it lives, its prior, and whether it
    is currently being retrieved."""

    name: str
    positions: np.ndarray          # eta bin centres for this parameter
    prior: np.ndarray              # prior value at each position
    sigma: float                   # prior 1-sigma (in the parameter's own units)
    corr_length: float = 0.05      # correlation length in eta
    free: bool = True              # False -> frozen at `prior`, contributes no elements
    kind: str = "scale"            # "scale" (multiplies prior) or "absolute"

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
        """Exponential-correlation prior covariance for this row alone."""
        d = np.abs(self.positions[:, None] - self.positions[None, :])
        return self.sigma ** 2 * np.exp(-d / max(self.corr_length, 1e-12))


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
        blocks = [np.linalg.inv(p.Sa_block()) for p in self.free_params]
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
    def interp_to(self, etas, x) -> dict:
        """`{name: value}` linearly interpolated onto `etas`, every row
        treated identically -- the state-space interpolation a forward model
        needs to build one atmosphere per anchor."""
        etas = np.asarray(etas, dtype=float)
        vals = self.unpack(x)
        out = {}
        for p in self.params:
            v = vals[p.name]
            out[p.name] = (np.full(etas.shape, v[0]) if p.n == 1
                           else np.interp(etas, p.positions, v))
        return out

    def __repr__(self) -> str:
        rows = ", ".join(f"{p.name}[{p.n}]{'' if p.free else ' FROZEN'}" for p in self.params)
        return f"StateSpec({rows}; n_free={self.n_free})"


def state_spec_from_scene(bin_centers, fields=None, free=("co2_ppm",),
                          sigmas=None, corr_length=0.05, kinds=None) -> StateSpec:
    """Build a :class:`StateSpec` whose priors are the truth scene's own
    values at `bin_centers` -- the joint block's existing "local-truth
    nuisance idealization", but now with every quantity present as a real,
    unfreezable row instead of being pinned invisibly inside the forward
    model.

    `free` names the rows that start free; everything else starts frozen, so
    the default `free=("co2_ppm",)` reproduces today's CO2-only retrieval
    exactly while making the other four one flag away from being retrieved.
    """
    from . import along_slit_scene as als

    fields = als.STATE_FIELDS if fields is None else fields
    bin_centers = np.atleast_1d(np.asarray(bin_centers, dtype=float))
    x_km = bin_centers * als.SLIT_HALF_KM
    default_sigma = {"co2_ppm": 0.10, "ch4_ppb": 0.10, "co_ppb": 0.20,
                     "h2o_surface_vmr": 0.25, "p_surface_hpa": 0.02}
    sigmas = {**default_sigma, **(sigmas or {})}
    kinds = kinds or {}
    return StateSpec([
        ParamSpec(name=name, positions=bin_centers, prior=np.asarray(fn(x_km), dtype=float),
                  sigma=float(sigmas.get(name, 0.10)), corr_length=float(corr_length),
                  free=(name in free), kind=kinds.get(name, "scale"))
        for name, fn in fields.items()
    ])
