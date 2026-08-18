"""Pure-sigma vertical levels for the GeoCarb simulator.

A deliberate, narrow departure from `gert.levels.gert_levels`, made so that
surface pressure has an EXACT analytic Jacobian. Everything else about the
level set is unchanged -- same count, same standard-atmosphere pressures,
same terrain-following idea.

The distinction
---------------
`gert` uses the Laprise (1992) mass coordinate, i.e. sigma generalised to a
nonzero model top::

    gert:  p(eta) = eta * (p_sfc - p_top) + p_top      p_top = 100 Pa
    here:  p(sigma) = sigma * p_sfc                    (pure sigma)

Pure sigma is the ``p_top = 0`` special case, and the difference is exactly
one constant offset -- but that offset is what decides whether a surface-
pressure Jacobian can be assembled exactly from what `gert` returns.

Why it matters (this is the whole reason this module exists)
------------------------------------------------------------
`gert`'s analytic pressure Jacobian, ``dtau_mol_dpscale_lay_hires``, is the
derivative with respect to UNIFORM SCALING of whatever ``p_levels`` it was
handed: ``dp_l/ds = p_l``. It is coordinate-agnostic -- it neither knows nor
cares which constructor built the levels.

So the only question is whether this scene's own ``dp_l/dp_sfc`` is parallel
to ``p_l``:

* Under Laprise eta, ``dp_l/dp_sfc = eta_l``, and ``eta_l / (p_l/p_sfc)``
  ranges from 0 at the model top to 1 at the surface -- NOT parallel. The
  layer-thickness and pressure-broadening contributions are then weighted
  differently by the true transform than by ``dtau_dpscale``, and they
  cannot be separated from the combined array. Measured cost: a floor of
  ~4e-4 relative in the p_surface Jacobian, against roundoff for every other
  state row.
* Under pure sigma, ``dp_l/dp_sfc = sigma_l = p_l/p_sfc`` -- exactly
  parallel, ratio spread 3.3e-16 (machine epsilon). Both terms pick up the
  same ``1/p_sfc``, so one constant converts `gert`'s array into the
  derivative this scene needs, with no split required and no approximation.

A second, independent benefit: under pure sigma, scaling every level by
``s`` IS the same coordinate at surface pressure ``p_sfc*s`` (verified to
1.5e-11 Pa, against 0.1 Pa for Laprise). That makes `gert`'s own
``StateVector`` ``p_scale`` element the exactly-correct transform for this
scene too -- relevant to the band-stress-test pipelines in
`scripts/gd_test.py`, which float ``p_scale`` and until now carried the same
~1e-3 mismatch.

What changes numerically
------------------------
:data:`SIGMA` is derived from `gert`'s OWN standard-atmosphere level
pressures (``GERT_P_LEVELS``), just divided by standard surface pressure
instead of Laprise-inverted. Consequences, measured:

* At standard surface pressure the levels are IDENTICAL to `gert`'s
  (7.3e-12 Pa).
* Away from it they differ by at most 26 Pa, at the scene's lowest surface
  pressure (750 hPa, the depression bottom) -- and that maximum sits at the
  model top, where there is no absorption. At 500 hPa the shift is 3.6e-4
  relative; at the surface it is exactly zero by construction.

Note ``SIGMA[0]`` is 9.87e-4, NOT zero. A literal ``p_top = 0`` would put
the top level at p = 0, where `model_sampler.pressure_to_alt_std_atm`
returns ``inf`` and every downstream T/H2O value becomes NaN. Taking sigma
from the anchor pressures keeps the top finite (z ~ 50 km at the lowest
surface pressure this scene reaches) while staying exactly proportional --
which is the property that buys the exact Jacobian, and it is a property of
the top moving WITH ``p_sfc``, not of it being small.
"""
from __future__ import annotations

import numpy as np

from gert.levels import GERT_P_LEVELS, GERT_P_SFC_STD

#: Pure-sigma coordinate, taken from `gert`'s own standard-atmosphere level
#: pressures so the two agree exactly at standard surface pressure.
SIGMA: np.ndarray = np.asarray(GERT_P_LEVELS, dtype=float) / float(GERT_P_SFC_STD)

#: Standard surface pressure [Pa], re-exported so callers need not decide
#: whether to import it from here or from `gert`.
P_SFC_STD: float = float(GERT_P_SFC_STD)


def sigma_levels(p_sfc: float) -> np.ndarray:
    """Pressure levels [Pa], TOA -> surface, for surface pressure `p_sfc` [Pa].

    Drop-in replacement for `gert.levels.gert_levels` with the same shape,
    ordering and units. See the module docstring for why this scene uses
    pure sigma rather than `gert`'s Laprise eta.
    """
    return SIGMA * float(p_sfc)


def dp_levels_dp_surface() -> np.ndarray:
    """``d(p_level)/d(p_sfc)`` -- which is just :data:`SIGMA`, exactly.

    Spelled out as a function because it is the entire content of the
    exactness argument in `geocarb_gert.jacobians.p_surface_dI_dparam`: the
    derivative is proportional to ``p_l`` with the single constant
    ``1/p_sfc``, which is what lets `gert`'s uniform-scaling Jacobian be
    converted without separating its terms.
    """
    return SIGMA.copy()
