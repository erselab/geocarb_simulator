"""Aerosol microphysics defaults, reused from `gert.aerosol_properties`
rather than re-invented (that registry -- `dust`/`sulfate`/`smoke`/
`sea_salt`/`cloud_water` -- already exists in `gert` and is not duplicated
here).

`gert.aerosol_properties.get_aerosol_props` assumes exactly 3 instrument
bands (GERT's own O2-A/CO2-weak/CH4 convention), which doesn't match this
project's single-window-per-FPA `Instrument` objects -- so this module uses
`get_aerosol_scalars` (no instrument needed) and picks band index 1 (the
"CO2 weak"-like slot), matching the fixed defaults chosen when aerosol was
first added (`along_slit_scene.AEROSOL_SSA = 0.87`, `AEROSOL_G = 0.50` --
exactly `smoke`'s band-1 values). `aerosol_type="smoke"` therefore
reproduces the existing behavior exactly; other registered types are
available for future multi-band work once each FPA's own window is threaded
through separately.
"""
from __future__ import annotations

from functools import lru_cache

from gert.aerosol_properties import get_aerosol_scalars

#: Which of the registry's 3 per-band scalar slots this single-band project
#: uses. See module docstring.
_BAND_INDEX = 1


@lru_cache(maxsize=None)
def aerosol_scalars_for(aerosol_type: str) -> tuple[float, float, float]:
    """`(ssa, g, qext_norm)` for `aerosol_type`, at this project's fixed
    band-1 convention."""
    scalars = get_aerosol_scalars(aerosol_type)
    return (float(scalars["ssa"][_BAND_INDEX]),
            float(scalars["g"][_BAND_INDEX]),
            float(scalars["qext_norm"][_BAND_INDEX]))
