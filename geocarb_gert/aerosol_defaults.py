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


#: Reference slot: `amplitude_aerosol` is the extinction-per-Pa at this slot's
#: wavelength (the CO2 1.6 um band -- what the project's original single
#: band-1 convention meant). Other bands scale tau by
#: `qext_norm[slot]/qext_norm[_BAND_INDEX]` (gert only normalises qext
#: WITHIN one run, so a one-window Instrument always sees ratio 1).
_REF_SLOT = _BAND_INDEX


def band_slot_for_wavelength_um(wl_um: float) -> int:
    """gert registry slot for a band centred at `wl_um`: 0 = O2-A (<1 um),
    1 = CO2 weak/strong (1.6-2.1 um, also the reference), 2 = CH4 (1.63-1.7 um
    is folded into 1 unless it is the CH4 window, so CH4 must be requested
    explicitly through `slot=`)."""
    return 0 if wl_um < 1.0 else 1


@lru_cache(maxsize=None)
def aerosol_band_props(aerosol_type: str, slot: int) -> tuple[float, float, float]:
    """`(ssa, g, tau_scale)` for registry `slot`; `tau_scale` multiplies the
    reference-wavelength column tau."""
    sc = get_aerosol_scalars(aerosol_type)
    q = sc["qext_norm"]
    return (float(sc["ssa"][slot]), float(sc["g"][slot]),
            float(q[slot]) / float(q[_REF_SLOT]))


import os

SMOKE_MIE = "smoke_mie"
_MIE_CENTRES_UM = None


def resolve_aerosol_type(aerosol_type=None) -> str:
    """The aerosol type in effect: an explicit argument, else the GEOCARB_AEROSOL_TYPE environment variable (set by
    the drivers' --aerosol-type, and inherited by worker processes), else the legacy registry "smoke"."""
    return aerosol_type or os.environ.get("GEOCARB_AEROSOL_TYPE", "smoke")


def mie_band_props_for_wavelength(wl_um: float) -> tuple[float, float, float]:
    """`(ssa, g, tau_scale)` of the Mie smoke model (aerosol_mie.py) for the FPA whose centre is nearest `wl_um`;
    `tau_scale` = qext(band)/qext(reference band, 1.6 um)."""
    from .aerosol_mie import BAND_CENTRES_UM, smoke_band_properties
    fpa = min(BAND_CENTRES_UM, key=lambda f: abs(BAND_CENTRES_UM[f] - wl_um))
    ssa, g, q = smoke_band_properties()[fpa]
    return float(ssa), float(g), float(q)
