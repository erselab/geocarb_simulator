"""Aerosol type resolution/validation (2026-09-28: rewritten to remove the legacy gert-registry two-slot
scheme -- user: "I don't want to use that setup any longer. I want a list of defined types with band-specific
parameters that are defined in literature. If a type is called that isn't in the list, the code should error,
not fall back to a default"). `aerosol_mie.AEROSOL_TYPES` (literature-sourced from O'Dell et al. 2018's ACOS
Fig. 3 -- see that module's own docstring for provenance) is now the ONLY source of per-band aerosol optical
properties in this project: every FPA gets genuinely distinct (ssa, g, qext) values, not two shared slots.

There is no default aerosol type any more. `resolve_aerosol_type` raises if neither an explicit type nor
`GEOCARB_AEROSOL_TYPE` names one, and `validate_aerosol_type` raises for any name not in
`aerosol_mie.AEROSOL_TYPES` -- both hard errors, never a silent fallback.
"""
from __future__ import annotations

import os

from . import aerosol_mie

#: The only valid aerosol type names -- see aerosol_mie.AEROSOL_TYPES for the literature source of each.
REGISTRY_TYPES = tuple(aerosol_mie.AEROSOL_TYPES)


def resolve_aerosol_type(aerosol_type: str | None = None) -> str:
    """The aerosol type in effect: an explicit argument, else the `GEOCARB_AEROSOL_TYPE` environment variable
    (set by the drivers' `--aerosol-type`, inherited by worker processes). Raises `ValueError` if neither is
    given, or if the named type isn't one of `REGISTRY_TYPES` -- no default, ever (2026-09-28)."""
    t = aerosol_type or os.environ.get("GEOCARB_AEROSOL_TYPE")
    if t is None:
        raise ValueError(
            f"no aerosol type given (pass one explicitly, or set GEOCARB_AEROSOL_TYPE) -- choose from "
            f"{REGISTRY_TYPES}")
    return validate_aerosol_type(t)


def validate_aerosol_type(aerosol_type: str) -> str:
    """Raise `ValueError` unless `aerosol_type` is one of `REGISTRY_TYPES`; otherwise return it unchanged."""
    if aerosol_type not in REGISTRY_TYPES:
        raise ValueError(f"unknown aerosol type {aerosol_type!r}; choose from {REGISTRY_TYPES}")
    return aerosol_type


def amplitude_reference_um() -> float:
    """Wavelength at which `amplitude_aerosol` / the scene AOD are defined: the O2-A band (0.765 um) -- see
    `aerosol_mie.REF_FPA`. Every type shares this one reference now (the legacy scheme's own 1.608 um
    reference no longer exists)."""
    return aerosol_mie.BAND_CENTRES_UM[aerosol_mie.REF_FPA]


def band_props_for_wavelength(aerosol_type, wl_um: float) -> tuple[float, float, float]:
    """`(ssa, g, tau_scale)` for `aerosol_type` at the FPA whose band centre is closest to `wl_um`."""
    t = resolve_aerosol_type(aerosol_type)
    fpa = min(aerosol_mie.BAND_CENTRES_UM, key=lambda f: abs(aerosol_mie.BAND_CENTRES_UM[f] - wl_um))
    ssa, g, q = aerosol_mie.band_properties(t)[fpa]
    return float(ssa), float(g), float(q)
