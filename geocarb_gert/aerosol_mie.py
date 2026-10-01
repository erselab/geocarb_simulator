"""Per-band aerosol optical properties, sourced directly from the operational ACOS/OCO-2 aerosol model
(2026-09-28; supersedes the earlier "typical literature-style values entered from memory" version of this
module -- user: "I don't want to use that setup any longer. I want a list of defined types with band-specific
parameters that are defined in literature. If a type is called that isn't in the list, the code should error,
not fall back to a default").

SOURCE: O'Dell et al. (2018), "Improved retrievals of carbon dioxide from Orbiting Carbon Observatory-2 with
the version 8 ACOS algorithm", Atmos. Meas. Tech. 11, 6539-6576, Figure 3 -- (Q_ext relative to 755 nm, single-
scattering albedo, asymmetry parameter) vs wavelength for the 7 MERRA-2-derived aggregated types ACOS's own
L2FP retrieval uses: DU (dust), SS (sea salt), BC (black carbon), OC (organic carbon), SO (sulfate), WC (water
cloud), IC (ice cloud). The paper's own Sec. 3.1 / Crisp et al. (2017) describe how these are aggregated from
MERRA-2's 15 native species; ACOS does not publish a numeric table, only Fig. 3's curves.

**These values are READ OFF Fig. 3 by eye** (dashed lines mark the OCO-2 band centres, 0.76/1.61/2.06 um --
three of GeoCarb's four bands almost exactly), not extracted from a machine-readable table -- precision is
roughly +/-0.05 in (ssa, g, qext-ratio), not the many-digit precision a real lookup table would give. GeoCarb's
4th band (FPA3, 2.325 um) is just past the right edge of ACOS's 3-band plot (which stops near 2.1 um); those
values are a short trend extrapolation of each curve's own visible slope near 2.06 um, marked below.

GeoCarb type -> ACOS component mapping:
  * "sulfate" = SO, "sea_salt" = SS, "dust" = DU, "cloud_water" = WC directly.
  * "smoke" = an 80/20-by-optical-depth OC/BC mixture (user's own choice, 2026-09-28): real biomass-burning
    smoke is a BC+OC mixture (Chin et al. 2002 cite a ~1:7 BC:OC mass ratio for biomass burning), not ACOS's
    own pure "BC" end-member -- pure BC alone (ssa ~0.03-0.15 across these bands) is far more absorbing than
    real-world smoke (typically ssa ~0.85-0.95). The two components are mixed by proper radiative-transfer
    mixing rules (`_mix_two`, optical-depth-weighted ssa, scattering-optical-depth-weighted g), not just
    averaged -- see that function's own docstring.

No default and no silent fallback: `band_properties(name)` raises KeyError for any name not in `AEROSOL_TYPES`
-- callers (`aerosol_defaults.validate_aerosol_type`) turn that into a clear `ValueError` listing the valid
types. There is no legacy/registry scheme left to fall back to (removed in the same change).
"""
from __future__ import annotations

import numpy as np

BAND_CENTRES_UM = {0: 0.765, 1: 1.608, 2: 2.065, 3: 2.325}     # FPA0..FPA3 band centres
REF_FPA = 0                                                    # amplitude_aerosol / the scene AOD refer to the O2-A band (0.765 um)

#: Raw ACOS-native curves, {component: {fpa: (ssa, g, qext_relative_to_O2-A)}}, read off O'Dell et al. (2018)
#: Fig. 3 at the three OCO-2 band centres (FPA0-2) plus a trend extrapolation to FPA3 (see module docstring).
#: qext is already relative to ~755 nm/O2-A, matching this project's own REF_FPA=0 convention directly -- no
#: rescaling needed (ACOS's reference wavelength, 755 nm, and GeoCarb's O2-A band centre, 765 nm, are close
#: enough that Fig. 3's own y-axis serves as-is).
_ACOS_CURVES = {
    "DU": {0: (0.93, 0.73, 1.00), 1: (0.91, 0.72, 1.05), 2: (0.90, 0.72, 1.08), 3: (0.90, 0.72, 1.09)},
    "SS": {0: (1.00, 0.73, 1.00), 1: (1.00, 0.76, 1.00), 2: (0.99, 0.80, 1.05), 3: (0.99, 0.82, 1.07)},
    "BC": {0: (0.15, 0.28, 1.00), 1: (0.04, 0.10, 0.55), 2: (0.015, 0.05, 0.35), 3: (0.01, 0.04, 0.30)},
    "OC": {0: (1.00, 0.60, 1.00), 1: (0.83, 0.30, 0.35), 2: (0.80, 0.18, 0.15), 3: (0.78, 0.15, 0.12)},
    "SO": {0: (1.00, 0.73, 1.00), 1: (0.97, 0.45, 0.35), 2: (0.85, 0.27, 0.15), 3: (0.80, 0.23, 0.12)},
    "WC": {0: (1.00, 0.83, 1.00), 1: (1.00, 0.80, 1.00), 2: (0.99, 0.83, 1.00), 3: (0.98, 0.83, 1.00)},
    "IC": {0: (1.00, 0.73, 1.00), 1: (0.98, 0.78, 1.05), 2: (0.85, 0.83, 1.05), 3: (0.82, 0.84, 1.05)},
}

#: Optical-depth split (at the O2-A reference band) for the smoke = OC+BC mixture -- user's choice, 2026-09-28.
_SMOKE_OC_FRAC = 0.80
_SMOKE_BC_FRAC = 0.20


def _mix_two(curve_a: dict, curve_b: dict, frac_a: float, frac_b: float) -> dict:
    """Combine two components' per-band (ssa, g, qext_rel) curves into one mixture's curve, `frac_a`/`frac_b`
    being each component's share of the total optical depth AT THE REFERENCE BAND (so `frac_a + frac_b == 1`
    there by construction). At every other band, each component's own qext_rel curve moves its share of the
    optical depth independently (`tau_x(fpa) = frac_x * qext_x(fpa)`, since qext_x is already relative to the
    ref band) -- the mixture is NOT the simple average of the two curves, which would silently assume both
    components' optical depth moves in lockstep across bands.

    Standard aerosol mixing rules (external mixture, i.e. distinct particles of each type coexisting, not one
    particle made of both -- the only kind of mixture that composes from each component's OWN Mie-derived
    ssa/g without needing a fresh Mie calculation on a combined size/composition distribution):
      qext_mix(fpa)  = tau_a(fpa) + tau_b(fpa)                                    -- extinction optical depths add
      ssa_mix(fpa)   = (tau_a*ssa_a + tau_b*ssa_b) / (tau_a + tau_b)              -- scattering / total extinction
      g_mix(fpa)     = (tau_a*ssa_a*g_a + tau_b*ssa_b*g_b) / (tau_a*ssa_a + tau_b*ssa_b)  -- g weights only the
                        SCATTERED photons (it describes the phase function of light that scatters, not light
                        that's absorbed), so the weight is each component's scattering optical depth, not its
                        total extinction optical depth.
    """
    out = {}
    for fpa in curve_a:
        ssa_a, g_a, q_a = curve_a[fpa]
        ssa_b, g_b, q_b = curve_b[fpa]
        tau_a, tau_b = frac_a * q_a, frac_b * q_b
        tau_ext = tau_a + tau_b
        sca_a, sca_b = tau_a * ssa_a, tau_b * ssa_b
        tau_sca = sca_a + sca_b
        out[fpa] = (tau_sca / tau_ext, (sca_a * g_a + sca_b * g_b) / tau_sca, tau_ext)
    return out


#: {geocarb type name -> {fpa: (ssa, g, qext_relative_to_O2-A)}}. The only source of per-band aerosol optical
#: properties in this project -- see module docstring for provenance and precision caveats.
AEROSOL_TYPES = {
    "sulfate": _ACOS_CURVES["SO"],
    "sea_salt": _ACOS_CURVES["SS"],
    "dust": _ACOS_CURVES["DU"],
    "cloud_water": _ACOS_CURVES["WC"],
    "smoke": _mix_two(_ACOS_CURVES["OC"], _ACOS_CURVES["BC"], _SMOKE_OC_FRAC, _SMOKE_BC_FRAC),
}


def band_properties(name: str) -> dict:
    """`{fpa: (ssa, g, qext_relative_to_the_O2-A_band)}` for GeoCarb aerosol type `name`.

    Raises `KeyError` for any name not in `AEROSOL_TYPES` -- no default, no fallback. Callers should use
    `aerosol_defaults.validate_aerosol_type`/`resolve_aerosol_type`, which turn this into a clear `ValueError`
    listing the valid types, rather than let a raw `KeyError` surface.
    """
    return AEROSOL_TYPES[name]


def bhmie(x: float, m: complex):
    """Efficiencies (Qext, Qsca) and asymmetry g for one homogeneous sphere (Bohren & Huffman BHMIE).

    Not used by `band_properties`/`AEROSOL_TYPES` above (those come directly from ACOS's own published
    per-band values, not a Mie calculation) -- kept as a general-purpose utility for anyone who wants to add
    a NEW type from an assumed refractive index and size distribution in the future, and exercised by this
    module's own `__main__` self-checks below.
    """
    nstop = int(x + 4.0 * x ** (1.0 / 3.0) + 2.0)
    y = x * m
    nmx = int(max(nstop, abs(y)) + 15)
    d = np.zeros(nmx + 1, dtype=complex)
    for n in range(nmx, 0, -1):                    # downward recurrence for D_n(y)
        d[n - 1] = n / y - 1.0 / (d[n] + n / y)
    psi0, psi1 = np.cos(x), np.sin(x)
    chi0, chi1 = -np.sin(x), np.cos(x)
    xi1 = complex(psi1, -chi1)
    qsca = qext = 0.0
    gsum = 0.0
    an1 = bn1 = 0.0 + 0.0j
    for n in range(1, nstop + 1):
        fn = (2.0 * n + 1.0) / (n * (n + 1.0))
        psi = (2.0 * n - 1.0) * psi1 / x - psi0
        chi = (2.0 * n - 1.0) * chi1 / x - chi0
        xi = complex(psi, -chi)
        an = ((d[n] / m + n / x) * psi - psi1) / ((d[n] / m + n / x) * xi - xi1)
        bn = ((m * d[n] + n / x) * psi - psi1) / ((m * d[n] + n / x) * xi - xi1)
        qsca += (2.0 * n + 1.0) * (abs(an) ** 2 + abs(bn) ** 2)
        qext += (2.0 * n + 1.0) * (an.real + bn.real)
        if n > 1:
            gsum += ((n - 1.0) * (n + 1.0) / n) * (an1 * an.conjugate() + bn1 * bn.conjugate()).real \
                + ((2.0 * n - 1.0) / ((n - 1.0) * n)) * (an1 * bn1.conjugate()).real
        an1, bn1 = an, bn
        psi0, psi1 = psi1, psi
        chi0, chi1 = chi1, chi
        xi1 = complex(psi1, -chi1)
    qsca *= 2.0 / x ** 2
    qext *= 2.0 / x ** 2
    g = 4.0 / (qsca * x ** 2) * gsum
    return qext, qsca, g


if __name__ == "__main__":
    x = 0.02
    q_ext, q_sca, g = bhmie(x, 1.5 + 0j)
    m2 = 1.5 ** 2
    ray = (8.0 / 3.0) * x ** 4 * abs((m2 - 1) / (m2 + 2)) ** 2
    print(f"Rayleigh check x=0.02: Qsca {q_sca:.4e} vs analytic {ray:.4e} (ratio {q_sca / ray:.4f}); Qext-Qsca {q_ext - q_sca:.1e}; g {g:.1e}")
    q_ext, q_sca, g = bhmie(3.0, 1.33 + 0j)
    print(f"non-absorbing x=3: Qext {q_ext:.4f} Qsca {q_sca:.4f} (equal) g {g:.3f}")
    print("\nAEROSOL_TYPES (O'Dell et al. 2018 Fig. 3, see module docstring for provenance/precision):")
    for n, t in AEROSOL_TYPES.items():
        print(f"  {n}:")
        for f, (ssa, g, q) in t.items():
            print(f"    FPA{f} {BAND_CENTRES_UM[f]:.3f} um: ssa={ssa:.3f} g={g:.3f} qext/qext(FPA{REF_FPA})={q:.3f}")
