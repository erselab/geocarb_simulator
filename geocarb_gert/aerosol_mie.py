"""Mie optical properties of the smoke aerosol at each GeoCarb band centre (2026-09-25).

Why: gert's registry only has three band slots (O2-A / 1.6 um / 1.65 um), and the simulator used the O2-A
slot for FPA0 and the SAME 1.6 um slot for FPA1, FPA2 and FPA3 (see aerosol_defaults.py). A real fine-mode
aerosol's extinction keeps falling from 1.6 to 2.3 um, and that spectral linkage is what ties the aerosol
optical depth of one band to the next in a multi-band retrieval (user, 2026-09-25). This module computes
(qext, ssa, g) at the four band centres from a stated size distribution and refractive index, so truth
and retrieval share a physically consistent, wavelength-dependent aerosol.

Model (an ASSUMPTION, stated openly): a single fine-mode smoke type,
  * lognormal number size distribution, geometric radius R_G_UM, geometric width SIGMA_G;
  * real refractive index N_REAL, constant with wavelength;
  * imaginary index K_IMAG, constant with wavelength.
The size is tuned so the O2-A/1.6 um extinction ratio equals the registry's (2.38); single-scattering albedo and
asymmetry are then predictions, compared with the registry in the printout below (the registry's asymmetry is lower).
Homogeneous spheres, Bohren & Huffman (1983) BHMIE; no external Mie library is available in this environment.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

R_G_UM = 0.235          # number-lognormal geometric mean radius [um]; tuned so qext(0.765 um)/qext(1.608 um) = 2.38, the
                        # registry's own smoke ratio (1.00/0.42). NOTE r_eff = R_G*exp(2.5 ln^2 sigma_g) = 0.41 um: the
                        # registry's extinction ratio needs particles larger than its quoted r_eff ~ 0.12 um (r_eff 0.12
                        # would give a ratio of ~11, Angstrom ~3.3, and SSA falling to ~0.5 by 2.3 um).
SIGMA_G = 1.6           # geometric standard deviation
N_REAL = 1.52           # real refractive index, constant with wavelength
K_IMAG = 0.02           # imaginary index, constant with wavelength (assumed; gives SSA ~0.90 at 0.765 um, like the registry)
BAND_CENTRES_UM = {0: 0.765, 1: 1.608, 2: 2.065, 3: 2.325}     # FPA0..FPA3 band centres
REF_FPA = 1                                                    # amplitude_aerosol's reference band (1.6 um), as before


def bhmie(x: float, m: complex):
    """Efficiencies (Qext, Qsca) and asymmetry g for one homogeneous sphere (Bohren & Huffman BHMIE)."""
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


@lru_cache(maxsize=None)
def _dist_props(wl_um: float, k: float):
    """(qext_per_particle_area-weighted ..., ) -> extinction cross-section per particle, ssa, g for the lognormal."""
    lnr = np.linspace(np.log(R_G_UM) - 4.0 * np.log(SIGMA_G), np.log(R_G_UM) + 4.0 * np.log(SIGMA_G), 161)
    r = np.exp(lnr)
    w = np.exp(-0.5 * ((lnr - np.log(R_G_UM)) / np.log(SIGMA_G)) ** 2)      # number weight per d ln r
    m = complex(N_REAL, k)          # Bohren & Huffman convention: m = n + ik, k > 0 absorbs
    q = np.array([bhmie(2.0 * np.pi * ri / wl_um, m) for ri in r])
    area = np.pi * r ** 2
    cext = np.sum(w * area * q[:, 0]) / np.sum(w)
    csca = np.sum(w * area * q[:, 1]) / np.sum(w)
    g = np.sum(w * area * q[:, 1] * q[:, 2]) / np.sum(w * area * q[:, 1])
    return float(cext), float(csca / cext), float(g)


#: Result of `compute_smoke_band_properties()` with the constants above, stored so runtime (and every respawned pool
#: worker) does not redo ~700 Mie evaluations: {fpa: (ssa, g, qext relative to the FPA1 / 1.6 um reference)}.
SMOKE_BAND_TABLE = {0: (0.89642, 0.71446, 2.37500), 1: (0.89645, 0.61929, 1.00000),
                    2: (0.87976, 0.55627, 0.60887), 3: (0.86765, 0.52037, 0.46639)}


def smoke_band_properties():
    """{fpa: (ssa, g, qext_relative_to_reference_band)} for FPA0..FPA3 (the stored table)."""
    return SMOKE_BAND_TABLE


@lru_cache(maxsize=None)
def compute_smoke_band_properties():
    """Recompute the table from the size distribution / refractive index (slow); `python aerosol_mie.py` checks the
    stored SMOKE_BAND_TABLE against it."""
    raw = {f: _dist_props(wl, K_IMAG) for f, wl in BAND_CENTRES_UM.items()}
    ref = raw[REF_FPA][0]
    return {f: (raw[f][1], raw[f][2], raw[f][0] / ref) for f in raw}


if __name__ == "__main__":
    # validation: Rayleigh limit and energy conservation for a non-absorbing sphere
    x = 0.02
    q_ext, q_sca, g = bhmie(x, 1.5 + 0j)
    m2 = 1.5 ** 2
    ray = (8.0 / 3.0) * x ** 4 * abs((m2 - 1) / (m2 + 2)) ** 2
    print(f"Rayleigh check x=0.02: Qsca {q_sca:.4e} vs analytic {ray:.4e} (ratio {q_sca / ray:.4f}); Qext-Qsca {q_ext - q_sca:.1e}; g {g:.1e}")
    q_ext, q_sca, g = bhmie(3.0, 1.33 + 0j)
    print(f"non-absorbing x=3: Qext {q_ext:.4f} Qsca {q_sca:.4f} (equal) g {g:.3f}")
    print("registry smoke (gert): ssa 0.90/0.87/0.86, g 0.55/0.50/0.49, qext relative to 1.6 um 2.38/1.00/0.98 (slots O2-A/1.6/1.65 um)")
    fresh = compute_smoke_band_properties()
    for f, (ssa, g, q) in fresh.items():
        print(f"FPA{f} {BAND_CENTRES_UM[f]:.3f} um: ssa={ssa:.3f} g={g:.3f} qext/qext(FPA{REF_FPA})={q:.3f}")
    assert all(abs(a - b) < 2e-5 for f in fresh for a, b in zip(fresh[f], SMOKE_BAND_TABLE[f])), "stored table is stale"
    print("stored SMOKE_BAND_TABLE matches the recomputed values")
