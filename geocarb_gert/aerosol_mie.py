"""Mie optical properties of each aerosol type at every GeoCarb band centre (2026-09-25).

Why: gert's registry only has three band slots (O2-A / 1.6 um / 1.65 um), and the simulator used the O2-A slot for
FPA0 and the SAME 1.6 um slot for FPA1, FPA2 and FPA3 (see aerosol_defaults.py). Real aerosol extinction, single-
scattering albedo and asymmetry all change across 0.76-2.3 um, and that spectral linkage is what ties the aerosol
optical depth of one band to the next in a multi-band retrieval (user, 2026-09-25: "what ACOS constrains ... is
linked by spectral information across the bands ... dependent on optical properties that are known/assumed").
So for every type, (qext, ssa, g) are computed at the four band centres from an assumed size distribution and
refractive index; truth and retrieval use the same values.

ASSUMPTIONS (typical literature-style values entered from memory, not looked up in a database -- replace with an
authoritative table, e.g. OPAC / AERONET / the MERRA-2 species used by ACOS, when one is chosen):
  * lognormal NUMBER size distribution (geometric radius rg [um], geometric width sigma_g);
  * complex refractive index m = n + ik at each band centre (Bohren & Huffman sign convention, k > 0 absorbs);
  * homogeneous spheres (dust is not spherical: its g and phase function are approximate).
Extinction is reported relative to the O2-A (FPA0, 0.765 um) band: `amplitude_aerosol` and the scene's AOD values
(0.05 background, 0.35 haze peak) are defined there for these types. (The legacy registry scheme defines them at 1.6 um.)
Bohren & Huffman (1983) BHMIE; no external Mie library is available in this environment.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

BAND_CENTRES_UM = {0: 0.765, 1: 1.608, 2: 2.065, 3: 2.325}     # FPA0..FPA3 band centres
REF_FPA = 0                                                    # amplitude_aerosol / the scene AOD refer to the O2-A band (0.765 um),
                                                               # as the scene docstrings and ACOS-style AOD conventions say

#: name -> microphysics. n, k are listed for FPA0..FPA3.
AEROSOL_MICROPHYSICS = {
    "smoke": dict(rg=0.09, sigma_g=1.5, n=[1.52, 1.51, 1.50, 1.50], k=[0.015, 0.015, 0.015, 0.015],
                  note="fresh biomass-burning smoke, fine mode (volume median radius ~0.15 um), moderately absorbing"),
    "sulfate": dict(rg=0.08, sigma_g=1.5, n=[1.52, 1.50, 1.48, 1.46], k=[1e-7, 1e-4, 5e-4, 1e-3],
                    note="ammonium sulfate, fine mode, essentially non-absorbing (weak SWIR bands)"),
    "sea_salt": dict(rg=0.6, sigma_g=1.8, n=[1.50, 1.49, 1.48, 1.47], k=[1e-6, 2e-4, 1e-3, 1e-3],
                     note="sea salt, coarse mode (r_eff ~1.4 um)"),
    "dust": dict(rg=0.7, sigma_g=1.9, n=[1.53, 1.52, 1.51, 1.50], k=[0.0030, 0.0025, 0.0020, 0.0030],
                 note="mineral dust, coarse mode (r_eff ~2 um), weakly absorbing; spheres only approximate its phase function"),
    "cloud_water": dict(rg=8.0, sigma_g=1.3, n=[1.329, 1.318, 1.305, 1.298], k=[1.3e-7, 8.7e-5, 1.4e-3, 3.5e-4],
                        note="liquid water droplets, r_eff ~9.5 um"),
}


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
def _dist_props(name: str, fpa: int):
    """Extinction cross-section per particle, ssa, g for the type's lognormal at FPA `fpa`'s centre."""
    mp = AEROSOL_MICROPHYSICS[name]
    wl = BAND_CENTRES_UM[fpa]
    rg, sg = mp["rg"], mp["sigma_g"]
    lnr = np.linspace(np.log(rg) - 4.0 * np.log(sg), np.log(rg) + 4.0 * np.log(sg), 161)
    r = np.exp(lnr)
    w = np.exp(-0.5 * ((lnr - np.log(rg)) / np.log(sg)) ** 2)          # number weight per d ln r
    m = complex(mp["n"][fpa], mp["k"][fpa])                           # B&H convention: k > 0 absorbs
    q = np.array([bhmie(2.0 * np.pi * ri / wl, m) for ri in r])
    area = np.pi * r ** 2
    cext = np.sum(w * area * q[:, 0]) / np.sum(w)
    csca = np.sum(w * area * q[:, 1]) / np.sum(w)
    g = np.sum(w * area * q[:, 1] * q[:, 2]) / np.sum(w * area * q[:, 1])
    return float(cext), float(csca / cext), float(g)


@lru_cache(maxsize=None)
def compute_band_table(name: str):
    """{fpa: (ssa, g, qext relative to the O2-A band)} recomputed from the microphysics (slow)."""
    raw = {f: _dist_props(name, f) for f in BAND_CENTRES_UM}
    ref = raw[REF_FPA][0]
    return {f: (raw[f][1], raw[f][2], raw[f][0] / ref) for f in raw}


#: Result of `compute_band_table` stored so runtime (and every respawned pool worker) does not redo the Mie
#: evaluations; `python aerosol_mie.py` checks it against a fresh calculation.
BAND_TABLES = {
    "smoke": {0: (0.89402, 0.49165, 1.00000), 1: (0.69182, 0.19139, 0.11854), 2: (0.53931, 0.11871, 0.05807), 3: (0.46141, 0.09433, 0.04307)},
    "sulfate": {0: (1.00000, 0.44485, 1.00000), 1: (0.99594, 0.15278, 0.07771), 2: (0.95947, 0.09334, 0.02891), 3: (0.88666, 0.07330, 0.01821)},
    "sea_salt": {0: (0.99997, 0.69708, 1.00000), 1: (0.99800, 0.69821, 1.13075), 2: (0.99209, 0.70843, 1.07637), 3: (0.99278, 0.71380, 1.01346)},
    "dust": {0: (0.91840, 0.72775, 1.00000), 1: (0.96615, 0.68756, 1.14410), 2: (0.97865, 0.68986, 1.16310), 3: (0.97183, 0.69829, 1.14964)},
    "cloud_water": {0: (0.99998, 0.85889, 1.00000), 1: (0.99372, 0.84411, 1.03436), 2: (0.93145, 0.85310, 1.05219), 3: (0.98278, 0.83562, 1.06681)},
}


def band_properties(name: str):
    """{fpa: (ssa, g, qext_relative_to_the_O2-A_band)} for aerosol type `name` (the stored table)."""
    return BAND_TABLES[name]


if __name__ == "__main__":
    import sys
    x = 0.02
    q_ext, q_sca, g = bhmie(x, 1.5 + 0j)
    m2 = 1.5 ** 2
    ray = (8.0 / 3.0) * x ** 4 * abs((m2 - 1) / (m2 + 2)) ** 2
    print(f"Rayleigh check x=0.02: Qsca {q_sca:.4e} vs analytic {ray:.4e} (ratio {q_sca / ray:.4f}); Qext-Qsca {q_ext - q_sca:.1e}; g {g:.1e}")
    q_ext, q_sca, g = bhmie(3.0, 1.33 + 0j)
    print(f"non-absorbing x=3: Qext {q_ext:.4f} Qsca {q_sca:.4f} (equal) g {g:.3f}")
    fresh = {n: compute_band_table(n) for n in AEROSOL_MICROPHYSICS}
    if "--print-table" in sys.argv:
        print("BAND_TABLES = {")
        for n, t in fresh.items():
            print(f'    "{n}": {{' + ", ".join(f"{f}: ({a:.5f}, {b:.5f}, {c:.5f})" for f, (a, b, c) in t.items()) + "},")
        print("}")
    for n, t in fresh.items():
        mp = AEROSOL_MICROPHYSICS[n]
        print(f"{n}: r_eff {mp['rg'] * np.exp(2.5 * np.log(mp['sigma_g']) ** 2):.2f} um -- {mp['note']}")
        for f, (ssa, g, q) in t.items():
            print(f"   FPA{f} {BAND_CENTRES_UM[f]:.3f} um: ssa={ssa:.3f} g={g:.3f} qext/qext(FPA{REF_FPA})={q:.3f}")
    if BAND_TABLES:
        assert all(abs(a - b) < 2e-5 for n in fresh for f in fresh[n] for a, b in zip(fresh[n][f], BAND_TABLES[n][f])), "stored tables are stale"
        print("stored BAND_TABLES match the recomputed values")
