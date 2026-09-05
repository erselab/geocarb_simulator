"""Along-slit-varying truth atmosphere for the composition/pressure stress test.

Every scene built elsewhere in this package (``focalplane.py``, ``scene.py``)
holds the *atmosphere* fixed along the slit and only varies surface albedo.
This module was originally built as the exact opposite: a truth atmosphere
whose gas columns (CO2, CH4, CO, H2O) and surface pressure vary continuously
along the ~2800 km slit, at one FIXED albedo, for testing whether
keystone/smile row-crossing confuses a retrieval when the along-slit
variability is geophysical, not just radiometric.

Since 2026-08-17 it can also vary surface albedo along the slit at the same
time (:func:`albedo_at`, enabled with ``build_lookup_radiance(...,
vary_albedo=True)``), giving a scene where composition, surface pressure and
reflectance all vary together -- the realistic case, and the one needed
before any albedo-induced bias can be measured. It is opt-in, and OFF by
default, so every result produced before that date reproduces bit-for-bit.

Design (see ``scripts/gd_along_slit_atm_profiles.py`` for the plots this was
approved from): baselines from :func:`geocarb_gert.scene.reference_atmosphere`
/ ``_WELL_MIXED`` (CO2 415 ppm, CH4 1900 ppb, CO 100 ppb, H2O surface 1% VMR,
p_surface 1013.25 hPa). Each quantity gets a broad smooth background
gradient; CO2 and CO share a broad co-located plume (a wildfire-in-mountains
scenario, combustion co-emission) plus a topographic pressure depression
nearby; CH4 gets its own broad plume at a different location (an
independent wetland/oil-and-gas-type source); H2O gets a smooth
arid-to-humid climatological gradient across the whole transect, no plume.
Small (~10 km wide, a few detector rows) "hot spot" point sources are
layered on top -- CO2 and CO hot spots are coincident (combustion
co-emission); CH4's are independent.

Coordinate convention: physical along-slit distance is linear in eta
(eta=-1..+1 -> -SLIT_HALF_KM..+SLIT_HALF_KM, ~2800 km total by default --
sourced from input/geocarb_instrument.yml's geometry.slit_length_km since
Phase B of the config-consolidation plan, which also brought
LongSlitGeoSatellite's own default into exact agreement with this value
rather than the two merely being close).
"""
from __future__ import annotations

import multiprocessing as mp
import warnings
from functools import lru_cache
from typing import Callable

import numpy as np

from gert.atmosphere import AtmosphericProfile
from .levels import sigma_levels

from model_sampler import pressure_to_alt_std_atm

from .scene import _MW_RATIO, _std_temperature, _WELL_MIXED
from .gd_render import available_cpus
from .mission_config import GeoCarbInstrumentConfig as _GeoCarbInstrumentConfig

#: Sourced from input/geocarb_instrument.yml's geometry.slit_length_km at
#: import time (Phase B of the config-consolidation plan) -- was a bare
#: 1400.0 literal. Kept as a module constant since it's imported bare
#: throughout this package and scripts/; a non-default geometry config
#: must read cfg.geometry.slit_half_km directly rather than this constant,
#: which always reflects the checked-in default YAML. See
#: geocarb_gert/mission_config.py.
SLIT_HALF_KM = _GeoCarbInstrumentConfig.from_yaml().geometry.slit_half_km

# -- baselines (geocarb_gert.scene.reference_atmosphere / _WELL_MIXED) --
XCO2_BG_PPM = 415.0
XCH4_BG_PPB = 1900.0
XCO_BG_PPB = 100.0
H2O_BG_VMR = 1.0e-2      # surface VMR, fraction
P_BG_HPA = 1013.25

#: How far below the standard-atmosphere ceiling (`P_BG_HPA`, = sea level)
#: the truth surface-pressure field is held. `pressure_to_alt_std_atm`
#: returns NaN above that pressure, so this is the room a RETRIEVAL has to
#: move surface pressure upward before the forward model goes NaN.
#:
#: Raised from 0.1 to 10.0 hPa (user, 2026-08-18). 0.1 hPa was enough to keep
#: the TRUTH scene valid, which was all it had to do while p_surface was
#: always frozen -- but it is not enough to RETRIEVE p_surface. With
#: `kind="scale"`, `gauss_newton_state`'s finite-difference probe alone is
#: `p * (1 + 1e-3)` ~ +1.0 hPa, an order of magnitude over that margin, so
#: every window spanning the eastern rise (rows 583-844, 26% of FPA2's slit)
#: died with `cannot convert float NaN to integer` in the first free-p sweep.
#: 10 hPa is ~1% of background: 10x the FD probe, and enough room for a
#: genuine ~1% retrieval excursion on top of it.
#:
#: Note the sign convention: a LARGER headroom means LOWER surface pressure,
#: i.e. the surface sits higher in altitude, further from the sea-level floor
#: of the standard atmosphere.
P_HEADROOM_HPA = 10.0

# Small, localized "hot spots" -- point-source-scale features, deliberately
# narrower than the along-slit ground sample distance (~2.7 km/row for the
# ~2800 km slit over 1024 rows) times a handful of rows, i.e. only a few
# rows wide -- to stress-test whether keystone/smile row-crossing preserves
# or smears a genuinely localized source, as opposed to the broad regional
# plumes/gradients below. Placed away from the broad features so each shows
# up in isolation.
HOTSPOTS_CO2 = [(-1100.0, 10.0, 5.0), (1050.0, 9.0, 4.0)]     # (x0_km, width_km, amp_ppm)
HOTSPOTS_CH4 = [(-50.0, 12.0, 35.0), (1050.0, 9.0, 25.0)]     # (x0_km, width_km, amp_ppb) -- second co-located with the CO2 one at +1050 km (a small combined facility)
HOTSPOTS_CO = [(-1100.0, 10.0, 35.0), (1050.0, 9.0, 28.0)]    # (x0_km, width_km, amp_ppb) -- coincident with both CO2 hot spots (combustion co-emission)


# -- surface albedo along the slit (opt-in; see `albedo_at`) --
#
# Added 2026-08-17. Until then this module varied ONLY the atmosphere
# (gases + surface pressure) and every caller passed one fixed scalar
# albedo -- `gd_test._band_setup` used `albedo_for(inst, "desert")`, a
# single desert value for the entire 2800 km slit. That was a deliberate
# design choice (see this module's own opening docstring), but it makes the
# scene unable to say anything about albedo-induced bias, and it is
# backwards relative to reality: per `JOINT_BLOCK_MIGRATION_PLAN.md` Sec.4.1,
# surface albedo varies along the slit MORE than well-mixed gases do, not
# less -- it is a land-cover property (bare soil vs. vegetation vs. water,
# field boundaries) that changes sharply within a single window's footprint.
#
# Structure, per that section's reasoning: land-cover PATCHES drawn from
# `geocarb_gert.scene._BAND_ALBEDO`'s own surface archetypes (so every band
# stays mutually consistent -- a forest patch is dark in CO2_strong and
# bright in O2_A automatically, rather than each band getting an
# independent random number), plus short-correlation-length fine-scale
# variability on top (within-field brightness variation), which is the
# component that actually varies on scales shorter than the gas features.
#
# Patch boundaries are irregular and the segment lengths (~90-400 km) are
# large compared to the ~2.7 km/row GSD, so each patch spans many rows
# while boundaries stay sharp relative to a row -- the regime that stresses
# keystone row-crossing.
SURFACE_PATCHES = [
    (-1400.0, "desert"), (-1210.0, "grass"), (-1015.0, "desert"),
    (-780.0, "forest"), (-505.0, "grass"), (-260.0, "desert"),
    (-95.0, "forest"), (185.0, "grass"), (430.0, "water"),
    (545.0, "forest"), (820.0, "desert"), (1015.0, "grass"),
    (1230.0, "forest"),
]
ALBEDO_EDGE_KM = 3.0      # boundary softness, ~1 detector row
ALBEDO_COV = 0.10         # fractional 1-sigma within-patch variability
ALBEDO_CORR_KM = 0.5      # its correlation length -- 2026-08-28: set to match a real
                          # MODIS-like external product's own ~500m resolution (the
                          # motivating example from the start of this feature's design
                          # conversation), well below anything a detector row (~6km GSD)
                          # or a retrieval bin can resolve directly -- genuine sub-bin
                          # texture, not just sub-detector-row-per-CO2-hotspot-scale
                          # (was 10.0, "matches the CO2 hot-spot scale", 2026-08-25).
ALBEDO_FINE_SEED = 20260817
_ALBEDO_MIN, _ALBEDO_MAX = 0.005, 0.95


def albedo_info_density(x_km, peak_width_km: float = 5.0, floor: float = 0.1):
    """Synthetic stand-in for "how much a real fine-resolution external
    product (e.g. MODIS albedo, ~500m) would inform us here" -- peaked at
    each internal `SURFACE_PATCHES` boundary (a real product would resolve
    a genuine, sharp transition there), a floor elsewhere. NOT derived
    from any real external data -- built to demonstrate the
    resolution-follows-information mechanism in
    `geocarb_gert.joint_state.information_weighted_bin_centers`; see
    `docs/PROJECT_STATUS.md` Sec.8. In a real system this would instead
    come from an actual MODIS-derived local-resolving-power/confidence
    field, reprojected onto the slit.

    Deliberately correlates with the truth's own patch structure -- that
    is the whole point (a real external product genuinely WOULD show more
    structure right at a real land-cover transition), not something being
    hidden.
    """
    boundaries = np.array([p[0] for p in SURFACE_PATCHES[1:]])
    x = np.atleast_1d(np.asarray(x_km, dtype=float))
    dist = np.min(np.abs(x[:, None] - boundaries[None, :]), axis=1)
    peak = np.exp(-0.5 * (dist / peak_width_km) ** 2)
    return floor + (1.0 - floor) * peak


def _gauss(x, x0, width, amp):
    return amp * np.exp(-0.5 * ((x - x0) / width) ** 2)


def _correlated_field(seed, corr_km=None, cov=None, oversample=5.0):
    """Deterministic Gaussian-smoothed white noise, grid spacing chosen to
    actually RESOLVE `corr_km` rather than a fixed 1 km grid.

    Before 2026-08-28, this hardcoded a 1 km grid, fine when `ALBEDO_
    CORR_KM` was 10-30 km (5-30 grid points per correlation length) but
    silently wrong once it dropped to a MODIS-like 500m: a correlation
    length SHORTER than the grid spacing can't be represented at all (the
    Gaussian kernel's own half-width, `4*corr_km` grid points, would round
    to 2 -- a couple of adjacent 1km cells, not a genuinely finer field).
    Grid spacing is now `min(1.0, corr_km/oversample)` km, `oversample=5`
    points per correlation length by default -- enough for the smoothing
    kernel to be resolved rather than aliased, capped at the original 1km
    so nothing changes for any existing caller using the original
    10-30km scale.

    Renormalised so its own standard deviation is exactly `cov` --
    smoothing otherwise shrinks the variance by an amount that depends on
    `corr_km`, which would make the requested coefficient of variation
    silently wrong.

    Seeded (not drawn fresh per call) so every process (including
    `build_lookup_radiance`'s forked pool workers) sees the identical
    field and the scene stays reproducible; callers `lru_cache` this
    themselves since the grid itself, not just the seed, determines cost.
    """
    corr_km = ALBEDO_CORR_KM if corr_km is None else corr_km
    cov = ALBEDO_COV if cov is None else cov
    dx = min(1.0, corr_km / oversample)
    x = np.arange(-SLIT_HALF_KM, SLIT_HALF_KM + dx, dx)
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(len(x))
    # Gaussian kernel, truncated at 4 sigma (in grid points, not km)
    half = max(1, int(round(4 * corr_km / dx)))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) * dx / corr_km) ** 2)
    k /= k.sum()
    f = np.convolve(w, k, mode="same")
    f *= cov / f.std()
    return x, f


@lru_cache(maxsize=1)
def _albedo_fine_field():
    """The true fine-scale albedo perturbation field -- see
    `_correlated_field`. Seeded and `lru_cache`d so the scene is
    reproducible across processes (including `build_lookup_radiance`'s
    forked pool workers)."""
    return _correlated_field(ALBEDO_FINE_SEED)


#: A real fine-resolution external product (MODIS-like) knows the fine-
#: scale texture's own STATISTICS (ALBEDO_COV, ALBEDO_CORR_KM) but not the
#: true field's own phase -- it is an independent, imperfect measurement
#: of a correlated random field, not a second copy of the same noise draw.
#: Distinct seed (2026-08-28), same generation as `_albedo_fine_field`.
ALBEDO_PRIOR_FINE_SEED = 20260828


@lru_cache(maxsize=1)
def _albedo_prior_fine_field():
    """Like `_albedo_fine_field`, but an INDEPENDENT noise realization with
    the SAME statistics -- the sub-bin_modulation oracle for "a real,
    imperfect prior product" (docs/PROJECT_STATUS.md Sec.10's follow-up),
    as opposed to `albedo_for_label(..., include_fine=True)`'s exact-truth
    oracle. Genuinely correlated with the truth's own patch structure
    (same `_patch_type_weights` blend underneath), but the fine-scale
    texture riding on top of that blend is a different draw, so a bin's
    OWN true value need not equal `c * mean_g_k` for the corrected self-
    consistency identity (Sec.10) to hold exactly here -- unlike the g=truth
    run, this is a genuine test of whether sub_bin_modulation still helps
    when g is informative but imperfect, not an oracle that already knows
    the answer.
    """
    return _correlated_field(ALBEDO_PRIOR_FINE_SEED)


def _patch_type_weights(x_km):
    """Soft membership weight of each surface type at `x_km`.

    Patches are combined with tanh edges of width `ALBEDO_EDGE_KM` rather
    than a hard `searchsorted` pick, so a boundary is continuous (a real
    edge is not infinitely sharp, and a discontinuous truth would make the
    render's own PSF/ILS the only smoothing present).
    """
    x = np.atleast_1d(np.asarray(x_km, dtype=float))
    starts = np.array([p[0] for p in SURFACE_PATCHES])
    types = [p[1] for p in SURFACE_PATCHES]
    # weight of patch i = (turn on at its start) * (turn off at the next start)
    w = np.zeros((len(types), len(x)))
    for i, x0 in enumerate(starts):
        on = 0.5 * (1.0 + np.tanh((x - x0) / ALBEDO_EDGE_KM))
        if i + 1 < len(starts):
            off = 0.5 * (1.0 + np.tanh((starts[i + 1] - x) / ALBEDO_EDGE_KM))
        else:
            off = np.ones_like(x)
        w[i] = on * off
    total = w.sum(axis=0)
    total[total <= 0] = 1.0
    return types, w / total


def _albedo_compose(x, labels, pert):
    """Shared by every `albedo_at*` variant: patch-archetype blend times a
    fine-scale perturbation `pert` (already evaluated at `x`, 1.0 for
    "no fine texture"). Factored out (2026-08-28) so a caller can swap in
    a DIFFERENT fine-scale field (the true one, none, or an independent
    "prior product" realization -- `_albedo_prior_fine_field`) without
    duplicating the patch-blend logic three times.
    """
    from .scene import _BAND_ALBEDO

    types, w = _patch_type_weights(x)
    out = np.empty((len(labels), len(x)))
    for j, lab in enumerate(labels):
        try:
            table = _BAND_ALBEDO[lab]
        except KeyError:
            raise KeyError(f"no albedo defined for band {lab!r}; "
                           f"add it to geocarb_gert.scene._BAND_ALBEDO") from None
        base = np.zeros(len(x))
        for i, t in enumerate(types):
            base += w[i] * table[t]
        out[j] = np.clip(base * pert, _ALBEDO_MIN, _ALBEDO_MAX)
    return out


def albedo_at(x_km, labels, include_fine: bool = True):
    """Surface albedo at along-slit position(s) `x_km`, for bands `labels`.

    `labels` are `SpectralWindow.label` strings (e.g. "CO2_strong"), so the
    result is per-band and mutually consistent across bands: the SAME
    land-cover patch and the SAME fine-scale brightness perturbation drive
    every band, only the archetype reflectance differs. That is the
    physically meaningful coupling -- a forest patch is dark at 2.06 um and
    bright at 0.76 um -- and it is what makes this scene usable for the
    multi-FPA joint retrievals of `JOINT_BLOCK_MIGRATION_PLAN.md` Sec.5,
    where per-bin albedo has to be retrieved per band.

    `include_fine=False` (2026-08-25) drops the fine-scale texture
    perturbation, leaving just the patch-archetype blend -- the surface-side
    analogue of `STATE_FIELDS_PRIOR`'s "background/topography, no localized
    detail" imperfect prior. See `SURFACE_FIELDS_PRIOR`/`albedo_for_label_prior`.

    Returns
    -------
    ndarray, shape (n_labels,) for scalar `x_km`, else (n_labels, n_x)
    """
    scalar = np.isscalar(x_km) or np.asarray(x_km).ndim == 0
    x = np.atleast_1d(np.asarray(x_km, dtype=float))

    if include_fine:
        xf, f = _albedo_fine_field()
        pert = 1.0 + np.interp(x, xf, f)
    else:
        pert = 1.0

    out = _albedo_compose(x, labels, pert)
    return out[:, 0] if scalar else out


def albedo_at_prior_fine(x_km, labels):
    """Like `albedo_at(..., include_fine=True)`, but the fine-scale texture
    is `_albedo_prior_fine_field`'s INDEPENDENT realization, not the true
    field's own -- "a real, imperfect external product" oracle for
    `sub_bin_modulation`, as opposed to the exact-truth oracle `albedo_at`
    itself supplies. See `_albedo_prior_fine_field`'s own docstring.
    """
    scalar = np.isscalar(x_km) or np.asarray(x_km).ndim == 0
    x = np.atleast_1d(np.asarray(x_km, dtype=float))
    xf, f = _albedo_prior_fine_field()
    pert = 1.0 + np.interp(x, xf, f)
    out = _albedo_compose(x, labels, pert)
    return out[:, 0] if scalar else out


def albedo_for_label(x_km, label, include_fine: bool = True):
    """`albedo_at` for a single band, returning a plain scalar/1-D array.

    The shape `albedo_at` returns is ``(n_labels, n_x)``, which is right for
    a multi-band caller but awkward for a state row, whose prior is one
    number per bin centre in one band. Kept as a thin wrapper rather than
    changing `albedo_at`, since the multi-band shape is what
    `JOINT_BLOCK_MIGRATION_PLAN.md` Sec.5's per-band albedo retrieval needs.
    """
    scalar = np.isscalar(x_km) or np.asarray(x_km).ndim == 0
    out = np.atleast_2d(albedo_at(np.atleast_1d(x_km), [label], include_fine))[0]
    return float(out[0]) if scalar else out


def albedo_for_label_prior(x_km, label):
    """The imperfect ("structural") surface prior: `albedo_for_label` with
    the fine-scale texture dropped -- the prior knows the large-scale
    land-cover patch layout but not the within-patch brightness variation
    (ecosystem health, darker/lighter grasses, ...). See
    `SURFACE_FIELDS_PRIOR`/`SURFACE_PRIOR_FIELD_SETS` below.
    """
    return albedo_for_label(x_km, label, include_fine=False)


def albedo_for_label_prior_fine(x_km, label):
    """`albedo_at_prior_fine` for a single band -- the `sub_bin_modulation`
    "real, imperfect product" oracle (2026-08-28), as opposed to
    `albedo_for_label(..., include_fine=True)`'s exact-truth oracle used
    for docs/PROJECT_STATUS.md Sec.10's first result. NOT the same thing
    as `albedo_for_label_prior` above -- that one has NO fine-scale texture
    at all (`include_fine=False`); this one has real, resolved sub-bin
    texture, just not the true field's own phase.
    """
    scalar = np.isscalar(x_km) or np.asarray(x_km).ndim == 0
    out = np.atleast_2d(albedo_at_prior_fine(np.atleast_1d(x_km), [label]))[0]
    return float(out[0]) if scalar else out


#: Truth-state parameters that are NOT part of `AtmosphericProfile` and so
#: cannot live in `STATE_FIELDS`: they are passed separately to
#: `ForwardModel.run`. Keyed by the argument name that call expects.
#:
#: Each entry takes ``(x_km, band_label)`` rather than just ``x_km``, because
#: surface reflectance is per-band in a way composition is not -- one
#: land-cover patch is dark at 2.06 um and bright at 0.76 um. That is exactly
#: the coupling that makes a shared surface across FPAs meaningful, and it is
#: why these could not simply be appended to `STATE_FIELDS`.
#:
#: RESOLVED (2026-08-25, was open 2026-08-18 -> 2026-08-25): the joint-block
#: sweep script now has its own `--vary-albedo` flag
#: (`scripts/gd_joint_block_whole_slit_sweep.py`), required (and validated)
#: whenever `"albedo"` is in `--free` -- see that script for the actual
#: production wiring. `gd_test._band_setup`'s own `vary_albedo` still
#: defaults to False for every OTHER caller, unaffected.
SURFACE_FIELDS = {
    "albedo": albedo_for_label,
}

#: Imperfect surface prior (2026-08-25) -- the surface-side sibling of
#: `STATE_FIELDS_PRIOR`: knows the large-scale patch layout, not the
#: fine-scale texture. Named `SURFACE_PRIOR_FIELD_SETS` (not folded into
#: `PRIOR_FIELD_SETS` itself) because surface fields take `(x_km,
#: band_label)` while atmosphere fields take just `(x_km)` -- see
#: `SURFACE_FIELDS`'s own docstring above. `joint_state.state_spec_from_
#: scene`'s `surface_fields` parameter reads from here, keyed by the SAME
#: `--prior-fields` value that selects `PRIOR_FIELD_SETS` for the
#: atmosphere rows -- one flag now drives both.
SURFACE_FIELDS_PRIOR = {
    "albedo": albedo_for_label_prior,
}

SURFACE_PRIOR_FIELD_SETS = {
    "exact": SURFACE_FIELDS,
    "structural": SURFACE_FIELDS_PRIOR,
}


def xco2_ppm(x_km):
    background = XCO2_BG_PPM + 2.0 * np.sin(2 * np.pi * (x_km + 1400) / 3200)
    plume = _gauss(x_km, x0=-500.0, width=60.0, amp=6.0)
    hotspots = sum(_gauss(x_km, x0, w, amp) for x0, w, amp in HOTSPOTS_CO2)
    return background + plume + hotspots


def xco_ppb(x_km):
    background = XCO_BG_PPB + 10.0 * np.sin(2 * np.pi * (x_km + 1400) / 2600 + 1.0)
    plume = _gauss(x_km, x0=-500.0, width=55.0, amp=160.0)   # co-located with the CO2 plume
    hotspots = sum(_gauss(x_km, x0, w, amp) for x0, w, amp in HOTSPOTS_CO)
    return background + plume + hotspots


def xch4_ppb(x_km):
    background = XCH4_BG_PPB + 15.0 * np.sin(2 * np.pi * (x_km + 1400) / 2200 + 2.5)
    plume = _gauss(x_km, x0=650.0, width=100.0, amp=45.0)   # separate location from CO2/CO
    hotspots = sum(_gauss(x_km, x0, w, amp) for x0, w, amp in HOTSPOTS_CH4)
    return background + plume + hotspots


def h2o_surface_vmr(x_km):
    # smooth arid -> humid climatological gradient across the whole transect
    t = 0.5 * (1.0 + np.tanh(x_km / 500.0))
    return 0.005 + t * (0.018 - 0.005)


def p_surface_hpa(x_km):
    # `pressure_to_alt_std_atm` (model_sampler.py) only extrapolates for
    # p <= standard sea level (1013.25 hPa) -- it returns NaN above that
    # (found 2026-07-24: a symmetric +/-3 hPa sinusoid here briefly exceeded
    # 1013.25 and silently corrupted T_levels/h2o downstream). Kept
    # asymmetric (oscillates *below* the offset background only) so the
    # background term can never push p_surface above standard sea level,
    # regardless of x_km -- and offset by P_HEADROOM_HPA rather than hugging
    # the ceiling, so a RETRIEVED p_surface has somewhere to go too. See
    # P_HEADROOM_HPA for why 0.1 hPa was not enough once p_surface went free.
    background = ((P_BG_HPA - P_HEADROOM_HPA)
                  - 3.0 * (1.0 + np.sin(2 * np.pi * (x_km + 1400) / 2800 + 0.5)))
    mountain = _gauss(x_km, x0=-250.0, width=140.0, amp=-250.0)   # topographic depression
    return background + mountain


# -- "structural" priors for the realistic-prior experiment class --
#
# A real L2 prior is not the truth: it knows large-scale/climatological
# structure and static, independently-known topography, but not localized,
# unmodeled features (plumes, point-source hot spots, today's synoptic
# weather). These mirror the truth functions above, keeping only the
# background term (plus, for p_surface, the topographic `mountain` term via
# a hypsometric-style adjustment) and dropping everything else. Each
# reproduces its sibling's `background` formula exactly -- a change to
# XCO2_BG_PPM, an amplitude, a period, etc. above needs the matching edit
# here too.

def xco2_ppm_prior(x_km):
    return XCO2_BG_PPM + 2.0 * np.sin(2 * np.pi * (x_km + 1400) / 3200)


def xch4_ppb_prior(x_km):
    return XCH4_BG_PPB + 15.0 * np.sin(2 * np.pi * (x_km + 1400) / 2200 + 2.5)


def xco_ppb_prior(x_km):
    return XCO_BG_PPB + 10.0 * np.sin(2 * np.pi * (x_km + 1400) / 2600 + 1.0)


def p_surface_hpa_prior(x_km):
    # Topography (the `mountain` term) is static and knowable via a
    # hypsometric adjustment, so it's kept exactly. The synoptic sinusoid is
    # day-to-day weather a static prior would not have, so it's replaced by
    # its own midline (the sinusoid averages to (P_BG_HPA - P_HEADROOM_HPA)
    # - 3.0 over a full period).
    background_ref = P_BG_HPA - P_HEADROOM_HPA - 3.0
    mountain = _gauss(x_km, x0=-250.0, width=140.0, amp=-250.0)
    return np.full_like(np.asarray(x_km, dtype=float), background_ref) + mountain


# The canonical truth-state parameter table: one row per along-slit-varying
# quantity, keyed by the exact keyword `atmosphere_from_params` takes. Every
# consumer (truth generation, state interpolation, forward-model anchors)
# iterates this rather than naming parameters individually, so no quantity is
# privileged -- CO2 is just another row -- and adding one is a single edit
# here instead of a new branch at every call site.
#
# Not yet included: surface albedo (`albedo_at`), because it is not part of
# `AtmosphericProfile` -- it is passed separately to `ForwardModel.run`. It
# belongs in the same table once the forward models take a combined
# state+surface record; see JOINT_BLOCK_MIGRATION_PLAN.md Sec.4.1/Sec.5,
# where albedo also has to become per-bin AND per-FPA.
STATE_FIELDS = {
    "co2_ppm": lambda x: xco2_ppm(x),
    "ch4_ppb": lambda x: xch4_ppb(x),
    "co_ppb": lambda x: xco_ppb(x),
    "h2o_surface_vmr": lambda x: h2o_surface_vmr(x),
    "p_surface_hpa": lambda x: p_surface_hpa(x),
}

# Same shape as STATE_FIELDS, but each field is the "structural" prior
# defined above -- background/topography-aware, never the localized plume/
# hot-spot/synoptic content. h2o has no such content to begin with (a single
# smooth climatological gradient), so it's unchanged from STATE_FIELDS.
STATE_FIELDS_PRIOR = {
    "co2_ppm": lambda x: xco2_ppm_prior(x),
    "ch4_ppb": lambda x: xch4_ppb_prior(x),
    "co_ppb": lambda x: xco_ppb_prior(x),
    "h2o_surface_vmr": lambda x: h2o_surface_vmr(x),
    "p_surface_hpa": lambda x: p_surface_hpa_prior(x),
}

# -- uniform multiplicative-bias priors (2026-08-20) --
#
# A different failure mode from STATE_FIELDS_PRIOR's "missing localized
# structure": here the prior keeps 100% of the truth's own spatial detail
# (background + plume + hot spots for CO2, background + topography for
# p_surface -- every term STATE_FIELDS has) and is wrong only in its
# absolute level, by a constant factor everywhere along the slit. This
# isolates "prior committed to the wrong absolute value" from "prior
# missing spatial detail" -- PROJECT_STATUS.md Sec.5 Phase 2's two
# distinct imperfect-prior questions -- rather than conflating both into
# one experiment. First pair (2026-08-20): CO2 +1% alone, then CO2 +1%
# together with p_surface -1%, to see whether a second simultaneously
# mis-set row changes how much g_ratio/anchor_density resolution the
# solve needs, the same conditioning question Phase 1's co2/co2p pair
# asked of representation error.
STATE_FIELDS_PRIOR_CO2_PLUS1PCT = {
    "co2_ppm": lambda x: 1.01 * xco2_ppm(x),
    "ch4_ppb": lambda x: xch4_ppb(x),
    "co_ppb": lambda x: xco_ppb(x),
    "h2o_surface_vmr": lambda x: h2o_surface_vmr(x),
    "p_surface_hpa": lambda x: p_surface_hpa(x),
}

STATE_FIELDS_PRIOR_CO2_PLUS1PCT_PSURF_MINUS1PCT = {
    "co2_ppm": lambda x: 1.01 * xco2_ppm(x),
    "ch4_ppb": lambda x: xch4_ppb(x),
    "co_ppb": lambda x: xco_ppb(x),
    "h2o_surface_vmr": lambda x: h2o_surface_vmr(x),
    "p_surface_hpa": lambda x: 0.99 * p_surface_hpa(x),
}

#: Standardized registry every prior-selecting call site reads from by
#: NAME, instead of each caller wiring its own boolean/enum for one prior
#: at a time (the pattern this replaces: `gd_joint_block_whole_slit_
#: sweep.py`'s old `--realistic-prior` boolean, which could only ever
#: mean "STATE_FIELDS_PRIOR or nothing"). "exact" (prior=truth) and
#: "structural" (the old --realistic-prior=True case) are included so
#: every prior configuration -- old and new -- goes through this same
#: lookup; adding a new imperfect prior means adding one entry here, not
#: a new flag at every call site.
PRIOR_FIELD_SETS = {
    "exact": STATE_FIELDS,
    "structural": STATE_FIELDS_PRIOR,
    "co2_plus1pct": STATE_FIELDS_PRIOR_CO2_PLUS1PCT,
    "co2_plus1pct_psurf_minus1pct": STATE_FIELDS_PRIOR_CO2_PLUS1PCT_PSURF_MINUS1PCT,
}


def atmosphere_at(x_km: float, h2o_scale_height_km: float = 2.0, fields=None) -> AtmosphericProfile:
    """The true ``AtmosphericProfile`` at one along-slit position [km] --
    i.e. this function *is* "the scene," in the physical sense
    (``geocarb_gert.focalplane``'s module docstring has the full
    terminology note: "scene" here means atmospheric state, not radiance).

    To get radiance from a state returned here, run it through
    ``gert.forward_model.ForwardModel`` -- radiance(x_km) = RT(atmosphere_at(x_km)).
    See ``scripts/gd_joint_block_retrieve.py``'s ``spectrum_for`` or
    ``scripts/gd_test.py``'s ``_band_setup`` for the concrete pattern. No
    function in this module performs that RT step itself; it only ever
    returns the state.

    Generalizes ``geocarb_gert.scene.reference_atmosphere`` to also vary
    CH4, CO, and surface pressure (that function only overrides CO2/H2O).

    ``fields`` (default ``None`` -> :data:`STATE_FIELDS`, today's exact
    behaviour) forwards straight to :func:`state_at` -- lets a caller swap
    in e.g. :func:`resolution_matched_fields` so the TRUTH scene itself is
    band-limited, not just a retrieval's prior.
    """
    return atmosphere_from_params(
        **{k: float(v) for k, v in state_at(x_km, fields=fields).items()},
        h2o_scale_height_km=h2o_scale_height_km)


def state_at(x_km, fields=None) -> dict:
    """The full truth state at along-slit position(s) ``x_km``, as a plain
    ``{param_name: value}`` dict keyed by :data:`STATE_FIELDS`.

    One generic accessor over the parameter table rather than five separate
    calls, so no parameter is privileged over another and adding one means
    adding a table row, not a new branch at every call site. This is the
    same "per-element table, not per-group code" principle
    ``JOINT_BLOCK_MIGRATION_PLAN.md`` Sec.4.0 settled on for the
    regularizer, applied to the truth scene.

    Accepts scalar or array ``x_km``; values follow its shape.
    """
    fields = STATE_FIELDS if fields is None else fields
    return {name: fn(x_km) for name, fn in fields.items()}


def interp_state(x_target, x_known, fields=None) -> dict:
    """Every state parameter linearly interpolated from ``x_known`` onto
    ``x_target``, as a :func:`state_at`-shaped dict.

    The state-space interpolation :func:`geocarb_gert.nearest_bin_scene`'s
    docstring prescribes ("the correct place to interpolate is the *state*
    ... followed by a fresh RT run"). Every parameter goes through the
    identical path -- CO2 included, with no special case -- so a forward
    model built on this cannot accidentally represent its retrieved quantity
    more accurately than the quantities it holds fixed, which is exactly the
    asymmetry that made surface pressure and H2O dominate the joint block's
    post-fit residual.
    """
    fields = STATE_FIELDS if fields is None else fields
    x_known = np.asarray(x_known, dtype=float)
    return {name: np.interp(x_target, x_known, fn(x_known))
            for name, fn in fields.items()}


def resolution_matched_fields(x_anchor_km, fields=None) -> dict:
    """``{name: fn(x_km)}`` -- each ``fn`` is the piecewise-linear
    interpolant of the TRUE field, sampled once at ``x_anchor_km`` -- i.e.
    a version of the truth with literally ZERO structure below that
    anchor spacing, exposed as reusable callables in the same shape
    :data:`STATE_FIELDS`/:data:`PRIOR_FIELD_SETS` entries already use (so
    it drops into any ``fields=`` parameter, including :func:`atmosphere_at`
    /:func:`build_lookup_radiance`, unchanged).

    2026-08-29 (user): built so a truth IMAGE can be rendered from a scene
    the anchor grid at a given ``anchor_density`` can, in principle,
    perfectly represent -- removing the "genuinely-unresolvable sub-anchor
    structure" confound from a retrieval-accuracy comparison entirely, so
    any remaining error is real retrieval/model error, not truth the
    anchor grid could never have captured.

    Same underlying math as :func:`interp_state` (which returns a one-shot
    dict of VALUES at ``x_target``); this returns reusable FUNCTIONS
    instead, each closing over its own ``x_anchor_km``/sampled-values pair.
    """
    fields = STATE_FIELDS if fields is None else fields
    x_anchor_km = np.asarray(x_anchor_km, dtype=float)
    known = {name: np.asarray(fn(x_anchor_km), dtype=float) for name, fn in fields.items()}
    return {name: (lambda x_km, xk=x_anchor_km, yk=known[name]: np.interp(x_km, xk, yk))
           for name in fields}


def resolution_matched_albedo_fn(x_anchor_km, label) -> Callable:
    """The surface-side sibling of :func:`resolution_matched_fields`: a
    single band's albedo, band-limited to ``x_anchor_km``. Reuses
    :func:`albedo_for_label`'s own existing patch-blend + fine-texture
    composition (`include_fine=True`) to get the anchor-point VALUES, then
    only band-limits the RESULT via linear interpolation -- no need to
    duplicate `_albedo_compose`'s own logic.
    """
    x_anchor_km = np.asarray(x_anchor_km, dtype=float)
    y_anchor = albedo_for_label(x_anchor_km, label, include_fine=True)
    return lambda x_km: np.interp(x_km, x_anchor_km, y_anchor)


def build_scene_fields(uniform: bool = False, barcode: bool = False,
                       realistic_barcode: bool = False, vary_albedo: bool = False,
                       barcode_bars: int = 10, band_labels=None,
                       constant_albedo: float = 0.0,
                       fields=None, surface_fields=None):
    """THE one place every truth-scene "mode" gets defined (2026-09-04,
    user: "barcode and realistic barcode are just different sets of scene
    parameters") -- `--uniform`/`--barcode`/`--realistic-barcode`/
    `--vary-albedo` all collapse to a particular choice of ``fields``
    (``{atmosphere_row: fn(x_km)}``, the same shape :data:`STATE_FIELDS`/
    :func:`resolution_matched_fields` already use) and ``surface_fields``
    (``{band_label: fn(x_km)}``, matching :data:`SURFACE_FIELDS`/
    :func:`resolution_matched_albedo_fn`'s own convention -- NOT
    `state_spec_from_scene`'s `{"albedo": fn(x_km, label)}`, since this
    feeds straight into `build_lookup_radiance`/the anchor-rendering
    path, not a `StateSpec`), fed into ONE rendering mechanism -- barcode
    stops being a separate code path.

    Real behavioural fix, not just a refactor: today's ``--barcode``
    computes ONE spectrum at the center atmosphere/albedo and multiplies
    a bar-pattern GAIN onto it post-hoc (linear-in-albedo approximation).
    Expressed as a real ``surface_fields`` albedo pattern instead, each
    bar gets a genuine fresh RT run at its own true albedo -- the same
    "never approximate what a real RT call can compute exactly" principle
    :func:`geocarb_gert.focalplane.footprint_average_scene` already
    applies to spectral interpolation. This changes ``--barcode``'s own
    numerical output (confirmed intentional, not a regression).

    ``fields``/``surface_fields`` (both default ``None``): an explicit
    caller override -- e.g. :func:`resolution_matched_fields`/
    :func:`resolution_matched_albedo_fn` for a deliberately band-limited
    truth. Given non-``None``, this function's own uniform/barcode/
    vary_albedo logic is skipped entirely and the override is returned
    as-is -- matches how ``fields=``/``surface_fields=`` already worked
    on every caller before this refactor.

    Parameters
    ----------
    band_labels : list of str
        Every band label a caller might ever query the returned
        ``surface_fields`` for (usually ``[[label]]`` for a single-FPA
        caller) -- REQUIRED whenever albedo needs to vary at all
        (``barcode``, ``realistic_barcode``, or ``vary_albedo``); a
        constant-albedo scene (``uniform``, or none of those three) never
        needs it, so it stays optional for that case.
    constant_albedo : float
        The scalar albedo used wherever nothing else makes it vary --
        `uniform`, `barcode`'s own base reflectance, or the non-`vary_
        albedo` fallback. Caller-supplied (e.g. `_band_setup`'s own
        ``albedo_for(wide_inst, "desert")``) rather than computed here,
        so this function never silently substitutes a different albedo
        SOURCE than whatever the caller was already using.
    """
    if fields is not None or surface_fields is not None:
        return fields, surface_fields

    def _const_field(v):
        return lambda x_km: np.full(np.atleast_1d(x_km).shape, float(v))

    if uniform:
        fields_out = {name: _const_field(fn(0.0)) for name, fn in STATE_FIELDS.items()}
    else:
        fields_out = dict(STATE_FIELDS)

    needs_albedo_pattern = barcode or realistic_barcode or vary_albedo
    if needs_albedo_pattern and not band_labels:
        raise ValueError("band_labels is required whenever albedo needs to vary "
                         "(barcode, realistic_barcode, or vary_albedo)")

    if barcode or realistic_barcode:
        n_bars = barcode_bars
        brightness = np.resize([1.0, 0.2], n_bars)
        # sharp bar edges (softness=0), same convention as focalplane.
        # barcode_scene's own default -- boundaries in ETA, converted to
        # x_km once per call below.
        boundaries_eta = -1.0 + np.cumsum(np.full(n_bars, 2.0 / n_bars))[:-1]

        def _bar_gain(x_km):
            eta = np.atleast_1d(x_km) / SLIT_HALF_KM
            g = np.full(eta.shape, brightness[0])
            for k, b in enumerate(boundaries_eta):
                g = np.where(eta > b, brightness[k + 1], g)
            return g

        if barcode:
            # Atmosphere ALSO fixed at center (matches --barcode's own
            # existing semantics: "same column of sky everywhere, only
            # reflectance varies" -- a ground-test diffuser illumination
            # pattern, not a genuine along-slit atmosphere).
            fields_out = {name: _const_field(fn(0.0)) for name, fn in STATE_FIELDS.items()}
            def _base_albedo(x_km):
                return np.full(np.atleast_1d(x_km).shape, constant_albedo)
        else:
            # realistic_barcode: real continuous atmosphere (fields_out
            # already set above); the bar gain multiplies WHATEVER the
            # base albedo would otherwise have been -- the real vary_
            # albedo pattern if that's also on, else the fixed constant.
            if vary_albedo:
                def _base_albedo(x_km, _lab=band_labels[0]):
                    return albedo_for_label(x_km, _lab)
            else:
                def _base_albedo(x_km):
                    return np.full(np.atleast_1d(x_km).shape, constant_albedo)

        def _albedo_fn(x_km, _base=_base_albedo):
            return _bar_gain(x_km) * np.asarray(_base(x_km), dtype=float)

        surface_fields_out = {lab: _albedo_fn for lab in band_labels}
    elif vary_albedo:
        surface_fields_out = {lab: (lambda x_km, _l=lab: albedo_for_label(x_km, _l))
                              for lab in band_labels}
    else:
        surface_fields_out = None  # every existing caller's own default (a single fixed scalar)

    return fields_out, surface_fields_out


def atmosphere_from_params(co2_ppm: float, ch4_ppb: float, co_ppb: float,
                           h2o_surface_vmr: float, p_surface_hpa: float,
                           h2o_scale_height_km: float = 2.0) -> AtmosphericProfile:
    """Build the truth ``AtmosphericProfile`` from explicit scalar state
    parameters, instead of from an along-slit position.

    Factored out of :func:`atmosphere_at` (2026-08-17) so the two share one
    code path exactly -- ``atmosphere_at(x)`` is now literally this function
    called on the analytic profiles evaluated at ``x``. That matters because
    it lets a forward model build an anchor's atmosphere from *interpolated*
    state parameters (the state-space interpolation
    :func:`geocarb_gert.nearest_bin_scene`'s own docstring prescribes:
    "the correct place to interpolate is the *state* ... followed by a fresh
    RT run") with a guarantee that nothing else about the profile
    construction differs from the truth generator.
    """
    # pure sigma, NOT gert_levels: see geocarb_gert.levels for why -- it is
    # what makes d(p_level)/d(p_sfc) exactly parallel to p, hence an exact
    # analytic surface-pressure Jacobian.
    p = np.asarray(sigma_levels(float(p_surface_hpa) * 100.0), dtype=float)  # TOA -> surface
    z_km = np.asarray(pressure_to_alt_std_atm(p / 100.0), dtype=float)
    T = _std_temperature(z_km)

    h2o = float(h2o_surface_vmr) * np.exp(-z_km / float(h2o_scale_height_km))
    gases = {g: np.full_like(p, v) for g, v in _WELL_MIXED.items()}
    gases["co2"] = np.full_like(p, float(co2_ppm) * 1e-6)
    gases["ch4"] = np.full_like(p, float(ch4_ppb) * 1e-9)
    gases["co"] = np.full_like(p, float(co_ppb) * 1e-9)
    gases["h2o"] = h2o

    w = _MW_RATIO * h2o
    q = w / (1.0 + w)
    return AtmosphericProfile(p_levels=p, T_levels=T, q_levels=q, gases=gases)


# -- globals populated in build_lookup_radiance() before the Pool is forked --
_G_LOOKUP = {}


def _lookup_sample(i: int):
    g = _G_LOOKUP
    x_km = g["x_samples_km"][i]
    atm = atmosphere_at(x_km, g["h2o_scale_height_km"], fields=g.get("fields"))
    fm = g["fm_cls"](atm, g["absco"], g["inst"], g["geo"],
                     solver=g["solver_cls"](), solar_spectrum=g["solar"])
    # Albedo is applied INSIDE the forward run (not multiplied onto the
    # spectrum afterwards), so a per-sample albedo genuinely re-runs the RT
    # at that reflectance rather than rescaling one shared spectrum.
    surface_fields = g.get("surface_fields")
    if surface_fields is not None:
        # Resolution-matched (or otherwise custom) per-band albedo, same
        # list shape [one value per band_labels entry] the raw albedo_at
        # call below already produces -- see resolution_matched_albedo_fn.
        albedo = [float(surface_fields[lab](x_km)) for lab in g["band_labels"]]
    elif g["vary_albedo"]:
        albedo = list(albedo_at(x_km, g["band_labels"]))
    else:
        albedo = g["albedo"]
    res = fm.run(albedo=albedo, albedo_slope=[0.0] * len(albedo))
    return i, res.I_hires[0]


def build_lookup_radiance(
    absco, inst, geo, solar, albedo,
    n_samples: int = 400,
    h2o_scale_height_km: float = 2.0,
    n_workers: int | None = None,
    uniform: bool = False,
    vary_albedo: bool = False,
    fields=None,
    surface_fields=None,
) -> tuple[np.ndarray, Callable]:
    """Precompute hi-res spectra at ``n_samples`` along-slit positions and
    return ``(wn_hires, radiance)`` where ``radiance(eta) -> spectrum`` does
    linear interpolation between the two nearest precomputed samples.

    The along-slit hot spots are ~10 km wide; with the default
    ``n_samples=400`` over the 2800 km slit, sample spacing is 7 km --
    roughly 3 samples across each hot spot's FWHM, resolving it without
    excessive forward-model cost. Increase ``n_samples`` for finer features.

    Each sample needs its own ``ForwardModel`` (a fresh one per sample
    atmosphere) -- embarrassingly parallel like ``gd_render.image()``, same
    fork + copy-on-write pattern so ``absco``/``solar`` aren't duplicated
    per worker.

    ``uniform=True`` collapses this to a single sample at the slit centre
    (x_km=0, matching the retrieval's prior atmosphere exactly): every row
    then sees the identical truth spectrum, so ``radiance(eta)`` is constant
    along the slit regardless of ``eta``. This isolates pure geometric-
    distortion bias (keystone/smile/PSF/rectification-interpolation) from
    composition/pressure-tracking bias, for direct comparison against a
    ``uniform=False`` run of the same band and against the older, uniform-
    composition dense-sweep design (``gd_dense_sweep.py``). Added
    2026-07-24 for the FPA2 with/without along-slit-variation comparison.

    ``fields``/``surface_fields`` (2026-08-29, both default ``None`` ->
    today's exact behaviour: raw :data:`STATE_FIELDS` and ``albedo_at``)
    let a caller render the TRUTH itself from custom field functions --
    e.g. :func:`resolution_matched_fields`/:func:`resolution_matched_
    albedo_fn`, so the rendered image reflects a deliberately band-limited
    scene rather than the full continuous truth. ``surface_fields`` is
    ``{band_label: fn(x_km) -> albedo}``, one entry per ``inst.windows``
    label; only consulted when ``vary_albedo=True`` (same as ``albedo_at``
    itself).

    .. deprecated:: 2026-09-04
        This two-sample linear spectral blend was confirmed (2026-09-02)
        to be the dominant source of the representability-gap artifacts
        an entire investigation chased before tracing it here -- see
        docs/PROJECT_STATUS.md Sec.13.1. New code should use
        :func:`geocarb_gert.joint_state.render_at_anchors` +
        :func:`geocarb_gert.focalplane.footprint_average_scene` instead
        (real per-anchor RT, exact sub-pixel footprint integration, never
        an interpolated stand-in). Kept for the several older scripts
        that still call it directly (``gd_toy_native_row_demo.py``,
        ``gd_reversed_scene_check.py``, and others) -- not removed, since
        migrating those is a separate, larger task.
    """
    warnings.warn(
        "build_lookup_radiance uses a two-sample linear spectral blend, "
        "confirmed to be the dominant source of representability-gap "
        "artifacts in prior investigations (docs/PROJECT_STATUS.md "
        "Sec.13.1). New code should use "
        "geocarb_gert.joint_state.render_at_anchors + "
        "geocarb_gert.focalplane.footprint_average_scene instead.",
        DeprecationWarning, stacklevel=2)
    from gert.forward_model import ForwardModel
    from gert.rt_solver import SingleScatterSolver

    if vary_albedo and uniform:
        # `uniform` collapses to a single sample at x_km=0, so albedo would
        # be frozen at whatever value happens to sit at the slit centre --
        # silently NOT a varying-albedo scene. Refuse rather than mislead.
        raise ValueError("vary_albedo=True is meaningless with uniform=True "
                         "(uniform collapses the scene to one sample at x_km=0)")

    if uniform:
        x_samples_km = np.zeros(1)
        n_samples = 1
    else:
        x_samples_km = np.linspace(-SLIT_HALF_KM, SLIT_HALF_KM, n_samples)

    if n_workers is None:
        n_workers = available_cpus()
    if mp.current_process().daemon:
        n_workers = 1

    _G_LOOKUP.update(dict(x_samples_km=x_samples_km, absco=absco, inst=inst, geo=geo,
                          solar=solar, albedo=list(albedo), h2o_scale_height_km=h2o_scale_height_km,
                          fm_cls=ForwardModel, solver_cls=SingleScatterSolver,
                          vary_albedo=bool(vary_albedo),
                          band_labels=[w.label for w in inst.windows],
                          fields=fields, surface_fields=surface_fields))

    # one throwaway call to get wn_hires (identical for every sample -- a
    # property of the Instrument/SpectralWindow, not the atmosphere)
    _, spec0 = _lookup_sample(0)
    n_hires = len(spec0)
    spectra = np.empty((n_samples, n_hires), dtype=float)
    spectra[0] = spec0

    remaining = range(1, n_samples)
    if n_workers <= 1:
        for i in remaining:
            _, spec = _lookup_sample(i)
            spectra[i] = spec
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for i, spec in pool.imap_unordered(_lookup_sample, remaining, chunksize=4):
                spectra[i] = spec

    wn_hires = inst.windows[0].wn_hires   # atmosphere-independent; no extra forward call needed

    def radiance(eta):
        eta = np.atleast_1d(np.asarray(eta, dtype=float))
        if n_samples == 1:   # uniform=True -- single sample, no interpolation needed
            return np.broadcast_to(spectra[0], (len(eta), spectra.shape[1]))
        x_km = eta * SLIT_HALF_KM
        idx_hi = np.clip(np.searchsorted(x_samples_km, x_km), 1, n_samples - 1)
        idx_lo = idx_hi - 1
        x_lo, x_hi = x_samples_km[idx_lo], x_samples_km[idx_hi]
        w_hi = np.clip((x_km - x_lo) / (x_hi - x_lo), 0.0, 1.0)
        w_lo = 1.0 - w_hi
        return w_lo[:, None] * spectra[idx_lo] + w_hi[:, None] * spectra[idx_hi]

    return wn_hires, radiance
