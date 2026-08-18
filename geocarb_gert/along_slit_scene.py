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
(eta=-1..+1 -> -1400..+1400 km, ~2800 km total, matching this repo's
LongSlitGeoSatellite's slit_length_km=3000 default closely enough for a
stress-test truth scene).
"""
from __future__ import annotations

import multiprocessing as mp
from functools import lru_cache
from typing import Callable

import numpy as np

from gert.atmosphere import AtmosphericProfile
from .levels import sigma_levels

from model_sampler import pressure_to_alt_std_atm

from .scene import _MW_RATIO, _std_temperature, _WELL_MIXED
from .gd_render import available_cpus

SLIT_HALF_KM = 1400.0        # -> 2800 km total slit length

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
ALBEDO_CORR_KM = 30.0     # its correlation length -- shorter than any gas feature
ALBEDO_FINE_SEED = 20260817
_ALBEDO_MIN, _ALBEDO_MAX = 0.005, 0.95


def _gauss(x, x0, width, amp):
    return amp * np.exp(-0.5 * ((x - x0) / width) ** 2)


@lru_cache(maxsize=1)
def _albedo_fine_field():
    """Deterministic smooth fractional-perturbation field on a 1 km grid.

    Gaussian-smoothed white noise, renormalised so its own standard
    deviation is exactly `ALBEDO_COV` -- smoothing otherwise shrinks the
    variance by an amount that depends on `ALBEDO_CORR_KM`, which would
    make the requested coefficient of variation silently wrong.

    Seeded and `lru_cache`d rather than drawn per call, so every process
    (including `build_lookup_radiance`'s forked pool workers) sees the
    identical field and the scene stays reproducible.
    """
    x = np.arange(-SLIT_HALF_KM, SLIT_HALF_KM + 1.0, 1.0)
    rng = np.random.default_rng(ALBEDO_FINE_SEED)
    w = rng.standard_normal(len(x))
    # Gaussian kernel, truncated at 4 sigma
    half = int(4 * ALBEDO_CORR_KM)
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / ALBEDO_CORR_KM) ** 2)
    k /= k.sum()
    f = np.convolve(w, k, mode="same")
    f *= ALBEDO_COV / f.std()
    return x, f


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


def albedo_at(x_km, labels):
    """Surface albedo at along-slit position(s) `x_km`, for bands `labels`.

    `labels` are `SpectralWindow.label` strings (e.g. "CO2_strong"), so the
    result is per-band and mutually consistent across bands: the SAME
    land-cover patch and the SAME fine-scale brightness perturbation drive
    every band, only the archetype reflectance differs. That is the
    physically meaningful coupling -- a forest patch is dark at 2.06 um and
    bright at 0.76 um -- and it is what makes this scene usable for the
    multi-FPA joint retrievals of `JOINT_BLOCK_MIGRATION_PLAN.md` Sec.5,
    where per-bin albedo has to be retrieved per band.

    Returns
    -------
    ndarray, shape (n_labels,) for scalar `x_km`, else (n_labels, n_x)
    """
    from .scene import _BAND_ALBEDO

    scalar = np.isscalar(x_km) or np.asarray(x_km).ndim == 0
    x = np.atleast_1d(np.asarray(x_km, dtype=float))
    types, w = _patch_type_weights(x)

    xf, f = _albedo_fine_field()
    pert = 1.0 + np.interp(x, xf, f)

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
    return out[:, 0] if scalar else out


def albedo_for_label(x_km, label):
    """`albedo_at` for a single band, returning a plain scalar/1-D array.

    The shape `albedo_at` returns is ``(n_labels, n_x)``, which is right for
    a multi-band caller but awkward for a state row, whose prior is one
    number per bin centre in one band. Kept as a thin wrapper rather than
    changing `albedo_at`, since the multi-band shape is what
    `JOINT_BLOCK_MIGRATION_PLAN.md` Sec.5's per-band albedo retrieval needs.
    """
    scalar = np.isscalar(x_km) or np.asarray(x_km).ndim == 0
    out = np.atleast_2d(albedo_at(np.atleast_1d(x_km), [label]))[0]
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
#: CAVEAT (2026-08-18): the joint-block truth images rendered so far all use
#: a single CONSTANT albedo -- `gd_test._band_setup`'s `vary_albedo` defaults
#: to False and no joint-block caller passes it. So a free albedo row fitted
#: against those bands is fitting a constant, and the patch/fine-scale
#: structure below is present in this module but absent from the data. Render
#: with ``vary_albedo=True`` (and re-cache the band image) before reading any
#: albedo retrieval as a test of along-slit albedo recovery.
SURFACE_FIELDS = {
    "albedo": albedo_for_label,
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


def atmosphere_at(x_km: float, h2o_scale_height_km: float = 2.0) -> AtmosphericProfile:
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
    """
    return atmosphere_from_params(
        **{k: float(v) for k, v in state_at(x_km).items()},
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
    atm = atmosphere_at(x_km, g["h2o_scale_height_km"])
    fm = g["fm_cls"](atm, g["absco"], g["inst"], g["geo"],
                     solver=g["solver_cls"](), solar_spectrum=g["solar"])
    # Albedo is applied INSIDE the forward run (not multiplied onto the
    # spectrum afterwards), so a per-sample albedo genuinely re-runs the RT
    # at that reflectance rather than rescaling one shared spectrum.
    albedo = (list(albedo_at(x_km, g["band_labels"])) if g["vary_albedo"]
              else g["albedo"])
    res = fm.run(albedo=albedo, albedo_slope=[0.0] * len(albedo))
    return i, res.I_hires[0]


def build_lookup_radiance(
    absco, inst, geo, solar, albedo,
    n_samples: int = 400,
    h2o_scale_height_km: float = 2.0,
    n_workers: int | None = None,
    uniform: bool = False,
    vary_albedo: bool = False,
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
    """
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
                          band_labels=[w.label for w in inst.windows]))

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
