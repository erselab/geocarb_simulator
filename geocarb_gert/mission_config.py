"""Mission-side config: ``input/geocarb_instrument.yml`` and
``input/retrieval_defaults.yml`` -> typed config objects.

This is Phase A of the config-consolidation plan (see the repo's
``docs/PROJECT_STATUS.md`` and the approved implementation plan): a single
source of truth for instrument parameters (keystone/smile/clocking, bands,
spectral/spatial resolution, slit length, noise calibration) and retrieval
sweep defaults, replacing the scattered, sometimes-disagreeing module
constants this package and ``scripts/`` have used until now.

**This module is purely additive in Phase A.** Nothing in ``geocarb_gert``
or ``scripts/`` imports from it yet -- ``GEOCARB_BANDS``,
``RADIOMETRIC_SPEC_BY_FPA``, ``GEOCARB_REF``, ``SLIT_HALF_KM``,
``DEFAULT_CORR_LENGTH_ETA``, and the sweep script's ``MIN_WINDOW``/``PAD``/
``G_RATIO`` all stay hardcoded exactly as before until Phase B repoints them
at a ``GeoCarbInstrumentConfig``/``RetrievalDefaults`` loaded from the
checked-in YAML at import time. This phase only has to prove the loader
reproduces today's values exactly (see ``scripts/check_mission_config.py``).

Deliberately does **not** route the ``bands:`` block through
``gert.instrument_config.InstrumentConfig``: ``geocarb_gert.instrument.
build_geocarb_instrument`` already builds ``SpectralWindow``/``ILS``
directly from wavenumber-native ``(label, wn_min, wn_max, molecules,
resolving_power)`` tuples (working, in production use today) --
``InstrumentConfig``'s own ``WindowSpec`` instead wants wavelength-space
``fwhm_nm`` per window and has no ``fpa`` field, so forcing the config
through it would mean a real nm<->cm-1 conversion plus a cross-repo
``WindowSpec`` extension, for a code path nothing currently calls. Kept the
existing, working conversion in ``instrument.py`` instead; this module just
supplies its inputs from YAML.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from .focalplane import FocalPlaneModel
from .paths import REPO_ROOT, config_root


# ── geocarb_instrument.yml ──────────────────────────────────────────────────

@dataclass
class MeasuredFocalPlaneSpec:
    """``focal_plane.measured`` -- metadata around the real, ground-test GD
    polynomial coefficients (keystone/smile/clocking stay implicit in the
    fit, not free scalars)."""
    gd_csv_path: Path
    n_fpa: int
    n_px: int
    dispersion_ascending: dict
    spatial_psf_fwhm_px: float


@dataclass
class FocalPlaneSpec:
    """``focal_plane`` mode switch: ``measured`` (real GD-polynomial data,
    what the production truth-rendering pipeline uses) or ``analytic``
    (free ``FocalPlaneModel`` parameters per FPA, for idealized/design-study
    scenes)."""
    mode: str
    measured: Optional[MeasuredFocalPlaneSpec]
    analytic: Optional[dict]  # {fpa: FocalPlaneModel}

    def __post_init__(self):
        if self.mode not in ("measured", "analytic"):
            raise ValueError(f"focal_plane.mode must be 'measured' or 'analytic', got {self.mode!r}")
        if self.mode == "measured" and self.measured is None:
            raise ValueError("focal_plane.mode == 'measured' but no 'measured:' block given")
        if self.mode == "analytic" and self.analytic is None:
            raise ValueError("focal_plane.mode == 'analytic' but no 'analytic:' block given")


@dataclass
class GeometrySpec:
    """Single source of truth for slit length / GSD / integration time /
    altitude -- resolves the along_slit_scene.SLIT_HALF_KM vs
    geosat_geometry.py vs radiometry.GEOCARB_REF triple-duplication."""
    slit_length_km: float
    pixel_size_ew_km: float
    pixel_size_ns_km: float
    integration_time_s: float
    sat_alt_km: float
    sat_lon_deg: float

    @property
    def slit_half_km(self) -> float:
        return self.slit_length_km / 2.0


@dataclass
class GeoCarbInstrumentConfig:
    """Loaded from ``geocarb_instrument.yml``. ``bands`` is
    ``GEOCARB_BANDS``-shaped (``(label, wn_min, wn_max, molecules,
    resolving_power)`` tuples, sorted by ``fpa``); ``band_fpa`` is the
    ``{label: fpa}`` sidecar dict recovering the index that shape drops."""
    name: str
    bands: list
    band_fpa: dict
    channels_per_fwhm: int
    hires_spacing: float
    focal_plane: FocalPlaneSpec
    geometry: GeometrySpec
    noise_by_fpa: dict

    def __post_init__(self):
        if self.focal_plane.mode == "measured":
            m = self.focal_plane.measured
            if m.n_fpa != len(self.bands):
                raise ValueError(f"focal_plane.measured.n_fpa={m.n_fpa} does not match "
                                 f"{len(self.bands)} bands in this config")
            # Structural facts about the GD-polynomial CSV format (4 FPAs,
            # 1024x1024) -- gd_polynomials.py keeps its own N_FPA/N_PX
            # constants rather than reading them from config (they aren't
            # free parameters, see this module's docstring), so this just
            # confirms the two don't drift, catching a stale/wrong CSV path
            # early rather than deep inside a later render call.
            from .gd_polynomials import N_FPA, N_PX
            if m.n_fpa != N_FPA:
                raise ValueError(f"focal_plane.measured.n_fpa={m.n_fpa} != "
                                 f"gd_polynomials.N_FPA={N_FPA}")
            if m.n_px != N_PX:
                raise ValueError(f"focal_plane.measured.n_px={m.n_px} != "
                                 f"gd_polynomials.N_PX={N_PX}")

    @classmethod
    def from_yaml(cls, path=None) -> "GeoCarbInstrumentConfig":
        path = Path(path) if path is not None else config_root() / "geocarb_instrument.yml"
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "GeoCarbInstrumentConfig":
        bands_raw = sorted(raw["bands"], key=lambda b: b["fpa"])
        bands = [(b["label"], float(b["wn_min"]), float(b["wn_max"]),
                 list(b["molecules"]), float(b["resolving_power"])) for b in bands_raw]
        band_fpa = {b["label"]: int(b["fpa"]) for b in bands_raw}

        fp_raw = raw["focal_plane"]
        measured = None
        if fp_raw.get("measured") is not None:
            m = fp_raw["measured"]
            measured = MeasuredFocalPlaneSpec(
                gd_csv_path=Path(m["gd_csv_path"]),
                n_fpa=int(m["n_fpa"]), n_px=int(m["n_px"]),
                dispersion_ascending={int(k): bool(v) for k, v in m["dispersion_ascending"].items()},
                spatial_psf_fwhm_px=float(m["spatial_psf_fwhm_px"]),
            )
        analytic = None
        if fp_raw.get("analytic") is not None:
            analytic = {}
            for entry in fp_raw["analytic"]:
                entry = dict(entry)
                fpa = int(entry.pop("fpa"))
                analytic[fpa] = FocalPlaneModel(**entry)
        focal_plane = FocalPlaneSpec(mode=fp_raw["mode"], measured=measured, analytic=analytic)

        g = raw["geometry"]
        geometry = GeometrySpec(
            slit_length_km=float(g["slit_length_km"]),
            pixel_size_ew_km=float(g["pixel_size_ew_km"]),
            pixel_size_ns_km=float(g["pixel_size_ns_km"]),
            integration_time_s=float(g["integration_time_s"]),
            sat_alt_km=float(g["sat_alt_km"]),
            sat_lon_deg=float(g["sat_lon_deg"]),
        )

        noise_by_fpa = {int(k): dict(v) for k, v in raw["noise"]["by_fpa"].items()}

        ss = raw.get("spectral_sampling", {})
        return cls(
            name=raw["name"], bands=bands, band_fpa=band_fpa,
            channels_per_fwhm=int(ss.get("channels_per_fwhm", 3)),
            hires_spacing=float(ss.get("hires_spacing", 0.01)),
            focal_plane=focal_plane, geometry=geometry, noise_by_fpa=noise_by_fpa,
        )

    # ── convenience wrappers over existing geocarb_gert/geosat_geometry code ──

    def build_instrument(self, snr: float = 300.0, noise_model=None):
        """Delegates to ``geocarb_gert.instrument.build_geocarb_instrument``
        (the existing, working band-tuple -> ``gert.Instrument`` builder)."""
        from .instrument import build_geocarb_instrument
        return build_geocarb_instrument(bands=self.bands, channels_per_fwhm=self.channels_per_fwhm,
                                        hires_spacing=self.hires_spacing,
                                        snr=snr, noise_model=noise_model)

    def noise_model(self, fpa: int):
        """``LinearShotNoise`` for one band, calibrated from this config's
        ``noise_by_fpa`` -- the config-driven equivalent of
        ``geocarb_gert.radiometry.geocarb_noise_model(fpa)``."""
        from gert.instrument import LinearShotNoise
        from .radiometry import linear_shot_noise_params
        spec = self.noise_by_fpa[fpa]
        N0, N1 = linear_shot_noise_params(spec)
        return LinearShotNoise(N0=N0, N1=N1, I_max=spec["I_max"])

    def noise_model_multi(self, fpas):
        """Stacked multi-band ``LinearShotNoise``, config-driven equivalent
        of ``geocarb_gert.radiometry.geocarb_noise_model_multi(fpas)``."""
        from gert.instrument import LinearShotNoise
        models = [self.noise_model(fpa) for fpa in fpas]
        return LinearShotNoise(N0=[m.N0 for m in models], N1=[m.N1 for m in models],
                               I_max=[m.I_max for m in models])

    def satellite(self):
        """Construct a ``LongSlitGeoSatellite`` from ``self.geometry`` --
        config-driven callers never see that class's own (currently
        disagreeing) constructor defaults."""
        from geosat_geometry import LongSlitGeoSatellite
        g = self.geometry
        return LongSlitGeoSatellite(
            sat_lon_deg=g.sat_lon_deg, slit_length_km=g.slit_length_km,
            pixel_size_ew_km=g.pixel_size_ew_km, pixel_size_ns_km=g.pixel_size_ns_km,
            integration_time_s=g.integration_time_s, sat_alt_km=g.sat_alt_km,
        )


# ── retrieval_defaults.yml ──────────────────────────────────────────────────

@dataclass
class RetrievalDefaults:
    """Loaded from ``retrieval_defaults.yml``. One field per sweep/algorithm
    default currently hardcoded across ``scripts/gd_joint_block_whole_slit_
    sweep.py`` (module constants and argparse defaults) and
    ``geocarb_gert/joint_state.py`` (``DEFAULT_CORR_LENGTH_ETA``)."""
    min_window: int
    pad: int
    g_ratio: float
    gamma: float
    sigma_abs: float
    jacobian: str
    state_interp: str
    anchor_density: int
    prior_form: str
    prior_fields: str
    free: tuple
    default_fpa: tuple
    corr_length_eta: dict
    n_lookup_samples_default: int
    n_lookup_samples_imperfect_prior: int

    @classmethod
    def from_yaml(cls, path=None) -> "RetrievalDefaults":
        path = Path(path) if path is not None else config_root() / "retrieval_defaults.yml"
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "RetrievalDefaults":
        t, s, b, samp = raw["tiling"], raw["solve"], raw["band"], raw["sampling"]
        return cls(
            min_window=int(t["min_window"]), pad=int(t["pad"]), g_ratio=float(t["g_ratio"]),
            gamma=float(s["gamma"]), sigma_abs=float(s["sigma_abs"]),
            jacobian=s["jacobian"], state_interp=s["state_interp"],
            anchor_density=int(s["anchor_density"]), prior_form=s["prior_form"],
            prior_fields=s["prior_fields"], free=tuple(s["free"]),
            default_fpa=tuple(int(f) for f in b["default_fpa"]),
            corr_length_eta={k: float(v) for k, v in raw["correlation_length_eta"].items()},
            n_lookup_samples_default=int(samp["n_lookup_samples_default"]),
            n_lookup_samples_imperfect_prior=int(samp["n_lookup_samples_imperfect_prior"]),
        )

    def apply_as_argparse_defaults(self, parser) -> None:
        """Set every matching ``dest``'s default on an already-built
        ``argparse.ArgumentParser`` (Phase C). Kept as an explicit,
        opt-in call rather than baked into every ``add_argument(...,
        default=...)`` line, so a script can choose whether/when to apply
        config-sourced defaults."""
        parser.set_defaults(
            min_window=self.min_window, g_ratio=self.g_ratio,
            gamma=self.gamma, sigma_abs=self.sigma_abs,
            jacobian=self.jacobian, state_interp=self.state_interp,
            anchor_density=self.anchor_density, prior_form=self.prior_form,
            prior_fields=self.prior_fields, free=",".join(self.free),
            fpa=",".join(str(f) for f in self.default_fpa),
        )
