#!/usr/bin/env python3
"""Phase A regression check for the config-consolidation plan (see the
approved implementation plan, and docs/PROJECT_STATUS.md): confirms
GeoCarbInstrumentConfig.from_yaml()/RetrievalDefaults.from_yaml() reproduce
every currently-hardcoded module constant they are meant to eventually
replace, EXACTLY -- this module is purely additive in Phase A (nothing
imports from it yet), so the only thing to verify is that the loader is
correct, not that anything downstream changed.

Run:  PYTHONPATH=. python scripts/check_mission_config.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from geocarb_gert.mission_config import GeoCarbInstrumentConfig, RetrievalDefaults  # noqa: E402
from geocarb_gert.instrument import GEOCARB_BANDS  # noqa: E402
from geocarb_gert.radiometry import RADIOMETRIC_SPEC_BY_FPA, GEOCARB_REF  # noqa: E402
from geocarb_gert.gd_polynomials import N_FPA, N_PX, DISPERSION_ASCENDING  # noqa: E402
from geocarb_gert.along_slit_scene import SLIT_HALF_KM  # noqa: E402
from geocarb_gert.joint_state import DEFAULT_CORR_LENGTH_ETA  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import gd_joint_block_whole_slit_sweep as sweep  # noqa: E402
from gd_joint_block_retrieve import FPA  # noqa: E402

_FAILURES = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"{'OK  ' if ok else 'FAIL'}  {name}: {got!r}" + ("" if ok else f"  != {want!r}"))
    if not ok:
        _FAILURES.append(name)


def main() -> int:
    cfg = GeoCarbInstrumentConfig.from_yaml()
    rdef = RetrievalDefaults.from_yaml()

    check("bands == GEOCARB_BANDS", cfg.bands, GEOCARB_BANDS)
    check("noise_by_fpa == RADIOMETRIC_SPEC_BY_FPA", cfg.noise_by_fpa, RADIOMETRIC_SPEC_BY_FPA)
    check("geometry.pixel_size_ns_km == GEOCARB_REF['gsd_km']",
         cfg.geometry.pixel_size_ns_km, GEOCARB_REF["gsd_km"])
    check("geometry.integration_time_s == GEOCARB_REF['t_int_s']",
         cfg.geometry.integration_time_s, GEOCARB_REF["t_int_s"])
    check("geometry.sat_alt_km == GEOCARB_REF['sat_alt_km']",
         cfg.geometry.sat_alt_km, GEOCARB_REF["sat_alt_km"])
    check("geometry.slit_half_km == SLIT_HALF_KM", cfg.geometry.slit_half_km, SLIT_HALF_KM)
    check("focal_plane.measured.n_fpa == N_FPA", cfg.focal_plane.measured.n_fpa, N_FPA)
    check("focal_plane.measured.n_px == N_PX", cfg.focal_plane.measured.n_px, N_PX)
    check("focal_plane.measured.dispersion_ascending == DISPERSION_ASCENDING",
         cfg.focal_plane.measured.dispersion_ascending, DISPERSION_ASCENDING)
    check("focal_plane.measured.spatial_psf_fwhm_px == 1.5",
         cfg.focal_plane.measured.spatial_psf_fwhm_px, 1.5)
    check("focal_plane.measured.gd_csv_path resolves to the real CSV",
         (REPO_ROOT / cfg.focal_plane.measured.gd_csv_path).exists(), True)

    check("RetrievalDefaults.min_window == sweep.MIN_WINDOW", rdef.min_window, sweep.MIN_WINDOW)
    check("RetrievalDefaults.pad == sweep.PAD", rdef.pad, sweep.PAD)
    check("RetrievalDefaults.g_ratio == sweep.G_RATIO", rdef.g_ratio, sweep.G_RATIO)
    check("RetrievalDefaults.default_fpa == (FPA,)", rdef.default_fpa, (FPA,))
    check("RetrievalDefaults.corr_length_eta == DEFAULT_CORR_LENGTH_ETA",
         rdef.corr_length_eta, DEFAULT_CORR_LENGTH_ETA)

    # analytic focal-plane mode: round-trip check independent of the
    # measured-mode checks above (constructs cleanly, one FocalPlaneModel
    # per band, fields match what was written into the YAML).
    from geocarb_gert.mission_config import GeoCarbInstrumentConfig as _Cfg
    import yaml
    raw = yaml.safe_load(open(REPO_ROOT / "input" / "geocarb_instrument.yml"))
    raw_analytic = dict(raw)
    raw_analytic["focal_plane"] = dict(raw["focal_plane"])
    raw_analytic["focal_plane"]["mode"] = "analytic"
    cfg_analytic = _Cfg.from_dict(raw_analytic)
    check("analytic mode builds 4 FocalPlaneModels", len(cfg_analytic.focal_plane.analytic), 4)
    check("analytic FPA2 keystone_frac == 1e-3",
         cfg_analytic.focal_plane.analytic[2].keystone_frac, 1e-3)

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} check(s) FAILED: {_FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
