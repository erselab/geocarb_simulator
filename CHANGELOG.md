# Changelog

Notable changes to the GeoCarb long-slit scan simulator.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This is
a research project rather than a released library, so entries are dated rather
than semantically versioned.

---

## [Unreleased]

### Added

**`geocarb_gert` — adapter driving the [GERT](../../gert) RT/retrieval library**

Enables instrument-design OSSEs: how GSD, dwell time and detector noise
propagate into the posterior uncertainty on retrieved XCO₂. The package is
deliberately thin — it contains **no radiative transfer and no retrieval code**.

* `geocarb_gert.instrument` — the four GeoCarb bands (O₂ A-band, CO₂ weak,
  CO₂ strong, CH₄/CO) as a `gert.Instrument`. All four are already densely
  covered by the existing ABSCO table; the O₂ band uses the 760 nm `o2` dataset.
* `geocarb_gert.radiometry` — the **mission-side** mapping from GSD and dwell to
  a `gert.RadiometricNoise`. GeoCarb stares from GEO, so integration time is a
  free parameter *independent of GSD* (étendue ∝ GSD²), unlike a LEO pushbroom
  where `t_int = GSD/v_ground`. Because a quoted SNR cannot separate étendue,
  throughput and QE, a lumped throughput is calibrated from a spec point and
  rescaled.
* `geocarb_gert.adapter` — `ScanBlock` → per-pixel `gert.Geometry` (using the
  block's `szas`, `vzas`, `relative_azimuth`) and `gert.osse.Scene`.
* `geocarb_gert.scene` — a US-Standard-Atmosphere `AtmosphericProfile` on the
  GERT level grid, for design studies and smoke tests. Model-driven scenes come
  from `model_sampler.sample_field_along_rays`.
* `scripts/run_design_sweep.py` and `notebooks/geocarb_design_sweep.ipynb` —
  a (GSD × dwell) design sweep. Since `S_ret = (KᵀSy⁻¹K + Sa⁻¹)⁻¹` does not
  depend on the noise *realization*, varying GSD/dwell only rescales `Sy` and the
  Jacobian `K` is computed **once** for the whole grid.
* `README.md` section documenting the seam and the layering rule
  (`docs/STATUS_AND_ROADMAP.md` §3.7 in the gert repo).

### Findings

* **σ(XCO₂) ∝ t_int^-0.5 in both the shot- and dark-limited regimes** (measured
  −0.521), because signal ∝ `t` while both noise terms ∝ `√t`. A clean −0.5
  dwell slope therefore says *nothing* about which noise source dominates.
* **The σ(XCO₂)-vs-GSD exponent runs between −1 and −2**, not a fixed −1.
  Shot-limited gives −1; dark current and read noise steepen it toward −2 at
  small GSD (measured −1.313, with dark current at 25–40 % of shot noise at
  3 km). The exponent is a property of the **scene and dwell**, not of the
  instrument — something a scalar `SNR = 300` specification cannot express.
* **Saturation is a constraint orthogonal to SNR.** A long dwell at coarse GSD
  overflows the well (12 km / 40 s → 1.6e6 e⁻ into a 1e6 e⁻ well); the design
  point is invalid however good its nominal σ looks. Saturated points are
  reported and excluded from the scaling fits.

### Requirements

* `gert >= 0.1` installed as a library (`pip install -e /path/to/gert`), with the
  user's own ABSCO and solar tables.

### Known issues

* **`__pycache__/*.pyc` files are tracked** and there is no `.gitignore`.
  Recommend adding one (`__pycache__/`, `*.pyc`, `.DS_Store`) and running
  `git rm -r --cached __pycache__` before the next commit.
* `notebooks/geocarb_design_sweep.ipynb` hardcodes the relative path to the gert
  repo (`../../../gert`); adjust if the layout differs.

---

## [Scan templates & timelines] — earlier

* Scan-template construction and per-column time stamping for full-day observing
  schedules (`build_day_schedule`, `chain_scan_blocks`, `ScanBlock.stamp`).

## [Model sampling] — earlier

* Bilinear sampling of 3-D/4-D model fields along satellite→ground ray paths,
  curvilinear (WRF Lambert / polar-stereographic) grid support via KDTree +
  inverse bilinear, out-of-domain masking for limited-area domains, and
  pressure-weighted column averaging (`model_sampler.py`, `scan_sampler.py`).

## [Initial release] — earlier

* `geosat_geometry.py` — long-slit geostationary observation geometry: slit
  pixel centres, footprint polygons, viewing and solar angles, satellite look
  vectors, 3-D ray tracing through spherical atmospheric shells, `ScanBlock`
  containers, coarsening, land-fraction masking, plotting and `.npz` persistence.