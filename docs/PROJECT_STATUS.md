# GeoCarb Simulator — Project Status

**As of 2026-08-19.** This is the current, authoritative description of the
project: what was tried and rejected (the 1D per-row pipeline era), and what
the design is now (joint multi-row retrieval on a `StateSpec`). It replaces
`KEYSTONE_SMILE_BIAS_PLAN.md`, `PROJECT_SUMMARY_AND_NEXT_STEPS.md`,
`JOINT_ROW_INVERSION_PLAN.md`, and `JOINT_BLOCK_MIGRATION_PLAN.md` as the
first thing to read — those four are archived in place (a banner at the top
of each, not moved) and still hold the full chronological detail, plots, and
section-numbered evidence behind everything summarized here.

---

## 1. What this project is

GeoCarb's real ground-test calibration report (`keystone_report.pdf`)
documents retrieval anomalies — non-convergence, residual striping — tied to
keystone/smile geometric distortion and the rectify-then-retrieve L1B
pipeline. This project builds a synthetic simulator (`geocarb_gert`, on top
of the `gert` forward-model/retrieval library) that reproduces the *real* GD
(geometric distortion) polynomials, detector geometry, and PSF, then uses it
to isolate which mechanisms actually drive retrieval bias, at what
magnitude, and what retrieval strategy actually copes with them.

## 2. The 1D-pipeline era (archived, retrospective)

The project's first phase treated every detector row independently: for a
given row, fit a spectrum against a single-atmosphere forward model, using
one of three geometric-handling strategies —

- **native**: fit directly on the true, distorted per-pixel grid.
- **undistorted**: fit against an idealized zero-distortion baseline.
- **rectified**: regrid (interpolate) the distorted detector onto a uniform
  wavelength grid first, then fit.

This is fully documented in `KEYSTONE_SMILE_BIAS_PLAN.md` (§1-12, the
chronological lab notebook) and `docs/NATIVE_VS_UNDISTORTED_ARTIFACT.html`
(the rendered battery of results across scenes and FPA combinations).
Confirmed findings, single-band:

- **Row-crossing/pixel-grid aliasing** is the leading single-band bias
  mechanism — a detector row's spectrum is an average over up to ~10 true
  along-slit rows near the slit edges. Matches real ground-test
  non-convergence patterns directly.
- **Rectification bias is the single biggest simulated bias source found**
  — tens-of-ppm, roughly slit-position-independent, ~10-14% outright
  non-convergence at full row density, and doesn't vanish at the smile-null
  row the way other mechanisms do.
- **PSF×smile coupling creates an irreducible, order-independent CO2 bias**
  (up to ~-5.9 ppm) even on a spatially uniform scene, from the PSF blending
  adjacent rows that each carry a slightly different real wavelength
  calibration.
- **The dispersion-order convolution artifact**: much of what looked like
  "native fits better than undistorted" was actually a numerical mismatch
  in ILS-kernel centering (`gert.instrument.ILS.convolve`'s
  `exact_center` only triggers when dispersion is in the state vector) —
  fixed by always floating a dispersion order, not a real physical finding.

Multi-band extension (still one row at a time per band, jointly fit):

- **Cross-band co-registration is solved** — the real slit angle `s` [deg]
  (not FPA-normalized `eta`) is the shared coordinate, confirmed by the
  actual ground-test calibration methodology.
- **Joint retrieval measurably helps**: FPA0(O2-A)+FPA2(strong CO2), no
  aerosol, reduces CO2 retrieval std ~4.4x vs. single-band, by breaking the
  H2O/surface-pressure degeneracy.

**The finding that ended this phase and motivated everything since**:
native's independent per-row retrieval systematically under-recovers real,
localized signal near high keystone — a true point source's peak amplitude
was captured at only 44% at row 910 (9.3 rows crossed) vs. 90% at row 112
(1.0 rows crossed), vs. ~99% for the idealized undistorted case at both.
**No 1D transformation of the FPA prior to fitting spectra escapes this** —
native, undistorted, and rectified all fit one row at a time, and the
information a real localized feature spreads across multiple true rows
(via keystone) is structurally unavailable to a single-row fit no matter how
that row's own geometry is handled. `JOINT_ROW_INVERSION_PLAN.md` (archived)
is the response to this finding: since the distortion operator is known
exactly, jointly inverting a neighborhood of rows is a classical inverse
problem, not something that needs new physics or ML.

The early joint-block prototype that followed (`JOINT_ROW_INVERSION_PLAN.md`,
`JOINT_BLOCK_MIGRATION_PLAN.md`, and the closed regression under
`analytic_jacobian_testing/`) served its purpose — establishing shared
definitions and catching real code bugs (a separable-rendering bug that
silently mixed two rows' dispersion calibrations; a `--uniform` flag that
never actually reached the state priors) — but has no standing value beyond
that. It is archived without a detailed summary here; its numbers were
superseded by the redesign in §3.

## 3. Current design

### State representation

`geocarb_gert/joint_state.py`'s `StateSpec`/`ParamSpec` represent the
retrieval state as a table of named rows (`co2_ppm`, `p_surface_hpa`,
`albedo`, ...), each independently free or frozen, each with its own
positions, prior, and regularization (sigma, correlation length) — no
parameter is structurally special. This is
`JOINT_BLOCK_MIGRATION_PLAN.md` §4.0's "regularization as a per-element
property, not per-group code," carried through unchanged. Freezing a row
and choosing what its prior contains are two independent decisions (see
"Prior construction," below) — never conflated.

### Two grids that don't nest

A retrieval window has two distinct spatial grids, and it's a common
confusion to conflate them:

- **The bin grid** (`G` values, via `g_ratio`): placed at *quantiles* of the
  real per-pixel η distribution in the window
  (`gd_joint_block_diagnostics.pixel_density_bin_centers`) — an
  equal-population, not equal-spacing, scheme, so bins pack tighter where
  keystone concentrates more real pixels into a narrow η range. This is
  where the retrieval's free parameters live.
- **The native/anchor grid**: a plain, uniformly-spaced-in-row grid,
  independent of pixel density, at a chosen `anchor_density` (anchors per
  detector row). This is the resolution RT is actually evaluated at.

These two grids are placed by genuinely different mechanisms and **do not
generally nest** — a bin center does not usually land on any anchor
position. Verified directly (2026-08-19) via `pcolormesh` renders of real
per-pixel bin/anchor assignment: every pixel's true `η(row, col)` is
monotonic in row at fixed column, which guarantees the assignment has no
holes at any single column, but the bin *boundary* itself is a curve across
columns (keystone/smile), not a flat row — a single bin's own defining
locus can cross most of a wide window's row range across the band.

### Rendering: `build_forward_state` + the analytic Jacobian

`build_forward_state` (`geocarb_gert/joint_state.py`) downscales the bin-grid
state onto the native anchor grid via `state_interp`, a string naming a
`scipy.interpolate.interp1d` kind — `"linear"` (default: blend the two
nearest bins) or `"nearest"` (piecewise-constant: each anchor takes its
single nearest bin's value). **Every row, free or frozen, always goes
through the same interpolation** — there is deliberately no bypass to exact
truth for frozen rows (removed 2026-08-19; it used to silently substitute
real truth regardless of what the prior was constructed to contain, which
violated "the way the FPA is rendered should be independent of the prior").
If a row needs to be as-good-as-truth, that's a prior-construction decision,
never a rendering one.

The analytic Jacobian (`geocarb_gert/jacobians.py`'s `linearize`) takes the
same `state_interp` kind and stays consistent with the forward model by
construction: `StateSpec.interp_weights` builds its chain-rule weight matrix
by running the *same* `interp1d` call on the identity matrix instead of real
values, for whichever kind is requested — so the weights can never drift
from what `interp_to` actually computed. This also means analytic Jacobians
are now unconditionally available for the hires solve, for either kind —
the old asymmetry (frozen rows silently exact-at-anchor while free rows
interpolated, which the Jacobian had no equivalent for, forcing hires to
fall back to finite differences whenever `state_interp` was off) no longer
exists, because there's nothing left for it to be asymmetric about.

Validated: `"nearest"`'s new analytic Jacobian agrees with central finite
differences to roundoff (`scripts/gd_jacobian_validate.py --interp-kind
nearest --h-scan`, clean V-curve, no floor) — the same quality bar already
established for `"linear"`. A real residual comparison on the high-keystone
window used throughout this validation (rows 925-967) confirmed
`"nearest"` gives ~1.8x worse representation error than `"linear"`, as
expected (piecewise-constant downscaling can't beat linear interpolation).

### Prior construction — its own, orthogonal axis

What a bin's prior *contains* is independent of the grid/interpolation
machinery above, controlled by `state_spec_from_scene`'s `fields=` (which
truth functions to sample: exact `STATE_FIELDS`, or the deliberately
degraded `STATE_FIELDS_PRIOR` — background/topography-aware, never
localized hot spots) and `prior_anchor_density` (how densely `fields` gets
sampled before being placed on the bin grid). Full detail, the forward-only
diagnostic tooling that validates all of this without running a real
retrieval, and the RT-resolution convergence findings (no plateau within any
practically reachable anchor density — representation error is already well
below the instrument noise floor at even the coarsest density tested, so
the actual limiting factor is prior *quality*, not grid resolution) are in
`REALISTIC_PRIOR_EXPERIMENT_CLASS.md` — current, not archived.

### What's validated as of now

- FD/analytic Jacobian agreement for both `state_interp` kinds.
- The bin-grid vs. native-grid non-nesting, and the pixel-snap/keystone
  mechanism setting the RT-resolution floor (`nearest_bin_scene`'s hard
  nearest-anchor assignment — never interpolated spectra — is first-order
  in anchor spacing with no inherent plateau).
- The realistic-prior forward-only residual comparisons (`exact` /
  `coarse<d>` / `oversample<d>` / `structural` / `highres` /
  `native_ad<d>`), all cross-checked against the same dense truth image.

## 4. Where things live

- **`docs/PROJECT_STATUS.md`** (this document) — start here.
- **`docs/KEYSTONE_SMILE_BIAS_PLAN.md`** (archived) — full chronological
  record of the 1D-pipeline era; search by "§N".
- **`docs/PROJECT_SUMMARY_AND_NEXT_STEPS.md`** (archived) — the 2026-07-28
  handoff snapshot §2's summary above is condensed from.
- **`docs/JOINT_ROW_INVERSION_PLAN.md`** (archived) — the pivot plan.
- **`docs/JOINT_BLOCK_MIGRATION_PLAN.md`** (archived) — the package-migration
  plan `StateSpec`'s design rationale comes from.
- **`docs/REALISTIC_PRIOR_EXPERIMENT_CLASS.md`** (current) — prior
  construction and the forward-check diagnostic tooling.
- **`geocarb_gert/joint_state.py`** — `StateSpec`/`ParamSpec`,
  `state_spec_from_scene`, `build_forward_state`, `gauss_newton_state`.
- **`geocarb_gert/jacobians.py`** — the analytic Jacobian (`linearize`).
- **`scripts/gd_joint_block_whole_slit_sweep.py`** — the production
  retrieval sweep (real `gauss_newton_state` solves, not forward-only).
- **`scripts/gd_realistic_prior_forward_check.py`** — forward-only
  diagnostics (no retrieval): residual comparisons, RT-resolution sweeps,
  the bin/anchor schematics.
- **`scripts/gd_jacobian_validate.py`** — analytic-vs-FD validation.
