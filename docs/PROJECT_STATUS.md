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
- The realistic-prior forward-only residual comparisons: `bingrid_g<N>_
  <kind>_ad<d>` (G-bin state, prior=truth at each bin, downscaled onto a
  fixed anchor density) against `nativegrid_ad<d>`/`native_ad<d>` (no G-bin
  layer, state built directly on the anchor grid) — all cross-checked
  against the same dense truth image. `"nearest"` downscaling is rejected
  (2026-08-19, `forward_check_g_ratio_sweep_fpa<N>.png`) — `"linear"` only
  going forward; the `nearest` sweep data is kept for reference under
  `results/realistic_prior/forward_check/nearest_deprecated/` and
  `plots/realistic_prior/nearest_deprecated/`, out of the default glob.
  `exact`/`coarse<d>`/`oversample<d>`/`structural`/`highres` (the
  `prior_anchor_density` sweep and the structural-fields check) are
  retired — superseded by the bingrid/nativegrid framing above, which
  covers the same resolution question without a separate mechanism.

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
- **`scripts/gd_joint_block_matrix.py`** — generic N-way config-matrix
  runner around the whole-slit sweep (free rows × windows × `g_ratio` ×
  `anchor_density` × `state_interp` × jacobian, each axis a CLI list) —
  the tool §5's retrieval sweep runs through.
- **`scripts/gd_realistic_prior_forward_check.py`** — forward-only
  diagnostics (no retrieval): residual comparisons, RT-resolution sweeps,
  the bin/anchor schematics.
- **`scripts/gd_jacobian_validate.py`** — analytic-vs-FD validation.

## 5. Retrieval plan (next steps)

Everything in §3 is forward-only (`forward(spec.x0())`, prior against
truth, never a solve) — it characterizes representation error, not
retrieval error. The open question this phase answers: does adding
measurement noise and an actual `gauss_newton_state` solve change the
g_ratio/anchor_density story §3 already measured, or does conditioning
(more free state than the data can actually constrain — see the "reason
not to solve on the native grid" discussion, `analytic_jacobian_testing`'s
own closed regression) become the dominant effect once real solves are in
the loop?

### Phase 1 — prior=truth at the bin scale, linear downscale

Repeats the `bingrid_g<N>_linear_ad<d>` mechanism from §3 (prior=truth
exactly at each G bin, `state_interp="linear"` downscale onto a fixed
anchor grid), but now actually solving instead of forward-rendering the
prior. Two free-parameter configs, both against the full 58-window FPA2
tiling:

- **CO2 only** (`--free co2` → `co2_ppm`).
- **CO2 + surface pressure** (`--free co2p` → `co2_ppm,p_surface_hpa`) —
  tests whether adding a second free row changes how much g_ratio/
  anchor_density resolution is needed to reach the noise floor (two rows
  competing for the same per-window information is a harder-conditioned
  problem than one, independent of the H2O/p_surface degeneracy §2 found
  joint-band retrieval fixes — this is a single-band, two-free-row case).

Both swept across `g_ratio` and `anchor_density`, via
`scripts/gd_joint_block_matrix.py` (`--hires-only`, since `coarse` never
touches `anchor_etas` and so cannot depend on `anchor_density` at all —
running it while sweeping that axis would be pure waste; `--jacobian
analytic`, already established against FD everywhere this tool sweeps;
`--state-interp linear`, the only kind still in use per §3):

```
PYTHONPATH=. python3 scripts/gd_joint_block_matrix.py \
    --tag retrieval_bingrid_v1 --free co2,co2p \
    --g-ratio 0.5,1,3,6,12 --anchor-density 1,4,16 \
    --hires-only --jacobian analytic
```

`g_ratio`/`anchor_density` values above are a starting proposal, not
fixed — chosen as a practically-costed subset of §3's own axes (dropping
`g_ratio=0.125,0.25`/`anchor_density`∈{2,8} initially; each config here is
a real 58-window GN solve, not a forward render — the closed
`anchor_density_v1` check under `results/config_matrix/` timed a single
`co2p`, `g_ratio=1`, `anchor_density=4` config at ~5-15 minutes, so the
30-config grid above (2 free-sets × 5 g_ratios × 3 anchor-densities) is a
real wall-clock commitment, worth timing one config before launching the
rest). Score against: (a) §3's own forward-only representation-error
ceiling at matching `(g_ratio, anchor_density)` — the floor retrieval RMS
should approach if the solve is well-conditioned, since with prior=truth
and no noise there's nothing else for the solve to correct; and (b) the
instrument noise floor (`1/SNR`), same convention as every §3 plot.

### Phase 2 — imperfect prior (after Phase 1)

Repeats a subset of Phase 1's grid with `fields=STATE_FIELDS_PRIOR`
(background/topography-aware, never plume/hot-spot content — see §3
"Prior construction") instead of prior=truth, via
`gd_joint_block_whole_slit_sweep.py`'s own `--realistic-prior` flag. This
is the first time the "prior=truth and frozen state are two separate
actions" design principle (settled earlier — no exact-truth bypass in the
forward path, prior content is the only place truth-vs-degraded is
decided) actually gets exercised in a real retrieval rather than a
forward-only check.

**Not yet wired**: `gd_joint_block_matrix.py`'s `run_one` doesn't forward
a `--realistic-prior`-equivalent flag to the whole-slit sweep script (its
own `--scene` axis controls atmosphere composition — uniform/barcode —
not prior-field quality). Needs a small addition (a `--prior-fields`
matrix axis, or reuse `--scene`) before Phase 2 can run through the same
matrix tooling as Phase 1; until then, individual
`gd_joint_block_whole_slit_sweep.py --realistic-prior ...` runs work
directly.
