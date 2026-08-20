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
- **Two distinct, separable contributors to the forward-only residual,
  quantified directly (2026-08-20)** — worth stating precisely, since
  "resolution error" undersells that there are two different mechanisms
  with different dominance regimes, not one:
  - *Bin-interpolation error*: the G-bin state's own missing curvature
    between bin centers (this is what `g_ratio` controls). Isolated by
    holding `anchor_density=16` fixed and sweeping `g_ratio` alone
    (`resid_bingrid_g<N>_linear_ad16_fpa2.npy`): residual RMS climbs from
    1.44e-4 (`g_ratio=0.5`) to 8.96e-3 (`g_ratio=12`), a 62x range.
  - *Pixel-to-anchor snap error*: `nearest_bin_scene`'s hard, non-
    interpolated nearest-anchor assignment — present even with a
    perfectly exact state at every anchor, since a whole neighborhood of
    real pixels still gets stamped with one anchor's own spectrum.
    Isolated using the `native`/`nativegrid` configs (no bin layer at
    all — state built directly at anchor resolution, so nothing but this
    mechanism can contribute): residual RMS falls from 4.58e-3
    (`anchor_density=0.25`) to 1.11e-4 (`anchor_density=16`), first-order
    in anchor spacing, confirming the qualitative claim above with real
    numbers.
  - **Which one dominates depends on where `g_ratio` sits, not a fixed
    ranking.** Comparing the full bin+anchor residual (at fixed
    `anchor_density=16`) against the anchor-only floor: at fine `g_ratio`
    (0.5–1) the full residual sits right at the anchor floor (1.3x it) —
    bin interpolation is already good enough not to matter, and the
    pixel-snap floor is the actual bottleneck. At coarse `g_ratio` (6–12)
    the full residual is 20–81x the anchor floor — bin-interpolation
    error now dominates completely. Neither mechanism is negligible in
    general; which one is the lever depends on which regime a given
    config is in.
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
- **`geocarb_gert/mission_config.py`** / **`input/geocarb_instrument.yml`**,
  **`input/retrieval_defaults.yml`** — single source of truth for
  instrument parameters (bands, focal-plane geometry, spatial resolution,
  noise calibration) and retrieval-sweep defaults, replacing scattered
  module constants. See §6.

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

**Phase 1 result (2026-08-20)**: all 30 configs ran (`results/
config_matrix/retrieval_bingrid_v1/REPORT.md`, figures under `plots/
config_matrix/retrieval_bingrid_v1/`). CO2 rms bias improves
monotonically as `g_ratio` shrinks for both free-parameter sets (e.g.
`co2-58-g0.5-ad16` 0.019 ppm vs. `co2-58-g12-ad16` 1.26 ppm), and
`anchor_density` gives a real but much smaller secondary improvement at
fixed `g_ratio`; `co2p` shows higher bias than `co2`-only at matched
settings, especially at loose `g_ratio` — the harder-conditioning
question that config was designed to probe.

Checked directly against §3's own forward-only residual (same
`g_ratio`/`anchor_density=16` configs, `resid_bingrid_g<N>_linear_ad16_
fpa2.npy`) — not just "does the solve approach the ceiling," which
undersells it:

| g_ratio | forward-only residRMS (no solve) | retrieved residRMS (co2) | improvement |
|---|---|---|---|
| 0.5 | 1.44e-4 | 1.59e-5 | 9.0x |
| 1 | 1.47e-4 | 1.60e-5 | 9.2x |
| 3 | 3.83e-4 | 1.90e-5 | 20.2x |
| 6 | 2.25e-3 | 5.19e-5 | 43.4x |
| 12 | 8.96e-3 | 2.19e-4 | 40.8x |

The solve doesn't just approach the forward-only ceiling — it goes well
below it, by 9x at fine `g_ratio` up to ~40x at coarse `g_ratio`. This
ties directly to the two-mechanism split above (§3, "What's validated as
of now"): the forward-only check freezes the state at exactly the bin-
center truth value, so its residual is 100% representation error (bin-
interpolation + pixel-snap, uncorrected). The real GN solve is free to
move the state away from that exact value whenever doing so reduces the
spectral residual — and it has more room to do that precisely where
representation error is larger to begin with, which is why the
improvement factor grows with `g_ratio` (more bin-interpolation error to
partially correct) rather than staying flat.

**The ppm-bias ceiling, checked directly (2026-08-20) — and it does NOT
track the spectral-residual result.** The forward-only side has no ppm
number stored, but it doesn't need re-rendering to get one: `prior_co2_
ppm_bins` at each window's own `bin_centers` already **is** exact truth
sampled at the bin scale (verified bit-identical against `als.STATE_
FIELDS["co2_ppm"]` evaluated at the same positions), so stitching it
across all 58 windows with the exact same `interp`-onto-`eta_rows`
machinery `gd_joint_block_matrix.py`'s own `stitch()`/`score()` uses for
the real posterior gives the representation-only ppm-bias ceiling with
no forward-model rerun needed — an apples-to-apples "prior, unsolved" vs.
"posterior, solved" comparison using identical scoring code:

| g_ratio | ceiling rms | co2 retrieved rms | co2 ratio | co2p retrieved rms | co2p ratio |
|---|---|---|---|---|---|
| 0.5 | 0.0007 | 0.0193 | **27.6x worse** | 0.0155 | 22.1x worse |
| 1 | 0.0021 | 0.0178 | **8.5x worse** | 0.0149 | 7.1x worse |
| 3 | 0.0464 | 0.0379 | 0.82x (at ceiling) | 0.0360 | 0.78x (at ceiling) |
| 6 | 0.2010 | 0.2454 | 1.22x worse | 0.1630 | 0.81x (at ceiling) |
| 12 | 0.3311 | 1.2591 | 3.80x worse | 0.7786 | 2.35x worse |

Unlike the spectral residual, which the solve always improved (9-40x),
the ppm bias is *worse than not solving at all* at fine `g_ratio` — by
more than an order of magnitude at `g_ratio=0.5`, where the unsolved
prior is already essentially exact (0.0007 ppm) and the solve pushes it
to 0.019 ppm. This is the two objectives (spectral fit vs. ppm accuracy)
diverging once representation error exists: `prior=truth` is not
actually the minimizer of the (noiseless) GN objective, because even the
exact bin-truth state, once downscaled and rendered, cannot reproduce
the dense-truth radiance exactly (§3's own representation-error floor,
never zero). The solver has no way to know that; it only sees a
nonzero residual and reduces it, which means moving the state away from
the ppm-accurate prior whenever that happens to fit the (partly
discretization-artifact) residual pattern better — textbook forward-
model/smoothing error aliasing into the retrieved state, not a bug in
the solve. `co2p` is consistently a little closer to the ceiling than
`co2`-only at every `g_ratio` (worth noting, not yet explained). At
`g_ratio=3` both free-sets sit almost exactly at the ceiling — the one
point in this grid where the two objectives roughly agree. **Practical
implication for Phase 2 and beyond**: "finer `g_ratio` gives a more
accurate retrieval" is true for the spectral fit but false for ppm bias
in this prior=truth, no-noise setting — coarser `g_ratio` is not
strictly worse once the state has room to drift from an already-correct
prior. Worth deliberately checking whether this reverses, or how much,
once Phase 2's imperfect (non-truth) prior removes the special case
where the unsolved starting point is itself near-perfect.

Robustness check: the ratio holds essentially unchanged across all three
tested `anchor_density` values (1, 4, 16), not just ad16 — e.g. at
`g_ratio=0.5` the retrieved/ceiling ratio is 29.5x/26.5x/26.3x across
ad=1/4/16, and similarly flat at every other `g_ratio`. `anchor_density`
being a much weaker lever than `g_ratio` (already established above)
means the "solving is worse than not solving" effect at fine `g_ratio`
is not an anchor-density artifact — it is `g_ratio`-driven alone.

**What is the solve actually tracking, if not the bin-center prior? Three
increasingly careful tests (2026-08-20), same `co2-58-g<N>-ad16`
configs.** All three ask the same question in a progressively more
faithful way: is there a *different*, physically sensible target — not
exact truth at the bin's own center point — that the retrieved bin value
sits closer to than it does to `prior_co2_ppm_bins`?

1. *Naive bin average*: the true CO2 field, pixel-averaged over each
   bin's own nearest-bin-center footprint (unweighted). Does not explain
   it — at `g_ratio` 0.5/1/3, the retrieved value is *closer* to the bin
   center than to this average (rms 0.019/0.016/0.031 vs. center vs.
   0.020/0.030/0.092 vs. average) — the opposite of the hypothesis. Only
   at coarse `g_ratio` (6, 12) do the two distances become close to each
   other, and even then it is a weak signal next to the >1 ppm retrieval
   error at that point.
2. *Interpolation-aware least-squares target*: properly accounts for how
   a bin's value actually reaches a pixel — piecewise-linearly
   interpolated onto the `anchor_etas` grid (the real `M` matrix
   `build_forward_state` uses for `state_interp="linear"`), then
   nearest-anchor-snapped to pixels, exactly matching the real rendering
   chain. Solves for the bin values that best reproduce true CO2 under
   that exact chain (ridge-stabilized against `prior_co2_ppm_bins`; one
   of 58 windows was rank-deficient after dropping anchors with zero real
   pixels, at `g_ratio` 0.5 and 1 only — does not change the pattern in
   the other 57). Still does not explain the fine-`g_ratio` divergence —
   0.0195/0.0188/0.0343 vs. this target at g=0.5/1/3, no better than the
   bin-center distances (0.0187/0.0164/0.0307). Only helps at coarse
   `g_ratio` (0.2195 vs. 0.2837 at g=6; 1.0714 vs. 1.1053 at g=12), and
   only modestly.
3. **PSF-aware least-squares target**: same chain as (2), but the true
   CO2 field is first convolved along rows with the actual spatial-PSF
   kernel `gaussian_blur_rows` uses (FWHM=1.5px, σ≈0.64 rows — rows only,
   never columns, matching that function's own mixing) before computing
   the anchor-level target. **This is the best predictor found, and the
   only one that helps across the entire `g_ratio` range rather than
   just one end of it**:

   | g_ratio | \|retr − bin center\| | \|retr − LSQ, no PSF\| | \|retr − LSQ, PSF-aware\| |
   |---|---|---|---|
   | 0.5 | 0.0187 | 0.0195 | **0.0151** |
   | 1 | 0.0164 | 0.0188 | **0.0139** |
   | 3 | 0.0307 | 0.0343 | **0.0301** |
   | 6 | 0.2837 | 0.2195 | **0.2183** |
   | 12 | 1.1053 | 1.0714 | **1.0712** |

   Largest improvement exactly where tests (1) and (2) explained nothing
   — 19% tighter at `g_ratio=0.5`, 15% at `g_ratio=1` — consistent with
   the physical picture: the GN solve is fitting the actually-measured,
   PSF-blurred radiance, so a target built from the same blur operator
   predicts its behavior better than one that ignores it. Does not close
   the gap entirely (0.014-0.015 ppm of divergence remains even against
   this target at fine `g_ratio`) — some genuine forward-model-error
   aliasing on top of the PSF effect is still unaccounted for, not a
   complete explanation, but the strongest one found so far and the only
   one that is not regime-specific.

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

*(2026-08-20 update: this note is stale — `--realistic-prior` was retired
in favor of the standardized `--prior-fields`/`als.PRIOR_FIELD_SETS`
registry, which IS wired through `gd_joint_block_matrix.py`'s own
`--prior-fields` axis; see §6's imperfect-prior "first try" results.)*

## 6. Config consolidation + the real noise model (2026-08-20)

Every "interesting" instrument parameter — bands, focal-plane geometry
(keystone/smile/clocking), spatial resolution, slit length, noise
calibration — and every retrieval-sweep default (`g_ratio`,
`anchor_density`, correlation lengths, ...) now has exactly one place
it's defined: `input/geocarb_instrument.yml` and
`input/retrieval_defaults.yml`, loaded via
`geocarb_gert/mission_config.py`'s `GeoCarbInstrumentConfig`/
`RetrievalDefaults`. This replaced several previously-scattered, and in a
few cases *disagreeing*, module constants — most notably
`along_slit_scene.SLIT_HALF_KM` (1400.0, what production truth-rendering
actually used) vs. `geosat_geometry.LongSlitGeoSatellite`'s own
`slit_length_km=3000.0` default (never actually wired in): the latter's
default now matches the former exactly, rather than the two merely being
close. `gd_joint_block_whole_slit_sweep.py` and `gd_joint_block_matrix.py`
both gained a `--config` flag (YAML values become argparse defaults, CLI
flags still override — SLURM sweep automation unaffected) and the sweep
script gained a real `--fpa` flag, making band selection a run-time
choice instead of an edit-the-source constant (single-band only for now
— multi-band joint retrieval, building on the already-existing
`geocarb_noise_model_multi`/`cross_band.nearest_row_pairing_multi`
machinery, is a later phase).

**The `Sy_inv` noise-model swap.** The real, ground-test-calibrated
GeoCarb noise model (`geocarb_gert.radiometry.geocarb_noise_model`,
backed by `RADIOMETRIC_SPEC_BY_FPA`) existed but was never actually used
to weight the production Gauss-Newton solve — `_solve_window` instead
used an ad hoc flat scalar, `Sy_inv_diag = 1/mean(|signal|)^2` for the
whole window. This is now replaced by the real per-pixel
`sqrt(N0^2 + N1*|I|)` noise model as the default (not opt-in — a
deliberate project decision, since a real noise model should be how the
code works, not a flag nobody remembers to pass).

**This changes retrieved numbers** relative to every result in §5 above
(the Phase 1 bingrid sweep and the Phase 2 imperfect-prior "first try"
results) — expected, not a bug. Quantified at the CO2/CO hot-spot window
(rows 890-935, `g_ratio=3`, `anchor_density=1`): the old "sigma" was a
spatially uniform 6.62 (radiance units) — nonsensically large, comparable
to the scene's own dynamic range (`y_true` ranges 0.10-11.97 there) — an
artifact of the old formula never having been a real noise model, just a
relative-weighting scalar dressed up as one. The new per-pixel sigma
averages ~0.018 (matching `y_true.mean()/SNR_ref` to the expected
order of magnitude) and genuinely varies across the window
(std ~0.005) tracking local radiance. Despite the ~365x difference in
absolute scale, the retrieved CO2 itself shifted by only up to ~0.03 ppm
at this window (out of a ~3.7 ppm peak enhancement) — this window is
heavily overdetermined (~47,000 pixels vs. 15 state unknowns), so the fit
is data-dominated under either weighting; the real payoff of the fix is a
*physically calibrated* posterior uncertainty (tighter in bright/
high-SNR regions, looser in dim ones), which the old flat scalar could
not represent, not a large change to the central retrieved value. A full
58-window smoke run (`--g-ratio 12`, coarse) confirmed no NaN/inf
anywhere and residual/bias behavior matching the already-documented
coarse-`g_ratio` degradation pattern above — a real, sane retrieval, not
noise-dominated garbage.

Any future comparison against the Phase 1/Phase 2 numbers above should
account for this: they were computed under the old, uncalibrated flat
weighting.
