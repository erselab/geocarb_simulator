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

**2026-08-21 update — reran Phase 2 under the corrected weighting, and
added `--flat-sy-inv` for deliberate old/new comparison.** The imperfect-
prior sweep's raw pkls from before Phase D were overwritten (accidentally,
mid-testing) before anyone thought to save them — `results/` is
gitignored, so there was no git history either, and they are permanently
gone. Added `--flat-sy-inv` to `gd_joint_block_whole_slit_sweep.py`/
`gd_joint_block_matrix.py`: reproduces the old flat-scalar `Sy_inv`
formula exactly (verified bit-identical to the surviving old REPORT.md
numbers, reconstructed from conversation transcript, across the full
30-config sweep) for deliberate A/B comparison — NOT a reversal of the
Phase D default, which is unchanged; `resolve_existing()` now checks each
cached file's own `flat_sy_inv` metadata so a mismatched reuse can't
silently happen. Reran both:
  - `results/config_matrix/retrieval_prior_{co2plus1pct,co2p}_v2/` —
    corrected (current default weighting)
  - `results/config_matrix/retrieval_prior_{co2plus1pct,co2p}_flatsyinv/` —
    old weighting, reproduced
  - `results/config_matrix/retrieval_prior_{co2plus1pct,co2p}_v1/` — the
    `_v1` tag itself now holds only the reconstructed old REPORT.md +
    restored-from-git old plots (its raw pkls were the thing overwritten;
    see the `results/.../v1/results/README.txt` left in place of them).

New comparison plots (`scripts/gd_retrieval_prior_convergence_compare.py`,
`scripts/gd_retrieval_prior_slit_chi2_compare.py`): CO2 rms improved
20-500x depending on config (e.g. `co2p` best case 4.24 -> 0.009 ppm), and
critically now shows the EXPECTED behavior of resolution improving
accuracy (old weighting was resolution-INSENSITIVE — flat regardless of
`g_ratio`, direct evidence it wasn't really using the data). Full
per-window pkls (unlike the original, now-lost run) also enabled a real
prior/posterior/chi2-vs-slit-position comparison and a chi2 defined
against the real per-pixel noise floor (careful: these are `noise=False`
sweeps, so chi2 here means representation-error-vs-assumed-noise-floor,
not a real measurement-noise goodness-of-fit test — no chi2~=1
expectation applies).

**Window-boundary finding (open, not yet acted on).** Each of the 58
windows is solved fully independently (no continuity constraint across
their shared edge), and at `g_ratio=0.5` this shows up as a real, if
small, boundary-specific accuracy cost: mean `|error|` at smooth-region
window boundaries is ~3.5x larger for `g_ratio=0.5` than `g_ratio=1`
(0.0066 vs 0.0019 ppm), on top of a much larger, clearly-localized jump
right at the sharp west-hotspot seam (rows 98-126, where both windows are
already pinned at `MIN_WINDOW`'s 9-row floor, so `g_ratio=0.5` packs
G=18 bins into 9 rows -- edge bins there get one-sided real-pixel support
instead of two-sided, so they're both more weakly constrained and more
sensitive to exactly where the boundary falls relative to the true peak).
CoV of the raw retrieved value at boundaries showed NO g_ratio-dependent
difference (dominated by the real background gradient, not retrieval
noise) -- the effect only shows up once the real trend is removed (CoV/
mean on `|error|` specifically). The sign pattern is systematic, not
random noise: a window's trailing edge consistently overshoots and the
next window's leading edge consistently undershoots (or vice versa),
tracking the local true-CO2 gradient direction -- classic edge bias of
independently-regularized local regression, and directly explained by the
`exponential`-form prior's own documented behavior (`joint_state.py::
Sa_inv_block`'s docstring): inverting a stationary exponential covariance
gives compensating corner terms but no edge weakening, unlike `tikhonov`'s
natural free-boundary condition. Two candidate fixes were considered --
(1) overlap the windows by a few rows and blend the overlap at stitch
time (cheap, output-only, doesn't touch the Bayesian solve), (2) weakly
couple each window's edge-bin prior to its neighbor's own posterior via a
Gauss-Seidel-style relaxation pass (a real statistical coupling, costs the
current one-shot-per-window parallelism). (1) was implemented; (2) was
not pursued.

**Posterior covariance + window-overlap merging (implemented,
2026-08-21).** Three pieces, in dependency order:

1. **Posterior covariance is now available from every solve.**
   `gauss_newton_state(..., return_cov=True)` returns `(x, S_ret)` with
   `S_ret = inv(A)` at the final Gauss-Newton iteration -- `A` is already
   factored for that iteration's own `np.linalg.solve`, so this is nearly
   free. `StateSpec.cov_for(name, S_ret_scale)` slices one row's own
   physical-units covariance sub-block out of the packed `S_ret_scale`
   (scale-kind rows convert via `prior[i]*prior[j]*S_ret[i,j]`, the
   delta-method Jacobian of `value = prior*x`); `StateSpec.project_cov`
   projects that block onto arbitrary eta positions via the same
   `interp_weights` matrix `interp_to` itself uses, so
   `Var(x_hat(eta)) = W @ Cov_row @ W.T` exactly, including off-diagonal
   terms between query points -- not an approximation.
   `gd_joint_block_whole_slit_sweep.py` now saves `cov` in every window's
   `snapshot()` (coarse and hires both) as a matter of course.

2. **`build_window_tiles(..., overlap=N)`** (new `--overlap` flag, default
   0 = today's exact non-overlapping tiling, unchanged for every existing
   caller). After the usual keystone-driven tiling, each internal boundary
   is widened by `overlap` rows on each side (clamped at the slit's own
   two ends). Because bin count scales with window width
   (`G = round(width/g_ratio)`), a widened window gets proportionally MORE
   bins over its now-wider eta range -- `pixel_density_bin_centers`
   re-places every bin from scratch, so overlapping windows' bins are
   never shared or coincident, only their *row ranges* overlap. Verified
   visually with `scripts/gd_window_overlap_diagram.py` (two independently
   -placed, non-coinciding bin sets shown side-by-side for `overlap=0` vs.
   `overlap=N`).

3. **The information-bearing state, and a separate query/postprocessing
   step.** An initial design directly interpolated each window's own bins
   onto a fixed 1024-detector-row grid and combined overlapping windows
   there (`along_slit_merge.py`, since replaced). That representation
   understated the real correlation structure: within one window, every
   reported row is a linear combination of the SAME G bins (rank ~G, not
   window-width), so adjacent rows are far more correlated than 1024
   independent numbers would imply -- exactly the kind of manufactured
   precision this project's own along-slit uncertainty work is trying to
   avoid. Replaced with a two-module split:
   - `geocarb_gert/along_slit_state.py::stack_windows_along_slit(windows,
     name)` builds the actual retrieved state: every contributing
     window's own bins concatenated, with the REAL joint covariance --
     exactly block-diagonal, since each window was solved fully
     independently (no shared data, no prior term crosses a window
     boundary). Nothing is interpolated or fabricated here; length is
     however many bins were actually retrieved, not tied to any detector-
     row count.
   - `geocarb_gert/along_slit_query.py::query_state(stacked, row_query,
     eta_query)` is the postprocessing step: given any caller-supplied
     grid, returns `(values, cov, n_covering)` -- the FULL covariance at
     that grid, not just a diagonal, via `W @ stacked.cov @ W.T` for one
     interpolation operator `W`. A query point covered by exactly one
     window reproduces `StateSpec.project_cov` exactly (unit-tested). A
     point covered by two overlapping windows gets the inverse-variance-
     optimal combination of their independent estimates, PLUS an explicit
     `disagreement_inflation` term (same idea as `var_excess` below) at
     exactly the multi-window points, since the documented overshoot/
     undershoot bias means two windows can disagree by more than their
     own stated uncertainties predict -- a bias effect, not something
     `W @ cov @ W.T` alone can represent. `n_covering` lets a caller tell
     a single-window point from a merged one from a gap, without
     re-deriving it. A 1024-row curve is now just one particular query a
     caller can make against the real state, not the state itself.

   Empirically validated (matching `overlap=0`/`overlap=2` sweeps at
   `g_ratio=1`, exact prior, `co2plus1pct`): merging overlap-band
   estimates cut mean `|error|` near the base-tiling seams from 0.0169 to
   0.0100 ppm (41%), max `|error|` from 0.176 to 0.113 (36%), and
   whole-slit mean `|error|` from 0.0126 to 0.0108 (14%).

**Not yet done:** `docs/PROJECT_STATUS.md` (this section) is now current,
but none of this is wired into the four existing downstream plotting
scripts (`gd_joint_block_whole_slit_plot.py`, `gd_joint_block_matrix.py::
stitch()`, `gd_retrieval_bingrid_plot.py`,
`gd_retrieval_prior_slit_chi2_compare.py`) -- deliberately scoped out
until the mechanism itself was proven correct. Whether `overlap>0` should
become the new default for production sweeps, or stay an explicit
per-run opt-in, is also still an open decision.

## 7. Heterogeneous surface albedo: feature, bugs found, and what the errors actually look like (2026-08-25)

### 7.1 What was built

Freed `albedo` as a real, jointly-retrieved state row (alongside `co2_ppm`,
`p_surface_hpa`) against a genuinely spatially-varying surface truth scene,
instead of every prior run's single constant scalar albedo per band.
Almost none of this was new engineering -- `geocarb_gert/along_slit_scene.py`
already had a fully-worked heterogeneous-albedo model (4 land-cover
archetypes -- desert/forest/grass/water -- arranged into 13 large-scale
patches with soft tanh edges, plus Gaussian-smoothed fine-scale texture on
top) sitting dormant since 2026-08-18, never turned on in the production
sweep path. This pass:

- Added `--vary-albedo` to `gd_joint_block_whole_slit_sweep.py` (required,
  and validated, whenever `"albedo"` is in `--free`).
- Threaded `band_label`/`surface_fields` through `joint_state.
  state_spec_from_scene` so a free albedo row actually gets added, and
  added the missing imperfect ("structural") surface prior --
  `albedo_for_label_prior`/`SURFACE_FIELDS_PRIOR`/`SURFACE_PRIOR_FIELD_SETS`
  (patches known, fine texture unknown -- the surface-side sibling of the
  atmosphere `structural` prior from Sec.6).
- Shortened the fine-texture correlation length `ALBEDO_CORR_KM` 30km ->
  10km, matching the CO2 hot-spot scale, per user request.
- New free-set `co2p_albedo` (`gd_joint_block_matrix.py::FREE_SETS`) and
  matching `--vary-albedo`/`resolve_existing()` metadata-guard plumbing,
  same pattern as `--overlap`/`--prior-form` in Sec.6.

### 7.2 Three real bugs found while validating this end-to-end

1. **`_make_state_spectrum`'s forward closure never accepted the surface
   argument** `build_forward_state` already knew how to pass whenever a
   `surface`-target row exists -- a hard blocker, not an edge case. Fixed
   in `gd_joint_block_whole_slit_sweep.py`.

2. **`pressure_to_alt_std_atm` silently returned `NaN`** for any
   `p_surface_hpa` above the US Standard Atmosphere's 1013.25 hPa
   sea-level reference (an entirely ordinary value -- 1014.27 hPa is what
   triggered it) -- the tabulated troposphere layer had an upper pressure
   bound with nothing beyond it. `geocarb_gert/scene.py::_std_temperature`
   then turned that `NaN` into **uninitialized-memory garbage** (built
   from `np.empty_like` + a `np.where` chain that never touches an entry
   whose condition is `NaN`, which fails every comparison) instead of
   propagating it -- in the case that surfaced this, the "temperature"
   that came out was literally the leftover bit pattern of the *pressure*
   array computed one line earlier (~650 K after layer-averaging, later
   confirmed as literally `1014.27`, the pressure value itself, before
   averaging). This crashed `gert`'s ABSCO lookup (`int(NaN)`) deep inside
   the RT solver, initially looking like a `gert` bug -- it wasn't; `gert`
   has no forward-model or RT-solver code of its own anywhere in this
   repo, confirmed by grep. Fixed both: extended the troposphere formula
   past the tabulated ceiling (`model_sampler.py::pressure_to_alt_std_atm`,
   `i==0`'s mask no longer upper-bounded), and made `_std_temperature`
   start from `np.full_like(z, np.nan)` so any future unhandled case fails
   loudly instead of silently returning garbage.

3. **SLURM CID-naming mismatch in the sweep launcher**, found running the
   real 30-config sweep (not a code bug, an operational one):
   `submit_retrieval_prior_albedo_v1.sbatch` wrote files as
   `co2p-58-g{G}...` while `gd_joint_block_matrix.py`'s own `config_id()`
   (given `--free co2p_albedo`) looked for `co2p_albedo-58-g{G}...` --
   and separately, `config_id()` elides the `-pf-exact` suffix entirely
   for the (default) `exact` prior, which the sbatch script appended
   unconditionally. `resolve_existing()` correctly decided the array
   job's own pkls "weren't there" and the postprocess step tried to
   regenerate all 30 configs from scratch inside its own tiny 8GB
   allocation -- timed out and OOM'd. Fixed by renaming the 30
   already-computed pkls to match `config_id()`'s real convention (no
   data lost, no recompute) and correcting the sbatch script for future
   reruns.

### 7.3 Sweep results: `retrieval_prior_albedo_{exact,structural}_v1`

Both free `co2_ppm, p_surface_hpa, albedo` jointly, `--vary-albedo
--overlap 2 --prior-form exponential`, across the standard `g_ratio` x
`anchor_density` grid (this was the last sweep to include `g_ratio=0.5` --
see 7.6).

| g_ratio | ad1 (exact) | ad4 | ad16 | ad1 (structural) | ad4 | ad16 |
|---|---|---|---|---|---|---|
| 0.5 | 4.04 | 0.91 | 0.27 | 4.68 | 1.03 | 0.30 |
| 1   | 3.77 | 0.68 | 0.32 | 4.72 | 0.69 | 0.27 |
| 3   | 3.48 | 0.64 | 0.31 | 3.93 | 0.75 | 0.39 |
| 6   | 3.85 | 3.07 | 3.00 | 4.90 | 4.21 | 4.20 |
| 12  | 8.95 | 8.46 | 8.43 | 11.88| 11.06| 10.98|

(CO2 rms, ppm, row-interpolated per `gd_joint_block_matrix.py::stitch()`
-- see 7.4's caveat on this).

Best case (`ad16`, fine `g_ratio`) is ~0.27-0.4 ppm -- roughly 40-50x
worse than `co2p` without albedo at the same settings (Sec.6: ~0.005-0.007
ppm). Freeing a third, only-weakly-constrained jointly-fit parameter costs
real accuracy even in the best case.

**A sharp break at `g6`/`g12`, unrelated to gradual diminishing returns.**
At `g0.5`-`g3`, `anchor_density` gives huge, fast-then-slower gains
(`g1`: 3.77 -> 0.68 -> 0.32, ~5.5x then ~2x). At `g6`/`g12` the curve is
nearly flat from the start (`g12`: 8.95 -> 8.46 -> 8.43, 6% then 0.4%) --
not diminishing returns, a near-immediate plateau. `residRMS` (against
real data, not truth) is also markedly higher at `g6`/`g12` (0.017-0.038)
than at `g0.5`-`g3` (0.0009-0.014), confirming the fit itself is worse
there, not merely biased relative to truth. Consistent with a
degrees-of-freedom collapse once three correlated parameters share too
few, too-wide bins -- `gauss_newton_state(..., return_avk=True)`
(7.5) exists now specifically to test this quantitatively, not yet wired
into `_solve_window`'s saved snapshot.

### 7.4 Native-resolution error methodology

Every number in every `REPORT.md` this whole project (Sec.5, Sec.6, and
7.3 above) is computed by `gd_joint_block_matrix.py::stitch()`, which
`np.interp`s each window's own bin values onto the fixed 1024-row grid
*before* comparing to truth and taking an RMS. That conflates genuine
retrieval error with row-interpolation smearing -- worst at coarse
`g_ratio` (bins sparse, interpolation between them doesn't track true
curvature), negligible at fine `g_ratio` (bins dense).

All analysis below instead uses `geocarb_gert.along_slit_state.
stack_windows_along_slit(windows, name)` directly -- the real,
uninterpolated bin positions/values/covariance, no `query_state`
involved, truth evaluated exactly at each bin's own position (no
interpolation needed on the truth side either, since it's a known
closed-form function). This is the bin-native error the retrieval's own
piecewise-linear forward model can actually be judged against; see the
2026-08-25 conversation this section documents for the full derivation
(the state's own bin value is a node of a piecewise-linear interpolant,
not a footprint average -- error is therefore only rigorously defined
pointwise, at the bin's own position).

### 7.5 CO2 error vs. local truth curvature -- theory-consistent, but swamped at coarse resolution

Piecewise-linear interpolation error is a curvature (2nd-derivative)
effect, not a gradient (1st-derivative) one (`error ~ kappa * h^2 / 8`,
classical). Checked directly: split native CO2 bins into curvature
quartiles (`retrieval_prior_albedo_exact_v1`, `co2p_albedo-58-g1-ad16`):

- `g1-ad16`: high-curvature bins have **1.16x** the mean `|error|` of
  low-curvature bins -- small, but correctly signed.
- `g12-ad16`: **0.61x** -- inverted. Overall error there is 25-35x
  larger than at `g1` and dominated entirely by the DOF-collapse
  mechanism above (7.3), which swamps and even reverses the smaller
  curvature-driven signal (plausibly because high-curvature = hot-spot
  peak = more absorption signal = partially offsetting SNR benefit).

Conclusion: finer `g_ratio` doesn't specifically fix error near sharp
features -- it fixes the dominant DOF problem everywhere, which happens
to unmask a real but secondary curvature effect only once the dominant
one is gone.

### 7.6 `anchor_density`'s real mechanism: not noise averaging, a stepping-bias fix

Checked whether `anchor_density` mainly shrinks posterior uncertainty
(`S_ret`, more independent measurements pooled) as originally assumed.
It does not: mean posterior sigma for `co2_ppm` at `g1` stays flat across
`ad1/4/16` (1.05, 1.02, 1.02 ppm) while actual mean `|error|` drops
sharply (1.65 -> 0.38 -> 0.15 ppm) and `residRMS` drops in lockstep
(0.0137 -> 0.0037 -> 0.00094).

The real mechanism, already documented in `build_forward_state`'s own
docstring (the "~(1/4)|f'|h stepping error... a property of the SAMPLING,
not of the state"): `anchor_density` doesn't add real data at all (`y_true`
is the fixed real-pixel image, independent of it) -- it controls how
finely `nearest_bin_scene` approximates the continuous piecewise-linear
state when assigning real pixels to predicted radiance. Coarse anchor
spacing snaps physically-distinct pixels to the same anchor's prediction,
dumping their genuine difference into the residual as if it were noise.
Finer spacing lets the model actually explain that real pixel-to-pixel
variation as signal. Confirms as a genuine *bias* fix, not noise
averaging: at `ad1`, mean `|error|` (1.65) exceeds the posterior's own
claimed sigma (1.05) -- the fit is biased beyond what its own error bars
represent; at `ad16`, error (0.15) falls well below sigma (1.02) -- the
bias is gone and what remains is close to the genuine noise/prior floor
`S_ret` was always describing correctly. This is a third, distinct error
axis from `g_ratio` (model capacity / smoothing error) and from
prior-driven regularization.

**User decision (2026-08-25): drop `g_ratio=0.5` from future sweeps.**
It is the one axis point asking for state resolution finer than the
detector's own native pixel sampling (`g_ratio<1` means `G > width`, more
than one bin per detector row) -- extra degrees of freedom no single
row's own data can independently constrain, filled almost entirely by
the prior's own correlation structure rather than real information (low
`trace(AVK)` expected there, not yet directly measured). Default sweep
axis going forward: `g_ratio in {1, 3, 6, 12}`.

### 7.7 Albedo error is sharply concentrated at patch boundaries

Patch edges (tanh-smoothed over only `ALBEDO_EDGE_KM=3km`, far sharper
than a Gaussian CO2 hot spot) are the cleanest confirmation of the
smoothing-error story anywhere in this project:

| config | overall mean\|err\| | near boundary (<=15km) | far (>15km) | ratio |
|---|---|---|---|---|
| `g1-ad16` (exact) | 0.0017 (0.9% frac.) | 0.0090 | 0.0007 | **12.8x** |
| `g12-ad16` (exact) | 0.0058 (3.7% frac.) | 0.0154 | 0.0044 | **3.5x** |

Away from any land-cover transition, albedo recovers to well under 1%
fractional error at fine resolution -- genuinely accurate. Essentially
all the meaningful error lives in a narrow band straddling each of the
13 patch edges.

**This leaks into CO2** despite CO2's own truth field having nothing
special at patch-edge positions -- the joint fit can partially "explain
away" an albedo-driven model mismatch by nudging the correlated CO2
estimate instead of leaving it as unexplained residual:

| config | CO2 overall mean\|err\| | near patch edge | far | ratio |
|---|---|---|---|---|
| `g1-ad16` | 0.147 ppm | 0.579 ppm | 0.080 ppm | **7.2x** |
| `g12-ad16`| 3.70 ppm | 6.78 ppm | 3.19 ppm | **2.1x** |

**Open, unexplained**: the `structural`-prior run showed *smaller*
absolute albedo error than `exact` at the same `g1-ad16` config (0.0002
vs 0.0017, boundary ratio 2.1x vs 12.8x) -- surprising, since the
imperfect prior should make things harder. Not chased down; possibly the
smoother structural prior acts as extra regularization that happens to
track the local average better than a strongly-pulled exact prior can
when the fine-texture realization aliases against the bin grid.

### 7.8 Cross-parameter error correlation: two different objects, easy to conflate

Checked whether CO2 and albedo errors are correlated near boundaries, two
ways:

- **Empirical, across windows** (per-window mean signed error, `g1-ad16`,
  exact prior): far from any boundary, `corr(co2_err, albedo_err) =
  +0.79` (strong). Near boundaries, `+0.02` (n=10 windows -- small
  sample, treat the near-zero reading as "much weaker," not confidently
  "none").
- **Posterior-claimed, within one window's own fit** (`S_ret`'s
  `co2 x albedo` cross-covariance block, converted to a correlation):
  essentially zero everywhere, both near and far (~0.002-0.003).

These are not the same question and shouldn't be expected to agree: the
posterior cross-covariance describes *within-one-window* noise
sensitivity (how co2 and albedo would covary if that same window were
refit with different noise) -- small here because `Sa_inv` has no
cross-row coupling and the data-driven `K^T Sy_inv K` coupling is
apparently weak at the bin-pair level. The empirical cross-window
correlation describes something the posterior was never built to
represent: whether some windows are systematically easier or harder for
*both* parameters at once (shared SNR/data-quality driver). The strong
far-field empirical correlation is real and unmodeled by `S_ret`; the
near-zero posterior cross-covariance is not wrong, it is just answering
a different question.

### 7.9 Error sign and magnitude are a near-linear function of the true jump size

The strongest, cleanest relationship found in this whole investigation.
For each of the 12 internal patch boundaries, correlated the true
reflectance jump (`_BAND_ALBEDO[type_next] - _BAND_ALBEDO[type_prev]`)
against the mean signed albedo error in the 10km bands immediately before
and after that boundary (`g1-ad16`, exact prior):

- `corr(jump, error_before) = +0.91`; `corr(jump, error_after) = -0.96`.
- `corr(|jump|, |error_before|) = +0.66`; `corr(|jump|, |error_after|) =
  +0.85`.

**Sign**: the bin just before a boundary is biased *toward* the value on
the other side (same sign as the jump); the bin just after is biased
back toward the value it just left (opposite sign). A textbook
leakage/blur pattern, symmetric about the boundary -- both the ILS/PSF
genuinely mixing radiance from both sides of a sharp transition, and the
piecewise-linear state's own inability to represent a step, contribute.
**Magnitude**: scales close to linearly with how sharp the true
transition is (desert<->forest, the largest jump at 0.33, gives the two
largest errors of any boundary; forest<->grass, jump 0.06, gives errors
5-10x smaller) -- strong evidence this is a genuine smoothing artifact,
not SNR-driven scatter (which wouldn't care how big the true step is).

Because sign and magnitude are this predictable from information already
available (the truth-scene design, or estimable from the retrieved levels
on either side of a detected boundary), this is a strong candidate for a
direct bias correction -- the same spirit as the window-overlap merge fix
for the analogous CO2 window-boundary bias (Sec.6), not yet built.

### 7.10 New tooling: averaging kernel, not yet wired into production

`geocarb_gert.joint_state.gauss_newton_state(..., return_avk=True)`
returns the Rodgers averaging kernel `AVK = Gain @ K` (`Gain = S_ret @
K^T @ Sy_inv`), one extra matmul reusing quantities the last iteration
already built -- no extra solve. Row `k` is the linear combination of the
TRUE state (at every other free element's own position) retrieved
element `k` actually reflects; `trace(AVK)` is the degrees-of-freedom-
for-signal for the whole solve. Validated on a synthetic well-/poorly-
constrained pair (`trace(AVK)/n` -> 1.0 and -> 0.0 respectively, exactly
as theory predicts). Explicitly does NOT see true sub-bin structure --
`AVK = Gain @ K_fine @ W`, and the trailing interpolation-weight matrix
`W` means it can only ever express sensitivity as filtered through the
same piecewise-linear model the retrieval already assumes. Quantifying
that gap (the Rodgers "smoothing error") would need a `K_fine` (Jacobian
w.r.t. the true field at every fine position, not just packed state
elements) that doesn't exist yet -- a real, separate undertaking, though
`S_true_hires` (the other ingredient) is unusually cheap here since the
truth scene's own statistics (`ALBEDO_COV`/`ALBEDO_CORR_KM`, hot-spot
width/amplitude) are already known exactly.

**Update (2026-08-25):** `return_avk` is now wired into `_solve_window`
for both coarse and hires solves, saving `avk`/`dof` into every window's
snapshot, with `dof_frac` (`trace(avk)/n_free`, averaged across a
config's own windows) surfaced as its own `REPORT.md` column in
`gd_joint_block_matrix.py`. Validated: a well-constrained `co2p` window
gives `dof/n_free ~= 0.9995`; a full `co2p`-only `g_ratio=12` run reports
`DOF frac = 1.000` in `REPORT.md`, consistent with the collapse being
specific to adding albedo as a third weakly-constrained free row, not an
inherent property of coarse `g_ratio` alone -- the DOF-collapse
hypothesis above is no longer argued only from indirect evidence.

## 8. Prior-information-driven bin placement: a synthetic demo, and a real negative result (2026-08-25)

Motivated by a conversation about a shared, information-weighted state
grid: today `albedo`'s bins are placed by `albedo_positions_for`, a flat
`surface_density=3` uniform oversampling of the gas grid's own span --
three times as many bins as `co2_ppm`/`p_surface_hpa`, everywhere,
whether or not anything actually supports that much local resolution. A
real mission would often have an external, spatially-resolved product
(MODIS albedo, ~500m) usable as a genuinely informative prior -- worth
concentrating retrieval resolution there rather than spreading it
uniformly.

**Built** (no core-architecture changes needed -- `ParamSpec.positions`
already accepts arbitrary irregular arrays under the `exponential` prior
form, today's default):
- `geocarb_gert.joint_state.information_weighted_bin_centers(eta_lo,
  eta_hi, G, weight_fn)` -- generalizes `pixel_density_bin_centers`'s
  equal-population quantile placement from raw pixel-density SAMPLES to
  an arbitrary density FUNCTION, via inverse-CDF sampling (`np.quantile`
  has no weight argument). Unit-tested: a constant `weight_fn` reproduces
  a uniform grid exactly; a peaked one concentrates bins ~9.7x denser at
  the peak than at the window edges.
- `geocarb_gert.along_slit_scene.albedo_info_density(x_km)` -- a
  SYNTHETIC stand-in for "how much a real fine-resolution external
  product would inform us here," peaked at each `SURFACE_PATCHES`
  boundary (a real product would resolve a genuine, sharp transition
  there), floor elsewhere. Not derived from any real external data --
  demonstrates the mechanism only. Deliberately correlates with the
  truth's own patch structure, since a real product genuinely would too.
- `scripts/gd_information_density_bins_demo.py` -- a standalone,
  self-contained demo (not wired into the production CLI): a bin-
  placement diagram (same convention as `gd_window_overlap_diagram.py`),
  and a before/after native-resolution accuracy comparison at a FIXED
  total bin budget (redistributed, not grown), reusing `state_spec_from_
  scene`'s existing `surface_positions` override rather than duplicating
  the production solve path.

**Result -- a real negative finding, not a bug.** At a properly-sized
test window (rows 884-924, G=40 albedo bins -- a narrower window with
only G=7 gave an unusably small, noisy comparison, itself a small
illustration of the same "too few bins to say anything meaningful"
problem this feature exists to address): the redistribution mechanism
works exactly as intended (bins visibly and dramatically cluster at the
boundary), but native-resolution accuracy right at the boundary got
WORSE, not better (mean `|error|` 0.0122 -> 0.0185), while the now-sparser
far field improved (0.0011 -> 0.0003). Ruled out one obvious confound --
the tightly-packed near-boundary bins are not anchor-starved to the
`ad1`-pathology degree (Sec.7.6): min anchors-per-bin-gap only drops from
6 to 3, not to 0. Leading, unconfirmed hypothesis: packing bins closer
than `ALBEDO_CORR_KM` (10km) apart makes them strongly prior-correlated
with each other, which is not the same as the data independently
resolving them, and may hurt Gauss-Newton conditioning exactly where the
extra resolution was meant to help. The natural next diagnostic --
comparing `trace(AVK)` per bin between the two schemes (Sec.7.10) -- is
not yet done.

**Takeaway**: naively moving resolution toward "where prior information
exists" without also confirming the *data* (anchor grid, and possibly the
prior's own correlation length relative to the new bin spacing) can
actually support that redistribution is not automatically a win, and can
make things worse exactly where you meant to help. Worth treating as a
real constraint on any future adaptive-resolution scheme, not a detail to
paper over.

## 9. A common grid across rows, with per-row interpolation kind (2026-08-25)

Follow-on to Sec.8, per user direction: albedo's own separate,
information-weighted grid from Sec.8 gave way to **one shared grid across
every jointly-free row** (`co2_ppm`, `p_surface_hpa`, `albedo` all on the
SAME bin positions), plus the ability for a row to pick a different
`interp1d` kind on that shared grid (`"nearest"` for a hard-edged feature
like a patch boundary, `"linear"` for a smooth one like a CO2 hot spot).

**Built, no core-architecture changes beyond what already existed:**
- `ParamSpec.state_interp: str | None = None` -- a row's own interpolation
  kind override; `None` (every row, until one opts in) falls back to
  whatever kind the caller passes to `interp_to`/`interp_weights`/
  `build_forward_state`/`linearize`, so every existing call site is
  byte-identical unless a row explicitly sets its own. `StateSpec.interp_
  to`/`interp_weights` resolve it with one line each (`kind = p.state_
  interp if p.state_interp is not None else state_interp`); `build_
  forward_state`/`jac.linearize` needed NO changes at all, since they
  already just forward their own argument into those two methods.
- `state_spec_from_scene`'s surface rows now default to the SAME
  `bin_centers` atmosphere rows use (previously `albedo_positions_for(
  bin_centers, surface_density)`, a separate 3x-denser grid) -- a real,
  deliberate behaviour change for any caller freeing a surface row without
  passing `surface_positions` explicitly, including the production sweep
  path. Already-banked results with a free albedo row (Sec.7's sweep) are
  unaffected (saved data doesn't change), but a bare rerun of the same
  command now places albedo on the shared grid instead of its own. New
  `row_state_interp: dict | None` parameter sets `ParamSpec.state_interp`
  on named rows at construction.
- `combined_information_weighted_bin_centers(eta_lo, eta_hi, G,
  weight_fns: dict)` -- the shared grid is placed by the ELEMENTWISE MAX
  across every row's own weight function, so it refines wherever ANY row
  wants it rather than diluting toward an average. Thin wrapper around
  `information_weighted_bin_centers` (Sec.8), no duplicated logic.

Validated: with every `state_interp` left `None`, `interp_to`/`interp_
weights` are provably unchanged (direct comparison against the pure-global-
kind call); setting one row to `"nearest"` while another stays `None`
(falling back to a global `"linear"`) gives each row its own, independently
correct interpolation. `state_spec_from_scene` confirmed to actually share
positions across rows now (`co2_ppm`/`albedo` positions `np.array_equal`)
and to honor `row_state_interp`.

**Result, extending Sec.8's demo with a third scheme (`--shared-grid` on
`gd_information_density_bins_demo.py`)**: co2/p_surface/albedo on one
combined-density shared grid, albedo alone `state_interp="nearest"`. At two
different boundary windows (rows 884-924 and rows 418-438), the near/far
error asymmetry that both linear-interpolation schemes showed (worse near
the boundary, better far -- the leakage/blur pattern from Sec.7.9) mostly
**disappears** under `"nearest"` (rows 418-438: near 0.0065 vs far 0.0064,
essentially equal) -- but overall accuracy is not better, it's a uniformly
elevated 0.0065 against the flat baseline's 0.0022. This is the tradeoff
named in conversation before building it, now measured directly:
piecewise-constant interpolation removes the specific asymmetric leakage
`"linear"` produces at a hard edge, but is a cruder approximation
everywhere else in the window that isn't a hard edge, and that costs more
than the leakage fix saves, at least in this configuration. Not yet tried:
`"nearest"` restricted to just the near-boundary bins with `"linear"`
elsewhere (mixing kinds along a single row, not just across rows) -- the
per-row field as built doesn't support per-BIN kind, only per-row, so that
would need a further extension, not implemented here.

## 10. Sub-bin anomaly: additive high-resolution sampling at anchor points (2026-08-28)

A MULTIPLICATIVE sub-bin scheme (`g(eta) * sum_k W[eta,k]*bin_value_k/
mean_g_k`) was built first, tested exhaustively (exact-truth oracle, an
independent-noise-statistics-matched "real prior" oracle, exact vs.
structural surface priors, 10km vs. 500m albedo correlation length), and
found to consistently make native-resolution accuracy WORSE than plain
linear interpolation, never better -- root-caused (a per-bin diagnostic
comparing scoring against the bin-center point value vs. the footprint
mean, plus the averaging kernel) to `mean_g_k` being a NEIGHBOR-BLENDING
(tent-weighted) smoothing operator, not a point-sample one: for a uniform
grid, `mean_g_k = (1/6)*truth_{k-1} + (2/3)*truth_k + (1/6)*truth_{k+1}`,
which equals the bin-center value only when the bin-center sequence has
zero curvature -- verified against `bin_footprint_means`'s own actual
output, not just derived on paper. That design and its own result writeup
are preserved on git branch `sub-bin-modulation-multiplicative`, NOT on
`main` -- entirely superseded by the design below.

### 10.1 Redesign: additive, not multiplicative

Restated goal (user): allow higher-resolution sampling of a profile `g` at
anchor points when it is available, with the requirement that this
reduces EXACTLY to today's plain linear interpolation whenever `g` is
itself linear between bin centers -- not just when it is constant, which
is all the multiplicative scheme could ever guarantee.

```
value(eta) = sum_k(W[eta,k]*bin_value_k)  +  [ g(eta) - sum_k(W[eta,k]*g_k) ]
```

`g_k = g(bin_center_k)`, a single point sample -- no footprint-averaging,
no fine grid, no `mean_g_k`. Two properties, both checked directly against
the actual code, not just derived:

- **Exact reduction**: `g` piecewise-linear on the row's own grid ->
  the bracketed correction is identically zero at every `eta`, not just
  the nodes -- `value(eta)` collapses to exactly today's plain interpolant
  (verified to machine precision, arbitrary non-node `eta`).
- **`value(bin_center_k) == bin_value_k` exactly, for ANY `g`** -- a bin's
  own retrieved value keeps its literal point-sample meaning, unlike the
  old scheme, where `mean_g_k` never quite equaled `bin_value_k`. This is
  also what makes `_native_error_summary`'s existing scoring convention
  (retrieved value vs. truth at the bin's own center) unambiguous again --
  no footprint-mean-vs-point-value question to resolve.

**The state Jacobian `K` is completely untouched**: `d(value)/d(bin_value_k)
= W[eta,k]`, unchanged, because `g_k` is a fixed constant that never
depends on the retrieved state. `StateSpec.interp_weights` needed NO code
change at all (confirmed: byte-identical output with `sub_bin_anomaly` set
vs. unset on the same row); `jacobians.linearize`'s existing loop that
builds `K` needed no change either. Only `StateSpec.interp_to` (the anchor
VALUES fed to the forward model) picked up a new branch -- and because
`build_forward_state`/`jacobians.linearize` both already build every
per-anchor state entirely through `interp_to`, neither needed further
changes to pick up the correction in the actually-rendered radiance. Built:
`SubBinAnomaly`/`ParamSpec.sub_bin_anomaly`, `state_spec_from_scene`'s
`row_sub_bin_anomaly`, and (same overall shape as before, but now a plain
LINEAR sensitivity with no quotient rule) `g_positions`/`g_cov` ->
`jacobians._g_anomaly_sensitivity` -> `K_g` -> `gauss_newton_state`'s
`g_cov` parameter propagating `g`'s own uncertainty into the posterior via
`Gain @ K_g @ g_cov @ K_g.T @ Gain.T`, strict no-op when `g_cov` is
`None`/empty.

**Validated (unit level, all checked directly)**: `sub_bin_anomaly=None`
no-op; `interp_weights` byte-identical with/without a row's
`sub_bin_anomaly` set; the exact-reduction property (machine precision, at
arbitrary non-node `eta`); `value(bin_center_k) == bin_value_k` exactly for
a genuinely non-linear `g`; `g_cov=None` reproduces `gauss_newton_state`'s
`return_cov`/`return_avk` outputs byte-identically; a nonzero `g_cov`
strictly increases reported posterior variance (`S_g_total` symmetric PSD,
positive diagonal) on a synthetic problem.

**Also confirmed directly (user requirement)**: with `state_spec_from_scene`
called the normal way (`surface_positions=None`), `co2_ppm`/`p_surface_hpa`/
`albedo` all solve on the exact same shared bin grid -- `sub_bin_anomaly`
only adds resolution at the anchor/forward-model level via `g_fn`, never a
second, separate retrieval grid for albedo.

### 10.2 Single-window result (rows 418-438, structural prior, 500m
correlation length, g_ratio=1) -- a mixed signal

| scheme | overall | near (<=15km) | far (>15km) |
|---|---|---|---|
| no anomaly (plain grid) | 0.0274 | 0.0289 | 0.0257 |
| anomaly(g=truth) | 0.0279 | 0.0242 | 0.0321 |
| anomaly(g=prior) | 0.0322 | 0.0289 | 0.0358 |

Near-boundary error genuinely IMPROVED with the exact-truth oracle (0.0242
vs. 0.0289, ~16%) -- a real win, right where a curved feature should
matter most, and something the multiplicative scheme never produced
anywhere. `g=truth` and `g=prior` are also now clearly DISTINGUISHABLE
(0.0279 vs. 0.0322) -- unlike the multiplicative scheme, where an exact
oracle and an uncorrelated-but-statistically-matched one performed
identically. But far-field got WORSE with `g=truth` (0.0321 vs. 0.0257),
not yet explained at the time -- candidates noted then: RT nonlinearity in
albedo, cross-talk with the jointly-free `co2_ppm`/`p_surface_hpa` rows, or
anchor-assignment "stepping" discretization (Sec.7.6). This window used
`g_ratio=1`, a deliberately fine bin grid built for the earlier
multiplicative-scheme investigation -- see 10.3 for why that turned out to
matter.

### 10.3 Whole-slit sweep (production g_ratio=3, all 58 windows) -- a decisive win

Built `--sub-bin-anomaly {none,truth,prior}` on the PRODUCTION sweep tool
(`gd_joint_block_whole_slit_sweep.py`, not a demo script) and
`scripts/submit_sub_bin_anomaly_v1.sbatch` (3 setups x anchor_density
{1,4,16}, 9 SLURM array tasks, `--prior-fields structural`, full 58-window
slit). All 9 tasks completed cleanly (58/58 windows each, no errors,
15-58 min per task). Scored with a new `scripts/gd_sub_bin_anomaly_sweep_
score.py`, generalizing the single-window near/far split to the whole
slit (near = within 15km of ANY of the 12 internal `als.SURFACE_PATCHES`
boundaries, not just one):

| setup | anchor_density | overall | near (<=15km) | far |
|---|---|---|---|---|
| none  | 1  | 0.0187 | 0.0207 | 0.0185 |
| none  | 4  | 0.0186 | 0.0210 | 0.0183 |
| none  | 16 | 0.0187 | 0.0212 | 0.0183 |
| truth | 1  | 0.0076 | 0.0057 | 0.0079 |
| truth | 4  | **0.0006** | 0.0005 | 0.0007 |
| truth | 16 | **0.0005** | 0.0003 | 0.0005 |
| prior | 1  | 0.0247 | 0.0201 | 0.0253 |
| prior | 4  | 0.0257 | 0.0210 | 0.0263 |
| prior | 16 | 0.0257 | 0.0211 | 0.0263 |

At the PRODUCTION default `g_ratio=3` (bins ~3x wider than the tight
`g_ratio=1` single-window test), the exact-truth oracle is a dramatic win
-- roughly a 30x error reduction at `anchor_density>=4` (0.0187 -> 0.0005-
0.0006), improving monotonically with anchor density exactly as expected
(more anchors -> the mechanism can sample `g`'s own resolution more
finely; `anchor_density=1` gives only a modest win, 0.0076, since there is
little room between one anchor per row for `g` to add texture at all).
`g=prior` is consistently WORSE than baseline across every anchor density
(~0.025 vs. ~0.0187, roughly flat) -- the mechanism is genuinely using
`g`'s actual content, not just benefiting from "more texture, any
texture," which is the discriminating signature a real refinement should
show and the multiplicative scheme never showed.

**Reconciling 10.2 and 10.3**: the single g_ratio=1 window's mixed result
(near improved, far got worse) now reads as a scale-sensitivity artifact
of that unusually fine bin grid, not a property of the mechanism itself --
at bins genuinely wide relative to the 500m correlation length (the
production default), there is real, exploitable sub-bin structure for `g`
to add, and the mechanism delivers a large, honest, content-sensitive
improvement. The far-field degradation noted in 10.2 as "not yet
explained" was not chased further given how decisively 10.3 resolves the
open question -- worth returning to only if a future run at very fine
g_ratio needs it.

### 10.4 CO2 side effect: sub_bin_anomaly cleans up cross-talk it never targeted

`co2_ppm` and `albedo` are jointly free/solved in every one of these
9 configs (`--free co2_ppm,p_surface_hpa,albedo`), and `sub_bin_anomaly`
was only ever set on `albedo`'s own row -- but scoring `co2_ppm` the same
way (same 9 pkls, same `stack_windows_along_slit`, truth = `als.xco2_
ppm`, near/far against the same 12 patch boundaries) shows a real,
substantial side effect:

| setup | anchor_density | overall (ppm) | near (<=15km) | far | rel. err |
|---|---|---|---|---|---|
| none  | 1  | 17.43 | 54.57 | 12.27 | 4.19% |
| none  | 4  | 18.27 | 52.69 | 13.49 | 4.40% |
| none  | 16 | 18.43 | 53.08 | 13.62 | 4.43% |
| truth | 1  |  9.50 | 12.38 |  9.10 | 2.29% |
| truth | 4  |  3.33 |  3.92 |  3.24 | 0.80% |
| truth | 16 | **1.48** | **1.21** | 1.51 | **0.36%** |
| prior | 1  | 24.97 | 38.97 | 23.03 | 6.01% |
| prior | 4  | 23.66 | 31.10 | 22.62 | 5.69% |
| prior | 16 | 24.05 | 33.00 | 22.81 | 5.79% |

Baseline CO2 error is dominated by exactly the patch-boundary leakage
Sec.7.8 documented: near-boundary error (~53-55 ppm) is ~4x the far-field
number (~12-14 ppm) -- an imperfectly-represented albedo boundary bleeding
into the jointly-solved CO2 retrieval right where the surface changes.
Giving albedo an accurate high-resolution `g` (truth) collapses this: CO2
near-boundary error drops from ~55 ppm to 1.21 ppm at `anchor_density=16`
(~45x), overall relative error from 4.4% to 0.36%, and the near/far GAP
essentially disappears (1.21 vs. 1.51 -- comparable, unlike baseline's 4x
split) -- once albedo's own forward-model value is accurate at high
resolution, there is much less unmodeled surface signal left for CO2 to
wrongly absorb near a boundary. `g=prior` degrades CO2 too (~24 ppm vs.
baseline's ~18 ppm), the same uncorrelated-texture penalty propagating
through the joint fit into a row `sub_bin_anomaly` was never applied to.
So the mechanism's real value here is larger than the albedo-only numbers
in 10.3 suggest -- it substantially cleans up known CO2-albedo cross-talk
as a side effect, not just albedo's own accuracy.

## 11. Consistency check: albedo frozen at truth, only co2/p_surface free (2026-08-29)

User request: run a whole-slit sweep with albedo FROZEN at the true
per-position values (not part of the free/retrieved state), only
`co2_ppm`/`p_surface_hpa` free, as a regression check that the sub_bin_
anomaly-era codebase still reproduces sensible, historically-consistent
behavior for this simpler configuration.

### 11.1 A real gap this surfaced, fixed

`gd_joint_block_whole_slit_sweep.py::_solve_window` gated `band_label`
(whether a surface `albedo` row is added to the `StateSpec` AT ALL) on
`"albedo" in free` alone: `band_label = GEOCARB_BANDS[FPA][0] if "albedo"
in free else None`. Every prior sweep either froze albedo AND rendered it
constant (no `--vary-albedo`, matched truth/retrieval by construction) or
freed albedo AND varied it (`--vary-albedo` + `"albedo" in --free`,
enforced by `main()`'s own validation) -- the combination this check
needs, "vary the TRUTH but freeze the RETRIEVAL's albedo row at that real
per-position truth," was never reachable: `--vary-albedo` without freeing
albedo silently fell back to `build_forward_state`'s single fixed scalar
`_SWEEP["albedo"]` for every anchor, mismatching a truth scene that
genuinely varies along the slit -- structurally the same class of bug as
the original 2026-08-18 CAVEAT (Sec.7.2), just the mirror-image case,
never exercised until now. Fixed: `band_label` is now set whenever
`--vary-albedo` is on, regardless of `--free` -- a frozen row's own prior
(default `surface_fields=als.SURFACE_FIELDS`, exact truth) is then
genuinely the true per-bin-center value. Verified directly on a smoke-test
window: `albedo free: False`, `max|albedo value - truth| == 0.0`.

### 11.2 Control check: the pipeline itself is sound

Before trusting any number from the fix above, ran the ORIGINAL "co2p"
configuration this whole heterogeneous-surface feature was built on top
of -- no `--vary-albedo` at all (constant truth albedo, no state row,
matching every pre-2026-08-17 result) -- on a 6-window subset (`g_ratio=3,
anchor_density=16`): `resid_hires_rms` ~0.0 on every window, CO2
`mean|err|` = **0.0055 ppm (0.0013% relative)**, matching Sec.6's
historical closed-regression baseline (~0.005-0.007 ppm) almost exactly.
Confirms the core Jacobian/frozen-row machinery is unaffected by every
change this session made -- the numbers below are real, not a regression.

### 11.3 Whole-slit result: frozen-at-truth is NOT an accuracy ceiling

Full 58-window sweep, `g_ratio=3`, `--vary-albedo`, `--prior-fields exact`
(default -- co2/p_surface priors ALSO exact truth here, more informative
than every `sub_bin_anomaly_v1` config in Sec.10, which used `structural`),
albedo frozen at truth, `anchor_density in {1,4,16}`:

| row | anchor_density | overall | near (<=15km) | far | rel. err |
|---|---|---|---|---|---|
| co2_ppm | 1  | 51.85 ppm | 72.38 | 49.00 | 12.48% |
| co2_ppm | 4  | 53.74 ppm | 82.70 | 49.71 | 12.93% |
| co2_ppm | 16 | 54.00 ppm | 84.56 | 49.76 | 12.99% |
| p_surface_hpa | 1  | 43.48 hPa | 49.13 | 42.69 | 4.49% |
| p_surface_hpa | 4  | 44.67 hPa | 57.04 | 42.95 | 4.61% |
| p_surface_hpa | 16 | 44.35 hPa | 54.31 | 42.96 | 4.58% |
| albedo | 1, 4, 16 | 0.0 (exact, by construction) | 0.0 | 0.0 | 0.0% |

Despite exact priors AND albedo pinned to the exact truth value at every
bin center, CO2 error here (~52-54 ppm, ~13%) is **~3x WORSE** than
`sub_bin_anomaly_v1`'s `none` baseline in Sec.10.3/10.4 (~18 ppm, ~4.4%)
-- which had IMPERFECT (`structural`) priors AND a free (not frozen)
albedo row starting from that same imperfect prior. Root cause, not a
bug (11.2 rules that out): **"frozen at truth values" only pins albedo at
the true value at each coarse bin center -- it does not give the forward
model the true CONTINUOUS field.** Between bin centers a frozen row is
still just the plain piecewise-linear interpolant (no `sub_bin_anomaly`
was set here), which at `g_ratio=3` cannot represent the sharp patch
boundaries OR the pervasive 500m fine texture (`ALBEDO_COV=10%` exists
everywhere, not just at boundaries -- consistent with far-field CO2 error
here being nearly as large as near-boundary, unlike Sec.7.7/7.8's own
boundary-concentrated leakage). A FROZEN row has no way to compensate --
it is locked to the point-truth value whether or not that is what best
explains the actual radiance. A FREE row, even starting from an imperfect
prior, can adjust `bin_value_k` to whatever locally minimizes the real
data residual, which (Sec.10's own footprint-mean-vs-point-value finding)
is generally NOT the point-truth value once real sub-bin structure exists
-- and that compensating freedom matters more here than starting from the
exactly-correct value does.

Practical implication: at production `g_ratio=3`, "if only we knew albedo
exactly" is not the right mental model for an accuracy ceiling -- a
coarse, non-adjustable representation of real sub-bin structure can hurt
the jointly-fit rows MORE than an adjustable, imperfectly-informed one.
This is also a natural next check, not yet run: freeze albedo at truth
AND give it `sub_bin_anomaly(g=truth)` too, to see whether that closes
most of this gap by letting even a frozen row represent the true
continuous field between bin centers.

Also note `anchor_density` barely moves these numbers (unlike Sec.10.3's
free-albedo runs, where `g=truth` improved sharply with anchor density) --
consistent with the error being dominated by the STATE's own coarse bin
grid (unaffected by anchor sampling) rather than by how finely the
forward model samples between anchors.

## 12. Why does g_ratio=1 give WORSE CO2 than g_ratio=3? A bug hunt that resolved to a representability-gap artifact (2026-08-30/31)

The whole-slit sweep in Sec.11's own config family (co2/p_surface/albedo
jointly free, structural prior, resolution-matched truth at
`anchor_density=4`) produced a genuinely surprising result: `g_ratio=1`
(3x more bins than `g_ratio=3`) gave a WORSE CO2 retrieval, not a better
one -- 32.78 ppm vs. 17.49 ppm, even with the retrieval's own anchor
sampling matched exactly to the truth's own resolution. The user's
reaction -- "this feels like a potential code bug" -- launched a
systematic elimination process, described here in the order it was
actually run, because each ruled-out hypothesis is itself informative.

### 12.1 Ruled out: DOF collapse

`trace(AVK)`/`dof_frac` (already saved via `return_avk=True`) stayed high
at `g_ratio=1` (0.960, vs. 0.999 at `g_ratio=3`) -- nowhere near the
collapse-toward-zero Sec.7.3's own g6/g12 mechanism produces. The
residual against the ACTUAL rendered data was also BETTER at `g_ratio=1`
(mean `resid_hires_rms` 0.139 vs. 0.225) -- a finer-bin model fitting the
real radiance more closely, exactly as expected from more free
parameters, with no sign of degenerate/ill-conditioned fitting.

### 12.2 Ruled out: prior tie-breaking a near-degenerate solution

Rerunning `g_ratio=1` with `--prior-fields exact` (instead of
`structural`) changed almost nothing: CO2 32.78 -> 32.52 ppm, albedo
0.0103 -> 0.0104, `dof_frac`/`resid_rms` unchanged to 3 decimal places.
If an imperfect prior were tie-breaking among several near-equally-data-
consistent combinations badly, an exact prior should have fixed most of
it. It didn't.

### 12.3 Ruled out: an analytic Jacobian bug at high bin density

`gd_jacobian_validate.py --h-scan` on the same window (rows 890-920) at
both `g_ratio=1` and `g_ratio=3`: `co2_ppm`/`albedo` columns agree with
finite differences to ~1e-9 to 1e-13 relative L2 (cosine 1.000000000) at
every `h`, identically at both resolutions; `p_surface_hpa` shows the
ordinary truncation-vs-roundoff tradeoff (best at `h=1e-3`, degrading
toward `h=1e-6` as `~eps/h` roundoff dominates), again identically in
magnitude at both `g_ratio`. No floor, no resolution-dependent
degradation -- the Jacobian is correct at both bin densities.

### 12.4 Ruled out: correctable cross-parameter leakage

Posterior cross-covariance between `co2_ppm` and `albedo`/`p_surface_hpa`
is real and substantial (mean \|corr\| 0.32-0.68) but not dramatically
stronger at `g_ratio=1` than `g_ratio=3`. More decisively: using the
retrieval's OWN reported cross-covariance as a proper Rodgers-style
correction operator (`co2_corrected = co2 - (Cov(co2,albedo)/Var(albedo))
* (albedo_retrieved - albedo_TRUE)`), with the TRUE albedo error as
input -- an oracle-level upper bound no real correction could ever
match -- moved CO2's MAE by ~0.0% at every config tested. `S_ret`'s cross-
covariance describes NOISE-driven sensitivity (how the estimate would
covary under a different noise realization); these runs are deterministic
(`noise=False`), so the actual error is a BIAS, and there's no reason a
noise-propagation covariance has to predict a systematic bias's
direction. It didn't, here.

### 12.5 A real, structural signal: CO2's own local truth curvature

Splitting bins into curvature quartiles (Sec.7.5's own methodology,
`|kappa| = |truth_{k-1} - 2*truth_k + truth_{k+1}|` via the standard
unequal-spacing 3-point formula) against CO2's OWN resolution-matched
truth field: high-curvature bins have 1.98x the error of low-curvature
ones at `g_ratio=3`, and a STRONGER 2.64x at `g_ratio=1` -- CO2's error
concentrates near its own plume/hot-spot features (real local structure),
not near albedo's patch boundaries (the cross-curvature ratio was weaker,
1.52x/1.30x). Raw Pearson correlations were all near zero for every
predictor tested -- the signal only appears in the quartile split,
matching Sec.7.5's own finding that the plain correlation coefficient
misses it (heavy-tailed/threshold-like error distribution).

### 12.6 Ruled out as the DOMINANT driver: per-bin sample/pixel starvation

`pixel_density_bin_centers` places bin CENTERS at equal-population
quantiles, but pixel ASSIGNMENT is nearest-center (Voronoi/midpoint)
logic -- a different operation that does NOT reproduce exactly equal
per-bin pixel counts. Measured directly (one real window, rows 418-438):
`g_ratio=3` pixels/bin min=1025 max=4337 mean=3072; `g_ratio=1` min=317
max=1255 mean=1024 -- a real ~3x reduction matching G's own increase, with
real (~4x) spread at both resolutions. Checking whether this explains the
`g_ratio=1` degradation: at `g_ratio=3`, sparse-data bins genuinely fit
WORSE (low/high pixel-count quartile MAE ratio 2.28x, matching intuition).
At `g_ratio=1` this REVERSES (ratio 0.62x -- sparse bins do BETTER) --
the same "dominant mechanism swamps and inverts a real secondary
correlation" signature Sec.7.5 already documented for a different
comparison. Rules out sample starvation as `g_ratio=1`'s dominant driver,
and is convergent evidence (a second, independent test) for 12.5's
curvature-linked mechanism being the real one.

### 12.7 The actual answer: a representability-gap artifact, not a bug or a persistent degeneracy

Every resolution-matched truth through Sec.11 was built from
`resolution_matched_fields`/`resolution_matched_albedo_fn` sampled at the
ANCHOR grid (`whole_slit_anchor_etas`, `anchor_density`-driven) -- but the
RETRIEVAL's own bin values are piecewise-linear at the (coarser)
**BIN** grid (`g_ratio`-driven). At `g_ratio=1`/`anchor_density=4`, bins
are ~2.7km apart, anchors ~0.67km apart -- a real ~4x gap. Even with the
prior set EXACTLY to truth at every bin center (Sec.12.2), the forward
model evaluated between bin centers does NOT reproduce the true
(anchor-resolved) radiance, so `observation - model` and `prior - model`
are both nonzero AT THE PRIOR, before any data-driven adjustment --
that nonzero residual is exactly what drives Gauss-Newton away from the
(exactly correct) starting point. This representability gap SHRINKS as
`g_ratio` gets finer (closer to the anchor grid), so it does not, by
itself, explain why `g_ratio=1` is WORSE than `g_ratio=3` -- but it does
mean neither config's error should be read as a pure measure of the
underlying joint-degeneracy mechanism, since both are also carrying this
confound, unquantified until now.

**Fix**: built a truth resolution-matched to the RETRIEVAL's own BIN grid
instead of the anchor grid -- `whole_slit_bin_centers(fpa, g_ratio)`
(`scripts/gd_build_resolution_matched_truth.py`, new, mirrors
`whole_slit_anchor_etas`'s per-window-union construction exactly, using
`_solve_window`'s own `G = max(2, round(width/g_ratio))`/
`pixel_density_bin_centers` formula) -- combined with a FINE
`anchor_density` (16) purely for forward-model sampling. This makes the
retrieval's own bin-to-bin piecewise-linear reconstruction bit-for-bit
the SAME function the truth was built from, everywhere -- not merely at
the bin centers -- a genuine ceiling test, not "as fine as the forward
model happens to sample." (Also required: `gd_joint_block_whole_slit_
sweep.py --resolution-matched-g-ratio-bins` -- and a real fix caught by
its own smoke test: `state_spec_from_scene`'s `surface_fields` convention
is `{"albedo": fn(x_km, band_label)}`, two positional args, keyed by ROW
NAME -- different from `build_lookup_radiance`'s own `surface_fields`
convention, `{band_label: fn(x_km)}`, one arg, keyed by BAND LABEL --
reusing the latter directly as the former raised `takes 1 positional
argument but 2 were given`; fixed with a small wrapper. Also extended the
"exact prior must mean the resolution-matched truth, not the raw
continuous one" fix from Sec.11 to this new bin-grid-matched mode.)

**Result** (`g_ratio=1`, `anchor_density=16`, `--prior-fields exact`,
truth matched to `g_ratio=1`'s own bin grid):

| config | CO2 overall | CO2 rel. | albedo overall | dof_frac | resid_rms |
|---|---|---|---|---|---|
| g3, ad4 (baseline, anchor-matched truth) | 17.49 ppm | 4.21% | 0.0171 | 0.999 | 0.225 |
| g1, ad4, structural (anchor-matched truth) | 32.78 ppm | 7.89% | 0.0103 | 0.960 | 0.139 |
| g1, ad4, exact (anchor-matched truth) | 32.52 ppm | 7.82% | 0.0104 | 0.960 | 0.139 |
| **g1, ad16, exact, BIN-matched truth (zero gap)** | **4.05 ppm** | **0.97%** | **0.0029** | 0.957 | **0.034** |

Removing the representability gap entirely drops CO2 error ~8x (32.78 ->
4.05 ppm) and albedo ~3.5x (0.0103 -> 0.0029) -- coming in well BELOW the
`g_ratio=3` baseline, reversing the original finding completely. The
residual against the real data also improved sharply (0.139 -> 0.034),
while `dof_frac` stayed essentially flat (~0.96) throughout -- so this
was never a conditioning story either; it is specifically that the model
can now genuinely represent what it is being asked to fit.

**Conclusion**: the joint co2/p_surface/albedo cross-correlation found in
12.4 is real (measured directly, 0.3-0.7), but it was never the DOMINANT
driver of the `g_ratio=1` degradation -- it is a real, secondary effect
that was almost entirely masked by the much larger representability-gap
artifact. The original "g_ratio=1 looks worse than g_ratio=3" result was
a genuine, reproducible finding FOR THAT SPECIFIC anchor-vs-bin-grid
mismatch, not evidence of a bug or an intractable degeneracy in the
retrieval mechanism itself -- once truth and bin grid are matched, finer
bins behave the way intuition always expected: strictly better, matching
Sec.7.3's own historical result (g_ratio=1 among the best configs there
too, under conditions without this mismatch).

**Confirmation: prior quality is irrelevant once the gap is closed.**
Rerunning the bin-matched ceiling test with `--prior-fields structural`
instead of `exact` gives an essentially identical result -- CO2 4.0592 vs.
4.0468 ppm, albedo 0.0029 vs. 0.0029, `dof_frac`/`resid_rms` unchanged.
This closes the loop with 12.2 (which already showed exact vs. structural
barely differed WITH the representability gap present): with `dof_frac
~0.96`, the fit is strongly data-constrained at every stage of this
investigation, so the prior was never actually driving the answer -- the
representability gap was the only thing standing between the retrieval
and a genuinely good fit.

One minor, unrelated artifact found and confirmed harmless along the way:
`whole_slit_bin_centers`'s per-window union can place two DIFFERENT
windows' own boundary bins extremely close together at a shared physical
seam (minimum observed: 0.43m, not an exact duplicate -- `x_km` stays
strictly increasing, `np.interp` is unaffected).

## 13. Footprint-integrated forward model, parallelism, and retiring spectral-blend truth (2026-09-01/04)

Work in this section predates Sec.14 chronologically and set up the tools
Sec.14 relies on (Mode-2 truth, the averaging kernel, LM convergence).
Partially committed: `investigate-g1-ad4-structural-divergence` branch,
commit `7daecec` (`footprint_average_scene`, `predict_neighborhood`'s
`footprint=` param, `gd_build_resolution_matched_truth.py`, the
`--anchor-workers` CLI flag, and the first pass of anchor-level
parallelism). The Jacobian-consistency fix (13.2), further LM refinement,
`build_scene_fields` (13.7), the `_band_setup` migration (13.8), and the
plot-script rewrite (13.6) were done afterward and are still uncommitted
as of this writing.

### 13.1 Motivation: spectral blending, not a genuine effect, was the dominant source of representability-gap artifacts

Every truth image before this point (`als.build_lookup_radiance`, used by
`gd_test._band_setup`/`_band_setup_cached` and hence every prior sweep in
Sec.7-12) rendered a detector pixel by taking exactly TWO nearby
precomputed spectra (`nearest_bin_scene`) and linearly blending them --
never a real sub-pixel footprint integral over genuinely-sampled RT
output. Confirmed this session to be the actual dominant source of the
representability-gap artifacts Sec.12's whole investigation chased before
tracing it here (Sec.12.7 found and fixed the anchor-vs-bin-grid mismatch,
a real, separate, and smaller effect layered on top of this one). Fix:
replace the two-sample blend with real per-pixel footprint integration of
exact, never-interpolated RT samples end to end.

### 13.2 The new mechanism: `footprint_average_scene` + `render_at_anchors`

`geocarb_gert.focalplane.footprint_average_scene(bin_centers, spectra,
eta_range=None)`: generalizes `nearest_bin_scene` from a point query to a
footprint query (`radiance(eta_lo, eta_hi)`), via a zone-overlap-weighted
average using a cumulative-integral trick -- `O(n_query)`, no per-pixel
anchor loop. Verified against brute force to `1e-14`. Genuine quadrature
over real samples, never spectral interpolation.

`gd_render.predict_neighborhood(..., footprint: bool = False)`: when
`True`, computes per-`(row,col)` footprint edges (eta at row +/- 0.5,
keystone-correct) instead of a single point eta.

`geocarb_gert.joint_state.render_at_anchors(fpa, rows_win, anchor_etas,
atm_params, surf_params, spectrum, wn_hires, ils, pad=4, n_workers=None,
return_spectra=False)`: the shared no-caching forward-model core now used
by both truth rendering and (via `build_forward_state`) the retrieval
itself -- fresh RT per anchor (parallelizable, 13.4), footprint-integrated
into pixels. `return_spectra=True` (added 13.8, to fix a real duplicate-
RT-computation bug there) returns `(image, anchor_etas_sorted, spectra)`
instead of just `image`, so a caller can reuse the same anchor spectra
elsewhere without a second RT pass.

**A real forward-model inconsistency found and fixed along the way**:
`jacobians.py::linearize`/`anchor_spectra_and_derivs` were still using
`nearest_bin_scene` while `build_forward_state` had already switched to
`footprint_average_scene` -- a genuine ~0.045 rms disagreement between the
Jacobian used to take a GN step and the forward model used to score it.
Both now use `footprint_average_scene` identically.

### 13.3 Levenberg-Marquardt damping and an early-exit fix in `gauss_newton_state`

Found a real GN non-convergence bug via live iteration traces: undamped
Gauss-Newton steps could increase the penalized objective and the solver
had no mechanism to reject them. Fixed with LM damping (a damped step is
accepted only if it decreases the full penalized objective) plus an
early-exit check (`dx_cheap` evaluated at the current `lam`, before
entering the expensive inner-try loop) so an already-converged point
doesn't waste `lm_max_tries` forward() calls chasing itself.

### 13.4 Anchor-level and two-level (SLURM array x anchor-workers) parallelism

`render_at_anchors` was single-threaded even though each anchor's RT run
is embarrassingly parallel. Added real anchor-level parallelism (module-
level global-dict + fork-based `multiprocessing.Pool`, guarded by
`mp.current_process().daemon` checks to avoid nested-pool errors) to:
`render_at_anchors` itself, `build_forward_state` (new `n_workers`/
`min_parallel_anchors` params, default `n_workers=1` since callers
usually already run inside a window-level pool), and
`jacobians.py::linearize`/`anchor_spectra_and_derivs`/the `L()` detector-
operator loop. Verified bit-identical to the serial path, with real
speedups: `linearize()` 5.28x on a 513-anchor window, `build_forward_state`
3.2x on the same.

New `--anchor-workers` CLI flag on `gd_joint_block_whole_slit_sweep.py`,
threaded to `build_forward_state`/`jac.linearize`'s own `n_workers`. Two
DIFFERENT parallelism axes interact: an in-process window-level Pool
(`--n-workers`, workers ARE daemons, so nested pools are forced back to
`n_workers=1` there -- an explicit console NOTE fires when both are set)
vs. SLURM array tasks (`--task-id`/`--n-tasks`, top-level processes, NOT
daemons, so `--anchor-workers` genuinely stacks on top) -- confirmed the
cluster wasn't node-limited (269 idle CPUs) before relaunching production
sweeps as 58-task-per-window job arrays with real two-level parallelism.

### 13.5 Mode 1 (dense) and Mode 2 (representative) whole-slit truth

`scripts/gd_build_resolution_matched_truth.py`:
- **Mode 1, "dense"** (`render_dense_truth_window`): anchors every `dx_km`
  (default 500m) uniformly across a window's own PAD-extended range,
  sampling the full continuous `STATE_FIELDS`/`SURFACE_FIELDS` -- the best
  affordable approximation to ground truth, not exactly representable by
  any retrieval's own bin grid.
- **Mode 2, "representative"** (`render_representative_truth_window`):
  literally reuses `state_spec_from_scene` + `build_forward_state` (the
  retrieval's OWN machinery) with the state fixed at `resolution_matched_
  fields(bin_centers)`'s own prior (exact at every bin center) and
  evaluates `forward(x0())` -- the SAME code path the retrieval calls
  every GN iteration, at the prior. Representability is therefore zero BY
  CONSTRUCTION (same function, same inputs), not by two independently-
  written pipelines happening to agree closely. This is what Sec.14 scores
  every imperfect-prior result against, so any residual error there is
  retrieval/prior error, never truth the model couldn't have represented.

`build_whole_slit_truth(fpa, mode, ..., overlap=0, stitch=True,
n_workers=None)`: stitches windows into one `(1024,1024)` array
(`stitch=True`, `overlap=0` only -- disjoint tiling required for
unambiguous stitching) or returns a per-window `{(row_lo,row_hi):
sub_image}` dict (`stitch=False`, any overlap, for debugging without
resolving the cross-window stitching-conflict question at `overlap>0`).

### 13.6 The plot script's own truth-reference bug: keystone-induced eta-interleaving, and its fix

Found while investigating a frozen-row/plot-vs-truth mismatch that turned
out NOT to be a retrieval bug: even at `overlap=0` (row-DISJOINT window
tiling), windows are NOT eta-disjoint, because eta depends on column too
-- a neighboring window's bin center can fall inside another window's own
eta span (the same keystone effect Sec.14.4 later uses to explain the
water-region smearing). `gd_joint_block_whole_slit_plot.py`'s old whole-
slit-union truth reference did along-slit interpolation across that union
grid, which made it subtly wrong at non-bin-center rows near window edges
-- a SCORING bug, not a retrieval one.

Fixed by eliminating along-slit interpolation entirely (per user
direction) rather than patching the union-grid construction: truth is now
evaluated exactly at each state's own bin-center positions (`_truth_fn_
for` -- `raw` and `bin` modes are now identical there, by definition of
bin-mode truth's exactness; `anchor` mode still needs `resolution_matched_
fields`/`whole_slit_anchor_etas`, since that truth is one coherent whole-
slit function with no per-window locality issue). Each window is plotted
as its own disconnected `-o` segment, with grey dashed vertical lines at
window boundaries. Eta -> row conversion is used for x-axis PLACEMENT
only (nearest-neighbor lookup), never for value reconstruction.

### 13.7 Scene-parameter unification: `build_scene_fields`

User direction: "Barcode and realistic barcode are just different sets of
scene parameters" -- shift every scene-mode definition to one place before
touching how they're rendered. `along_slit_scene.build_scene_fields
(uniform, barcode, realistic_barcode, vary_albedo, barcode_bars,
band_labels, constant_albedo, fields=None, surface_fields=None)` unifies
all 4 scene "modes" into one `{name: fn(x_km)}`/`{band_label: fn(x_km)}`
factory (matching `build_lookup_radiance`'s own convention, not `state_
spec_from_scene`'s `{"albedo": fn(x_km, label)}`).

**A real, deliberate physics change**: barcode's own bar pattern is now
expressed as a real `surface_fields["albedo"]` step function (sharp edges
via `eta > boundary`, matching `barcode_scene`'s own softness=0 default)
rather than a post-hoc linear gain multiplied onto one center spectrum --
a genuine per-bar RT rerun instead of a linear-in-albedo approximation.
Confirmed intentional with the user before implementing. Verified via
smoke test across all 6 modes (plain realistic, uniform, barcode,
realistic_barcode with/without vary_albedo, vary_albedo-only, explicit-
override passthrough), including a direct `real_albedo x bar_gain` math
check.

### 13.8 `_band_setup` migrated off `build_lookup_radiance`

`scripts/gd_test.py::_band_setup` rewritten to use `build_scene_fields` +
`render_at_anchors` + `footprint_average_scene` instead of `als.build_
lookup_radiance`/`gd_render.image` -- the same footprint-integrated
mechanism now used everywhere else, closing the loop on 13.1. New `dx_km`
parameter (default 0.5, whole-slit anchor spacing); `n_lookup_samples` is
now an unused-but-harmless parameter kept for `_band_setup_cached`'s
existing positional call signature/cache-key back-compat.

Caught and fixed a real bug in the migration itself: an early draft
called `render_at_anchors` for the image, then separately recomputed the
same anchor spectra a second time for the point-query `radiance` shim
(`_undistorted_row`'s own compatibility need) -- a real 2x RT cost
regression. Fixed by adding `render_at_anchors`'s `return_spectra=True`
(13.2) and reusing `(A, anchor_etas_sorted, cache_S)` for both.

Verified end-to-end (not just `py_compile`): `_band_setup(fpa=2,
barcode=True, barcode_bars=4, dx_km=20.0, n_workers=8)` returns a sane
`(1024,1024)` image (range 0.03-12.1) and a working point-query
`radiance()` shim, in ~208s. `_band_setup_cached`'s existing positional
call to `_band_setup` is unaffected (the new `dx_km` param simply keeps
its default, appended after every existing positional argument).

`build_lookup_radiance`/`gd_render.image` are no longer called anywhere
in `gd_test.py` (confirmed via grep) -- left in place in `along_slit_
scene.py`/`gd_render.py` themselves in case another caller still needs
them; not removed, since that wasn't requested.

### 13.9 Truth-cache and repository cleanup

`geocarb_gert.truth_cache` (hash-keyed, `cache_key(**fields)` -> SHA-256
-> `f"{key}.pkl"`) had 13 entries, ALL of them rendered via `build_lookup_
radiance`'s spectral blending (confirmed: `build_lookup_radiance` is the
ONLY thing that ever wrote to this cache) -- deleted entirely per 13.1;
every result that ever consumed one of these cached renders is unusable
as of this cleanup. `results/truth_cache/MANIFEST.md` records the
historical hash -> config mapping for anything that needs to be
reproduced/audited later. Renaming cache files to something descriptive
was considered and rejected -- it would silently break the hash-based
lookup (a cache miss, not an error) -- the manifest exists instead.

Directory reorganization (`plots/joint_block/`, `results/`): superseded
plot files (18) deleted, 128 diagnostic-run files moved to `diagnostics_
archive/`; 22 legacy top-level result pkls + a stale `multi_block_fd_
rerun_candidates/` dir deleted; 106 other top-level pkls moved to
`results/archive/`; `_parts/` directories with a merged equivalent
elsewhere deleted (3 large ones, 4 single-file smoke-test ones, and a
59-file legacy-schema `gratio1_parts/` with only partial recoverable
assumptions); `resolution_matched_v1/results/` organized into `mode2_
representative_truth/`, `mode1_dense_truth/`, and `superseded_pre_
footprint_fix/` (14 files predating 13.1's fix, kept but clearly
separated rather than deleted, since they were real production results
under the old, since-superseded mechanism).

### 13.10 Multi-FPA verification (2026-09-04)

Everything through 13.9 was exercised exclusively on FPA2 (`CO2_strong`).
Before treating this branch as merge-ready, verified the migrated
pipeline on all 4 GeoCarb bands (`GEOCARB_BANDS`: 0=`O2_A` (o2,h2o),
1=`CO2_weak` (co2,h2o), 2=`CO2_strong` (co2,h2o), 3=`CH4_CO`
(ch4,co,h2o,n2o)):

**Forward-render smoke tests** (`_band_setup` directly, realistic scene):
all 4 bands render a sane `(1024,1024)` image with a working `radiance()`
shim; FPA2 additionally checked with `noise=True` (the actual default
production path, not yet directly exercised by any prior test this
session). FPA0 was notably slower (490s vs. ~230-250s for the others at
matched coarse `dx_km`) -- a throughput note, not a correctness issue.

**Full retrieval smoke tests** (`gd_joint_block_whole_slit_sweep.py`,
forward render -> analytic Jacobian -> GN solve): a single-window test
per band with the DEFAULT `--free co2_ppm` solved without error on all
4 bands, but gave `dof=0/n_free` (fully prior-dominated, zero data
constraint) on FPA0 and FPA3 -- initially concerning, but correctly
diagnosed (by the user) as not a bug: FPA0 and FPA3 have no CO2 in their
molecule list at all, so retrieving `co2_ppm` from either is physically
meaningless regardless of how correct the forward model is. Re-run with
each band's own physically appropriate target gas instead (`--free
p_surface_hpa` for FPA0, `--free ch4_ppb,co_ppb` for FPA3, covering
roughly half of each band's own windows -- 31/63 and 39/78
respectively): both now converge fully data-constrained everywhere
(`dof_frac` 0.999-1.000 for FPA0, 0.988-0.999 for FPA3, tiny residuals
throughout, 0 errored windows). Confirms the migrated pipeline has real,
correctly-conditioned sensitivity to each band's own target gas, not
just "doesn't crash." `--fpa`-based window tiling itself also differs
per band (`build_window_tiles` depends on that band's own keystone
curve): 58/66/63/78 natural windows for FPA2/1/0/3 respectively at the
same default settings -- see Sec.15 for why this matters for eventual
multi-band retrieval.

### 13.11 Old-vs-new forward-model regression comparison -- closed

Last open item on the merge checklist: how much do actual pixel
radiances change between the OLD mechanism (`build_lookup_radiance`'s
two-sample linear spectral blend, `n_samples=400` -- what every
production sweep used before this branch) and the NEW mechanism
(`render_at_anchors`/`footprint_average_scene`), for the identical
physical truth scene and ForwardModel setup?
`scratch_work/old_vs_new_forward_model.py` (SLURM job 1859496) renders
both for the same 51-row window (rows 400-450, FPA2, a generic mid-slit
location with no plume/hotspot/albedo-boundary structure nearby) --
79 anchors (`dx_km=2.0`) for the new path vs. 400 whole-slit samples for
the old one.

**Result**: rel. diff rms = 5.9e-5 (0.006%), max = 3.5e-4 (0.035%) --
the two mechanisms agree closely at a location without sharp local
structure, as expected: a widely-spaced linear blend only accumulates
real error near features narrower than its sample spacing (plume/hotspot
edges, the water/albedo boundary), exactly what 13.1/14.3-14.4 found.
This is a spot check at ONE structure-free location, not a re-derivation
of 13.1's own finding (which was about behavior AT sharp features) --
but it does confirm the two mechanisms are not wildly divergent in
general, which is what this checklist item needed. **Closed.**

## 14. Isolating the source of large bin-center errors under an imperfect prior (2026-09-04)

Sec.12 closed the representability-gap confound with an EXACT prior. The
open question this section answers: with the prior deliberately imperfect
again (`--prior-fields structural`, free = `co2_ppm,p_surface_hpa,albedo`),
against a Mode-2 representative truth (Sec.11-era `render_representative_
truth_window`/`build_whole_slit_truth`, representability = 0 by
construction at the retrieval's own bin/anchor grid) -- why are the
resulting bin-center errors still large, and how much of that is
reducible?

Config for everything in this section unless noted: `g_ratio=1`,
`anchor_density=4` (switched down from the earlier `ad16` mid-session for
speed -- required a FRESH Mode-2 truth build, `scratch_work/
whole_slit_truth_ad4.pkl`, since representability is defined relative to a
specific anchor_density; an `ad16` truth is not exactly representable at
`ad4` bin/anchor spacing). Two whole-slit (58-window) runs:
- `results/realistic_prior/config_matrix/resolution_matched_v1/results/
  mode2_representative_truth/co2p_albedo-58-g1-ad4-pf-structural-
  MODE2TRUTH_analytic.pkl` -- baseline, structural prior on every row
  (free and frozen alike).
- `..._frozenexact-MODE2TRUTH_analytic.pkl` -- ablation (14.1 below): same
  in every other respect, but the three FROZEN atmosphere rows
  (`ch4_ppb`, `co_ppb`, `h2o_surface_vmr`) fixed at exact truth instead of
  the structural prior. Driver scripts: `scratch_work/retrieval_vs_
  mode2_truth_{structural,frozen_exact}_array_ad4.py` (job arrays
  1852938/1852939, 8 of 58 wide windows re-run at `--time=02:00:00` after
  timing out at the original 45min budget: 1854008/1854009). Merged with
  `scripts/gd_joint_block_whole_slit_merge.py` (its 7-key metadata
  whitelist drops `prior_fields` -- patched back into both merged pickles
  directly afterward so the plot script's prior-overlay line renders
  correctly; the merge script itself untouched). Plots: `plots/joint_block/
  co2p_albedo-58-g1-ad4-pf-{structural,frozenexact}-MODE2TRUTH_
  analytic.png`.

### 14.1 Ablation: frozen-row contamination — ruled out completely

Hypothesis (from the session's own earlier concern: "the frozen variables
were taken from the structural prior -- this will make it impossible to
converge properly"): maybe the large errors trace to the FROZEN rows'
imperfect prior contaminating the fit, not a genuine problem with the
free rows themselves.

Result: the three free rows' retrieved values (`co2_ppm`, `p_surface_hpa`,
`albedo`) are **bit-identical** between the two runs, everywhere, across
all 58 windows (`max|diff| = 0.0` exactly, to float precision). Verified
this wasn't a wiring no-op: the frozen row itself (`ch4_ppb`) really does
differ substantially between the two configs (rms diff ~12 ppb, max ~45
ppb -- matching the CH4 hotspot amplitude), so the ablation's prior swap
took effect; it just has zero downstream effect on the free-row solution.

Both configs converge to a near-zero residual (`resid_hires_rms` ~1e-5,
essentially at optimizer tolerance) while still landing far from truth
(`co2_ppm` rms error 0.191 ppm, max 2.263 ppm) and far from ITS OWN prior
(moved rms 1.29 ppm to get there) -- a real, converged fit, not a
no-op. **Interpretation**: at `g_ratio=1`/`ad4` with 3 free rows, the
system is close to exactly-determined -- GN finds a genuine near-zero-
residual solution regardless of what the frozen rows assume, because the
free parameters have enough capacity to fully absorb whatever the
frozen-row mismatch does to the radiance. Where in that near-null-space
the solve lands (and hence its bias against truth) is decided entirely by
the free rows' own prior/spatial-regularization pull, not by frozen-row
error. This rules out frozen-row contamination and points at a genuine
free-parameter degeneracy resolved by the correlation-length prior rather
than by data content -- confirming the mechanism the user's own message
originally floated, rather than merely suspecting it.

### 14.2 Averaging-kernel diagnostic: prior-pull explains most of it, unevenly by row

`gauss_newton_state`'s `avk` (Rodgers averaging kernel, already saved per
window, Sec.7.10's "not yet wired into production" tooling) was pulled
directly from the structural-baseline pkl -- no new run needed.

| row | mean diag(A) | frac bins diag(A)<0.9 | corr(1-diag(A), \|error\|) | RMS\|err\|, worst vs. best diag(A) tercile |
|---|---|---|---|---|
| co2_ppm | 0.992 | 0.9% | 0.34 | 0.328 vs 0.013 ppm (25x) |
| p_surface_hpa | 0.843 | **70%** | 0.41 | 0.022 vs 0.008 hPa (2.7x) |
| albedo | 0.998 | 0.1% | **0.997** | 0.00023 vs 0.0000064 (37x) |

Overall DOF-for-signal is high (mean 50/53 ~= 0.94 of free capacity is
genuinely data-constrained), but that average hides real structure:
- **Albedo's error is almost entirely explained by prior-domination**
  (corr 0.997, essentially deterministic) -- wherever the averaging
  kernel collapses (a handful of bins per window, scattered through the
  interior, NOT concentrated at window edges -- checked directly, only
  1.7% of per-window worst-diag(A) bins fall in the outer 10% of a
  window), the fit leans on its own imperfect (structural) prior and
  that's exactly where the error is large.
- **CO2 shows the same mechanism only partially** (corr 0.34) -- real,
  but with enough scatter that something else also contributes (14.3/14.4
  below).
- **p_surface_hpa is prior-dominated most of the time** (70% of bins
  below diag(A)=0.9) but stays low-error in absolute terms, because its
  structural prior keeps the topographic "mountain" term exactly and only
  drops the synoptic sinusoid -- prior-domination there is frequent but
  mostly harmless.

### 14.3 A near-zero-albedo (water) region drives most of the worst CO2 error -- and it is NOT a prior-pull effect there

Visual inspection of the plots (13's plot files, `--truth-image
scratch_work/whole_slit_truth_ad4.pkl` panel added per 14.5's own tooling
note) showed the worst CO2/p_surface performance concentrated in one
region with near-zero albedo (water). Quantified via the new continuum-
SNR-vs-row panel (`gd_joint_block_whole_slit_plot.py --truth-image`,
14.5): continuum = each row's own brightest/least-absorbed column,
`sigma` from the real per-band `geocarb_noise_model` (`LinearShotNoise`).
Splitting all 1024 bins at continuum SNR > 100:

| row | SNR>100 (992 bins) | SNR<=100 (32 bins, water) | ALL (1024) |
|---|---|---|---|
| co2_ppm rms / max\|err\| | 0.108 / 1.74 ppm | 0.899 / 2.26 ppm | 0.191 / 2.26 ppm |
| p_surface_hpa rms / max\|err\| | 0.0152 / 0.159 hPa | 0.0461 / 0.0948 hPa | 0.0171 / 0.159 hPa |
| albedo rms / max\|err\| | 0.000138 / 0.00431 | 0.0000176 / 0.0000430 | 0.000135 / 0.00431 |

The SNR<=100 set is a single **contiguous 32-row block (rows 686-717,
3.1% of the slit)**, mean true albedo ~0.015 (water). Identical between
the structural baseline and the frozen-exact ablation (14.1).

Adding the PRIOR error (not just posterior) to the same split is what
actually explains the mechanism, and it differs by row:

| row | subset | post rms | prior rms | prior mean |
|---|---|---|---|---|
| co2_ppm | SNR<=100 (water) | 0.899 | **0.0 (exact)** | 0 |
| co2_ppm | SNR>100 | 0.108 | 1.299 | -0.418 |
| p_surface_hpa | SNR<=100 (water) | 0.0461 | 2.995 | **-2.995** (worst-case) |
| p_surface_hpa | SNR>100 | 0.0152 | 2.099 | +0.074 |
| albedo | SNR<=100 (water) | 0.0000176 | 0.00134 | -0.000023 |
| albedo | SNR>100 | 0.000138 | 0.0265 | -0.000497 |

**CO2 and p_surface are degraded in the water region for two DIFFERENT
reasons, not one "low SNR" story**:
- CO2's structural prior is *exactly correct* in the water region (rms
  error = 0.0 -- that stretch is far from the CO2 hotspot/plume, so
  there is nothing localized for the structural prior to be missing).
  Yet the POSTERIOR is worse there (0.899 vs 0.108 rms) than anywhere
  else on the slit -- a good prior getting pulled AWAY from truth. Most
  likely mechanism: low albedo starves the CO2 Jacobian of sensitivity,
  so the near-null-space fit (14.1) resolves in a direction that trades
  away CO2 accuracy specifically where the retrieval has the least
  independent leverage to pin it down -- an information-starvation /
  ill-conditioning effect, not prior-pull.
- p_surface's structural prior is at its WORST in the water region (rms
  2.995 hPa, mean essentially exactly -3.0 hPa -- the full dropped-
  sinusoid amplitude, i.e. this stretch sits near the sinusoid's trough).
  The posterior still corrects most of it (down to 0.046 hPa, a 65x
  reduction) -- a good correction despite low SNR, just not as tight as
  the 0.015 hPa achieved elsewhere. This IS closer to ordinary prior-pull
  behavior, just starting from a much larger prior error.
- Albedo's own error is smaller in absolute terms in the water region for
  both prior and posterior, simply because a near-zero true value has
  little room to be wrong about.

### 14.4 Keystone smearing extends the water region's damage into neighboring high-SNR rows

Among the SNR>100 bins, the worst CO2 errors are NOT randomly
distributed -- the three exceeding 1.0 ppm (rows 718, 719, 721; the next-
worst rows 720, 722-728 decay smoothly from there) sit immediately
adjacent to the water region's edge (row 717), not scattered elsewhere on
the slit. Window tiling was checked and ruled out as a competing
explanation: this stretch (rows 701-733) is one continuous window, and
the bad rows sit well inside it, not at its boundary.

The mechanism is keystone, confirmed by direct calculation: `_eta_of`
evaluated across all 1024 spectral columns at a FIXED detector row shows
each row's own spectral samples span ~20 km along the slit (not a single
point) -- e.g. at row 719, columns 0 and 1023 correspond to along-slit
positions 20.1 km apart. Comparing each row's keystone-shifted footprint
against the water region's boundary (x_km <= 530.5, i.e. row 717):

| row | center x_km | nearest-edge x_km (keystone-shifted) | overlaps water? | CO2 error |
|---|---|---|---|---|
| 718 | 533.2 | 522.4 | yes | -1.20 |
| 719 | 535.9 | 525.1 | yes | **+1.74 (worst)** |
| 721 | 541.3 | 530.4 | yes, barely | -1.42 |
| 722 | 544.0 | 533.1 | **no** | +0.90 |
| 723-728 | -- | -- | no | decaying smoothly to ~0.2 |

Rows 718-721 are exactly the rows whose keystone-shifted spectral
footprint STILL physically overlaps the water region; row 722 is the
first row whose full column range clears the water boundary, and that is
precisely where the error drops sharply and then decays smoothly outward
as the overlapping fraction of columns shrinks toward zero (out to ~10
rows total, consistent with the ~20km keystone spread against the local
~2.7km/row bin spacing there). So this is not a "high-SNR bad CO2"
anomaly independent of the water region -- it IS the water region's
effect, reaching a few rows further than its own nominal boundary via
keystone. The row-based SNR>100/<=100 split in 14.3 therefore understates
the water region's true footprint of damage.

### 14.5 New tooling: continuum-SNR-vs-row panel

`scripts/gd_joint_block_whole_slit_plot.py` gained an optional
`--truth-image PATH` flag (off by default -- the sweep's own output
pickle never carries the rendered detector image, only derived state/
residual quantities, so this needs the actual truth image the run's
`y_true` came from, e.g. `scratch_work/whole_slit_truth_ad4.pkl`). When
given, adds a bottom panel: continuum SNR vs. detector row (== eta,
monotonically -- the same x-axis every other panel already uses),
log-scale, with the same window-boundary lines as the rest of the
figure. Continuum per row = that row's own brightest (least line-
absorbed) column -- an honest, deterministic proxy given these are
noise-free synthetic truth images; a real per-pixel-noise image would
need a true continuum-region mask instead of a raw max. `sigma` from the
real per-band `geocarb_noise_model`/`LinearShotNoise` calibration, the
same one the retrievals themselves weight by.

### 14.6 Net conclusion

Combining 14.1-14.4: the large bin-center errors under an imperfect
prior are **not** explained by frozen-row contamination (ruled out
completely, bit-identical either way) or by a classical representability
gap (the Mode-2 truth is exactly representable by construction). They
are explained by a combination of (a) genuine prior-pull in poorly-
constrained directions of the free-row fit itself -- dominant for
albedo, partial for CO2, mostly harmless for p_surface because its
structural prior stays close to truth even at its worst -- and (b) a
near-zero-albedo (water) region that degrades CO2 for a DIFFERENT reason
(information starvation pulling an already-exact prior away from truth,
not prior-pull), degrades p_surface by starting from an unusually bad
prior there, and whose damage reaches several rows beyond its own
boundary via keystone smearing. Both mechanisms are in principle
reducible: prior-pull via a looser/shorter spatial-correlation prior or
an averaging-kernel-weighted post-hoc bias correction (as the user
originally proposed); the water-region CO2 effect via either explicitly
modeling the keystone-smeared footprint's albedo heterogeneity in the
forward model, or accepting it as a real, bounded, and now-understood
floor near strong along-slit albedo contrast.

Not yet run: diagnostic #4 from the original menu (single-parameter
perturbation sweep, to separate CO2's own remaining unexplained scatter
into "co2/p_surface/albedo degeneracy" vs. other causes) and a looser-
spatial-prior test targeting albedo specifically, both proposed as
natural next steps but not yet requested.

## 15. Open design question: consistent bin placement across bands for multi-band retrieval (2026-09-04)

Surfaced while verifying Sec.13's forward-model migration on the other
three bands: a single-window retrieval smoke test requested via `--fpa 0
--n-windows 58` (the count that tiles cleanly for FPA2) failed outright --
`ValueError: no window_scale in [0.5, 6] gives 58 windows at
min_window=4; reachable counts near it: [52, 53, 54, 55, 57, 59, 60, 61,
63]`. Re-run with default tiling instead: FPA3 alone naturally lands on
**78** windows for the same 1024-row detector where FPA2 lands on 58.

**Why**: `build_window_tiles`/`scale_for_window_count` picks each band's
own tiling independently, driven entirely by THAT band's own `rows_
crossed` keystone curve -- there is no shared spatial grid across bands
at all. This has never had to be reconciled because multi-band retrieval
isn't implemented yet -- `gd_joint_block_whole_slit_sweep.py --fpa`
explicitly refuses more than one band (`"--fpa currently supports
exactly one band"`).

**The real problem, for whenever multi-band retrieval is built** (user,
2026-09-04): a sounding is one physical along-slit location with one
true atmospheric/surface state, viewed simultaneously by multiple bands
-- so state-vector bins must live on a SINGLE shared spatial (x_km/eta)
grid across bands, not each band's own separately-tiled one. Two
objectives in tension:
1. Maximize the number of spectral points captured per bin, across
   every band jointly -- not letting one band's coarser natural tiling
   waste another band's finer one.
2. Concentrate bins near each band's own high-keystone rows (where eta
   changes fastest with row -- the same mechanism Sec.14.4 found
   smearing the water region's error into neighboring rows; finer bins
   there are what keep the footprint-integration/representability error
   controlled, per Sec.12's whole investigation).

Today's per-band `scale_for_window_count` optimizes (2) locally per
band, which is exactly why FPA2 and FPA3 disagree on natural window
count for nominally the same request -- a multi-band scheme needs one
JOINT criterion instead (e.g. bin edges chosen so every band's own
worst-case footprint-integration error stays below a threshold
simultaneously, or the union of each band's own high-keystone regions)
rather than optimizing per-band and reconciling after the fact. Not
designed or implemented -- flagged here as an open problem for whenever
multi-band retrieval work begins.
