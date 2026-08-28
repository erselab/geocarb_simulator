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
