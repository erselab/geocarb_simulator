# Plan: joint multi-row inversion to recover a decoupled spatial/spectral field from the distorted FPA

Sketch only — not started. Written 2026-08-12 in response to a question
about whether an ML model, given enough training data, could learn to
invert the distorted FPA back into a decoupled spatial/spectral array.
Recommendation before reaching for ML: the distortion operator (keystone +
along-slit PSF) is already known **exactly** — it's the same
`xy_to_wavelength_slit`/`gaussian_blur_rows` machinery this whole
investigation has been using — so this is a classical (if nonstandard)
inverse problem with a known forward operator, not a "learn the physics
from data" problem. This document sketches that classical approach, and
closes with where ML could add value on top of it later, once the
classical solution exists as a validation target.

See `KEYSTONE_SMILE_BIAS_PLAN.md` §11o/§11p for the evidence this plan
responds to: native's independent per-row retrieval doesn't inflate
apparent resolution (§11o), but it does substantially under-recover a real,
localized signal near the high-keystone end of the slit — 44% of a true
point source's peak amplitude captured at row 910 (9.3 rows-crossed)
vs. 90% at row 112 (1.0 rows-crossed), vs. undistorted's ~99% at both.
That's the concrete failure this plan aims to fix.

## 0. A cheaper first cut: trace-and-select single-atmosphere retrieval

Proposed by the user, 2026-08-12, as a simpler alternative to try *before*
investing in the joint block below (§1-§4) — do this first; if it's
sufficient, the bigger build may not be needed at all.

**The idea.** For a target `η₀`, don't retrieve a whole row. Trace `η₀`'s
exact crossing across a neighborhood of rows — for each row `i`, find the
column `j(i)` where `η(i, j(i)) ≈ η₀` (exactly, or within some tolerance)
— and retrieve `η₀`'s atmosphere directly against just those individually-
selected real pixels, using the **existing, unmodified single-atmosphere**
`GERTRetrieval`/`ForwardModel` machinery. No new solver, no `G`-atmosphere
state, no Jacobian-gather structure — just a different choice of which
real pixels feed a retrieval call that already exists today (structurally
the same call `_native_row`/`_joint_retrieve` already make, just sourcing
its `obs_grid` from a trace across several rows instead of one row's own
1024 columns).

**Why "area/containment" turns out simpler than it sounds — first pass,
later corrected.** The original reasoning here was: at a given column, row
and `η` are close to linearly related, and keystone is entirely about how
that row↔`η` relationship's *offset* shifts with column, not about
blowing up the row-to-row mapping at a *fixed* column — so a correctly-
defined containment criterion should be satisfied at essentially every
exactly-traced pixel. **That reasoning was incomplete** — see §0a below,
where checking it directly (2026-08-12) turned up a real, code-verified
correction: `rows_crossed(fpa, row)` is an **endpoint-to-endpoint**
difference (`s` at column 4 vs. column 1019, `gd_polynomials.py:170`),
not a measure of the row's total internal excursion. A row can have
`rows_crossed≈0` (its two endpoints coincide) while still wiggling by
several tenths of a km internally, due to smile's own row-dependent
curvature — e.g. FPA2 row 25 (the canonical keystone-null row) swings from
−0.26 km to +0.37 km relative to its own center as column sweeps 0→1023,
a 0.63 km peak-to-peak excursion the endpoint-difference metric doesn't
see at all. So "how much of a row is usable for a given `η₀`" needs to be
evaluated per-pixel, not inferred from `rows_crossed` directly.

**Same safety property as native and the corrected joint block (§2)**:
every accepted point is a real, unmodified individual pixel at its own
true `(η, ν)` — nothing is ever interpolated or blended in radiance space,
whether tracing exactly or with a loosened window. This doesn't
reintroduce rectify's §9l/§9m bias either way.

**Sample budget, quantified directly (2026-08-12) — supersedes the
original "~20-30 samples, one per row" estimate below.** Built
`scripts/gd_toy_trace_pixels.py` (plots `plots/gd_toy_trace_pixels_fpa2.png`)
to evaluate *every* `(row, column)` pixel in a neighborhood against a
target `η₀`, not just the nearest column per row, with the overlap
tolerance exposed as `--tolerance-km` (default 0.2 km, the user's own
90%-area-overlap criterion). Actual included-pixel counts, FPA2:

| target row | `rows_crossed` | pixels included (tol=0.2 km) | rows contributing |
|---|---|---|---|
| 25 (null) | 0.00 | 441 | 1/9 (three separate column segments, from the internal wiggle above) |
| 100 | 0.83 | 427 | 1/9 |
| 300 | 3.00 | 153 | 3/15 |
| 512 | 5.24 | 147 | 5/25 |
| 700 | 7.18 | 152 | 8/33 |
| 950 | 9.69 | 154 | 10/43 |

Two things this revises:
- The sample budget is **not** "one point per row" — near-null-keystone
  rows contribute *many* columns each (the internal-wiggle effect above),
  while higher-keystone rows contribute few columns each but spread across
  more rows. Total pixel count is a better budget number than row count.
- That total is **roughly flat at ~150 pixels** across `rows_crossed` 3.0
  through 9.69 (153, 147, 152, 154) — the sample budget doesn't visibly
  shrink as keystone grows in this range, it just redistributes across
  more, thinner rows. It's much larger only very near the null region
  (441, 427), where the internal-wiggle effect briefly dominates.

~150 pixels (mid-to-high keystone) is a real, usable-looking budget for a
single-atmosphere fit (still tempered by the two risks below), well above
the original ~20-30 estimate. Two real risks remain, not just generic
photon noise:

- **Ordinary precision loss** relative to native's 1024 columns, tempered
  by the fact that native's columns are already heavily ILS-correlated,
  so the *effective* independent-sample loss is smaller than the raw
  count suggests.
- **A sharper risk**: nuisance parameters like the per-band dispersion
  terms (§11k) need several points *within* a single line's shape to be
  constrained at all. Whether ~150 pixels, scattered non-uniformly across
  a band (dense near the trace's own row, sparse elsewhere), resolve any
  individual line finely enough is not yet known — risking the same
  underconstrained-nuisance-parameter convergence fragility already seen
  elsewhere in this investigation (§9k/§11k), from a different cause.

**Why this goes first**: it doesn't need to answer "does `GERTRetrieval`
extend to a block-sparse multi-atmosphere state" at all (the joint block's
own Phase 0, §4) — it never extends it. Success or failure here is
informative regardless of whether the bigger build is ever attempted, and
it's a much smaller thing to build.

### 0a. Go/no-go test

- `scripts/gd_toy_trace_pixels.py` already builds the core piece: given
  `eta0` and a row window, it returns every `(row, column)` pixel within
  `--tolerance-km` of `eta0` (real, code-verified, not the original
  nearest-column-only sketch). Extend it to return `(ν, real pixel value)`
  pairs instead of just plotting them.
- Feed those pairs directly into the existing `GERTRetrieval`/
  `ForwardModel` call exactly as `_native_row` does today, just with this
  scattered, multi-row, multi-pixel-per-row `obs_grid` in place of one
  row's own 1024 columns. Use row 910 (§11p's hot-spot row) as the first
  target, tolerance=0.2 km, matching the table above (~150 pixels
  expected).
- **Go/no-go criterion**: does it converge at all (checking specifically
  whether dispersion/nuisance terms are well-constrained, per the risk
  above), and does its recovered peak amplitude beat native's 44% capture
  at row 910 from §11p? If yes, this is very likely sufficient on its own,
  and the much bigger joint-block build (§1-§4) may not be needed at all,
  except for the resolution/averaging-kernel rigor it uniquely offers
  (§3).
- If it fails to converge, or badly underperforms despite ~150 real
  pixels, that's itself a fast, cheap, informative result: it means the
  extra machinery in the joint block — which uses *all* pixels, not just
  high-overlap ones, optimally weighted rather than binarily
  included/excluded — is doing real, necessary work, not just added
  complexity. Worth knowing before building it.

## 1. Why this is a solvable classical problem, not a black box

`gd_render.image()` (`geocarb_gert/gd_render.py:149`) already shows the
forward operator explicitly, per row `i`, per column `j`:

```
lam_row, s_row   = xy_to_wavelength_slit(fpa, cols, full(N_PX, i))   # both f(i, j)
nu_row           = 1e4 / lam_row
eta_row_true     = s_row / s_max(fpa)
S_row            = radiance(eta_row_true)          # (N_PX, n_hires) -- one hi-res spectrum per column
A[i, j]          = ILS_convolve(S_row[j], centered at nu_row[j])
A                = gaussian_blur_rows(A, 1.5)       # THEN the PSF mixes rows
```

Two things follow directly from this, both already confirmed empirically
in §11o/§11p:

1. **Every pixel `(i, j)` has its own exactly-known `(η, ν)` pair** —
   `xy_to_wavelength_slit(fpa, j, i)` returns both, from real, calibrated,
   smooth bivariate polynomials (`_poly2d(coeffs[f"A{fpa}"], x, y)` /
   `coeffs[f"B{fpa}"]`). Nothing about this mapping is unknown or needs to
   be learned — it's the same object `xy_to_wavelength_slit_assumed()`
   already perturbs for the (unrelated) calibration-mismatch experiment in
   §12, i.e. the codebase already treats "the mapping" and "belief about
   the mapping" as separable, known objects.
2. **A row's own column range sweeps a range of `η`** exactly
   `rows_crossed(fpa, row)` rows wide (§11's own keystone-amplitude
   metric, already computed and plotted across all 1024 rows in §11p).
   Two rows separated by fewer rows than either one's own keystone
   amplitude have **overlapping** `η`-coverage — different rows are
   different, overlapping "views" of much of the same underlying strip of
   ground, sampled through different columns (different `ν`, i.e.
   different absorption lines) at each `η`.

That overlap is the information a per-row-independent retrieval (native)
throws away — it fits one atmosphere to each row's own spliced spectrum
and discards the fact that neighboring rows constrain the *same* ground
locations through different spectral windows. This is structurally the
same idea as multi-look/drizzle super-resolution reconstruction in
astronomical imaging or synthetic-aperture along-track oversampling in
microwave radiometry: many overlapping, shifted, partial views of a scene,
combined, resolve more than any single view — the physics here just
happens to come from a spectrometer's own dispersion+keystone coupling
instead of a moving telescope.

## 2. Reframing as one joint inverse problem over a neighborhood of rows

**State**: instead of one atmosphere per row (native's `StateVector`,
`gert.retrieval.StateVector`), define `G` atmospheres on a grid of `η`
bins covering the along-slit range spanned by a chosen neighborhood of
rows (e.g. `η_g`, `g = 1..G`, evenly spaced, `G` a free resolution choice
— start with `G` ≈ the number of rows in the neighborhood, i.e. no
resolution gain claimed yet, just decoupling; increase later once the
degenerate-information floor from §3 below is characterized).

**Neighborhood**: rows `[i0, i1]` wide enough that every `η` bin in range
is actually sampled by at least a few rows' worth of columns — in
practice `i1 - i0` should be a small multiple of `rows_crossed(fpa,
i_center)`, e.g. ±2-3x the local keystone amplitude (a handful of rows
near FPA2's null row 25, ~20-30 rows near row 910).

**Per-pixel prediction — state-space only, never radiance-space
combination.** Pixel `(i, j)` in the neighborhood has a known `(η(i,j),
ν(i,j))`. Its predicted radiance must come from running `ForwardModel.run()`
**once**, at that pixel's own exact `η(i,j)`, then ILS-convolving at
`ν(i,j)` — never by blending two bins' already-computed hi-res *spectra*.
If `η(i,j)` falls between two `η`-bins and a smooth forward map is wanted,
the correct place to interpolate is the **atmospheric state itself** (CO2
ppm, pressure — physically smooth quantities) between the two bins' current
state vectors, and *then* run RT once on the interpolated state. The
simpler and safer default for a first build: **no interpolation at all**,
nearest-bin assignment only (`G` chosen so bin width matches local
keystone amplitude) — every pixel's prediction comes from exactly one
bin's RT output, nothing is ever blended in radiance space, data side or
model side. (This distinction is not pedantic — see the note below.) Once
each pixel's own exact per-position, per-wavelength prediction exists, the
same `gaussian_blur_rows` is applied across the neighborhood's full
predicted image (a real, known, physical operator, not a numerical
convenience) before comparing to the real, PSF-blurred measurement
`A[i, j]`.

**Why this distinction matters (raised directly by the user, 2026-08-12):**
it's tempting to read this plan as "the same kind of thing `rectify()`
does, just done locally and smarter" — combine nearby pixels/values to
estimate the value at a target position. That's exactly the operation §9l/
§9m already found to be the single largest bias source in the whole study
(tens-of-ppm CO2 bias from bilinear-interpolating *measured radiances*,
present even on a perfectly uniform scene with nothing spatially varying to
blend). The joint-inversion approach only avoids that trap if it never
combines radiances/spectra (measured *or* predicted) directly — only
physical state may be interpolated, and RT must be re-run per pixel at
that pixel's own exact position afterward. Native retrieval already gets
half of this right (it never touches/combines real pixel values, which is
why it doesn't have rectify's bias); its failure mode (§11p, 44% capture)
is a *different* one — fitting one homogeneous atmosphere where several
different true atmospheres are actually spliced together. The joint block
fixes that specific error (by giving the fit `G` atmospheres instead of 1)
while preserving native's one correct property, as long as Phase 1 below
is actually built the state-space/nearest-bin way described here and not
the radiance-interpolation way an earlier draft of this section sloppily
described.

**This is not a new radiative-transfer model** — every single-pixel
prediction still goes through the exact same `ForwardModel`/`ILS`/
`gaussian_blur_rows` calls this whole codebase already uses, run once per
pixel at that pixel's own real position. What's new is only the
*bookkeeping*: `G` atmospheres solved jointly against every real pixel in
the neighborhood (not one atmosphere solved against one row's 158 spliced
columns), with the known `η(i,j)` map deciding which single state each
pixel constrains.

## 3. Jacobian, regularization, and the honest resolution limit

The Jacobian `K` (pixel × state) is mostly a **known, sparse gather
structure** (which `η`-bin(s) pixel `(i,j)` depends on, from the exactly-
known `η(i,j)` map) composed with the RT Jacobian each bin's own
`ForwardModel` already computes for a single atmosphere (existing GERT
machinery, unchanged), then passed through the same ILS-convolution and
`gaussian_blur_rows` linear operators already used in rendering. None of
that needs a new numerical method — `gert.retrieval.GERTRetrieval`'s own
optimal-estimation machinery generalizes directly if `K` is assembled this
way; the state vector is just bigger (`G` atmospheres instead of 1) and
each pixel contributes one row of `K` instead of only its own row's
atmosphere contributing.

**This will be underdetermined/ill-conditioned as `G` grows**, exactly
like any tomographic reconstruction — rows deep inside a run of high
keystone amplitude contribute highly correlated (nearly redundant)
constraints on overlapping `η` ranges, so `K` develops near-degenerate
directions. This is not a bug to engineer around; it's the actual, honest
information-content limit of the real optics — the same thing OE's
averaging kernel `A = (KᵀSy⁻¹K + Sa⁻¹)⁻¹KᵀSy⁻¹K` already diagnoses for a
single atmosphere, generalized here to `G` atmospheres. A smoothness prior
on `Sa` across `η` (correlation length tied to a physical footprint scale,
same spirit as the existing `h2o_scale`/`p_scale` priors) regularizes it,
and the resulting averaging kernel's row-sum/width **is** the real,
defensible answer to "what spatial resolution did we actually recover" —
a principled replacement for eyeballing capture-fraction numbers like
§11p's 44%/90%.

## 4. Phased build (reusing existing code, no new physics)

**Phase 0 — smallest possible extensibility test, before anything else.**
§1-3 establish that the physics/math is sound; they do *not* establish
that `gert.retrieval.GERTRetrieval`/`StateVector` — built for one
atmosphere per fit — can actually be extended to a block-sparse,
multi-atmosphere joint state without a from-scratch solver. That's the
real unknown, and it should be answered as cheaply as possible before any
larger investment:

- `G = 2` atmospheres only — the minimum that's a genuine joint problem,
  not a relabeled single-atmosphere fit.
- A handful of rows (~5), not a real 20-30 row block.
- A **synthetic, noise-free, idealized** two-state scene with a known
  closed-form right answer (e.g. a hard step in true CO2 between the two
  halves of the neighborhood's `η` range) — still rendered through the
  real `xy_to_wavelength_slit`/`ForwardModel`/ILS/PSF machinery, but not
  yet §11p's real, messy point source. The goal here is purely "does the
  code converge to the known right answer," decoupled from "does this
  help the real dilution problem."
- **Success criterion**: the `G=2` joint solve recovers both true states
  to within ordinary retrieval noise, and — the important control —
  rerunning the identical setup with `G=1` degrades back toward native's
  own already-measured dilution bias, confirming `G=2` is doing something
  native structurally cannot, not just a reparameterization.
- The deliverable here isn't a resolution number, it's an answer to "does
  `GERTRetrieval` extend this way, or does this need a new solver built
  alongside it" — which determines whether everything below is a moderate
  extension (days) or a separate solver build (much more).

Only once Phase 0 passes:

1. **Forward-operator function**: `predict_neighborhood(fpa, rows, states)`
   — given a list of `G` atmospheric states on an `η` grid and a row
   range, produce the predicted (PSF-blurred) sub-image, using
   `xy_to_wavelength_slit`, `ForwardModel`, `_diagonal_ils_convolve`,
   `gaussian_blur_rows` exactly as `gd_render.image()` does today. Sanity
   check: with `G=1` state covering the whole neighborhood's `η` range,
   this should reduce close to (not identical to, since real `η` varies
   continuously) native's own per-row rendering.
2. **Jacobian assembly**: finite-difference or reuse GERT's existing
   per-atmosphere analytic Jacobian, replicated per `η`-bin and gathered
   through the known `η(i,j)` weights — generalizing Phase 0's `G=2` proof
   to real `G`.
3. **Realistic diagnostic solve**: re-run exactly §11p's row-910 hot-spot
   case (FPA2, `results/gd_joint_fpa2.pkl`'s underlying scene), this time
   solving jointly for `G≈10-20` `η`-bin atmospheres across rows ~890-935
   instead of one atmosphere per row. Direct, apples-to-apples comparison
   against native's 44% and undistorted's 99% capture fractions from
   §11p — the clearest possible go/no-go signal for whether the joint
   approach is worth pursuing at realistic scale.
4. **Averaging-kernel characterization**: for a range of neighborhood
   widths / `G` / prior correlation lengths, compute the generalized
   averaging kernel and report its effective resolution (row-sum width) as
   a function of local keystone amplitude — the rigorous version of §11o's
   autocorrelation-based resolution question.
5. Only after 0-4 exist and are validated: consider ML, scoped narrowly
   (§5).

## 5. Where ML could still help, and how to keep it honest

Two narrow, defensible roles, both *given* the classical solver above
exists as ground truth to validate against — not a replacement for it:

- **Learned prior / regularizer.** Replace the hand-picked smoothness
  `Sa` with one learned from realistic-scene statistics (real plume
  shapes, real albedo covariance) — a much smaller, more constrained
  learning problem than "learn the whole inverse," since the deterministic
  optics (`K`) stay analytic and only the prior over plausible atmospheres
  is learned.
- **Amortized/fast approximate inversion.** Train a network to approximate
  the optimal-estimation solution for speed (e.g. real-time quicklook),
  with the classical solver as the training target and as a standing
  validation check on held-out/out-of-distribution scenes.

Both sidestep the failure mode flagged when this was first proposed: an
end-to-end black box trained to invert the FPA would be re-learning an
operator (`xy_to_wavelength_slit` + PSF) already known in closed form, and
its failure mode on scenes unlike its training set — e.g. a point source
narrower or more off-center than anything it saw in training — would be
the same silent-dilution problem §11p just found, except hidden inside a
black box instead of being as diagnosable as native's 44% number was.

## 6. Open risks / questions, not yet resolved

- **Cost — correcting an intuitive-but-wrong worry, then the real drivers**
  (discussed with the user 2026-08-12). It's tempting to assume RT has to
  run per *pixel* (to get a properly-centered ILS window), which would
  multiply cost by pixel count. That's not how it scales: `ForwardModel.
  run()` already produces one full-band hi-res spectrum per **atmosphere**,
  reused across every pixel assigned to it via cheap ILS convolution —
  exactly like native reuses one row's spectrum across its 1024 columns
  today. So RT+Jacobian cost scales with `G` (bins), not with pixel count,
  and since `G` is meant to be *smaller* than the row count it replaces,
  the joint approach's RT cost should be *less* than retrieving that many
  rows independently, not more. Anchor number: this session's FPA0
  realistic no-noise battery did 3072 single-atmosphere fits in 2509s on
  32 CPUs ≈ 26 CPU-seconds per converged single-atmosphere retrieval
  (`gd_test.py`'s own `.out` log). A `G≈15` block covering ~20-30 rows
  costs roughly `15 × 26s ≈ 6.5` CPU-minutes for the RT/Jacobian part —
  cheaper than retrieving those rows independently.

  The real cost drivers are elsewhere:
  - **Per-iteration linear algebra scales with total pixels in the block**
    (~20-30k for a 20-30 row block), not with `G` — forming/factoring
    `KᵀSy⁻¹K` where `K` is pixels × (`G`×state-size). State size stays
    small (`G`×~10 ≈ 150), so this is moderate, not extreme, but it's new:
    today's per-row fits never touch more than ~1024 channels at once.
  - **Loss of today's embarrassing parallelism.** `gd_test.py`'s
    `_worker()`/`Pool.imap_unordered` pattern parallelizes over
    independent rows trivially. A joint block is one coupled optimization
    problem — parallelism is only across blocks, not within one, and each
    block's solve is a bigger serial computation than any single row fit
    is today.
  - **Tiling choice directly multiplies cost.** Scaling to a full FPA
    needs ~1024/20 ≈ 50 blocks; tiled (cheaper, possible seams at block
    boundaries) vs. overlapping (smoother, directly multiplies cost by the
    overlap factor) isn't decided yet.
  - **Engineering is likely the dominant real cost, not FLOPs.**
    `GERTRetrieval`/`StateVector` are built for one atmosphere per fit;
    whether they extend to a block-sparse `G`-atmosphere state or need a
    parallel solver built alongside them is an open question — see Phase 0
    in §4, added specifically to answer this cheaply before committing
    further. Also needed and not existing anywhere in the repo today: the
    pixel→bin sparse gather structure, the cross-`η` smoothness prior,
    PSF-blur edge padding per block (the blur reaches a few rows past any
    block edge), block-stitching, and the averaging-kernel diagnostic
    (§3/§4 step 4).
  - **New, less-isolated failure modes.** Today, one bad row fails in
    isolation — 1023 others are unaffected. In a joint block, one
    poorly-constrained bin (too few pixels, near a block edge) can degrade
    or blow up convergence for the *whole* block, since the state is
    coupled through one shared nonlinear solve.
  - **Unresolved design choices are their own cost.** Is dispersion
    (currently a per-row nuisance parameter, §11k) shared across a block
    or still per-row? If per-row, `G` doesn't actually buy as many fewer
    unknowns as it looks like; if shared/smoothed, that's another new
    regularization structure to design before a real cost number is
    knowable at all.
- **Nonlinearity / relinearization.** GERT's forward model is nonlinear;
  a joint `G`-state solve needs the same Levenberg-Marquardt-style
  relinearization `GERTRetrieval` already does per row, just at `G`x the
  state size — convergence behavior at realistic `G` is unknown until
  tried.
- **Where `G` should actually come from.** Tied to `η`-bin width is a
  resolution *claim* — the averaging-kernel analysis in §3/§4 needs to
  come before any claim like "this recovers 2x native's resolution," not
  after.
- **Rectified precedent.** The existing "rectified" pipeline is the
  closest thing already built to this idea (shared-grid interpolation
  across rows) and is already known (§9l/§9m/§11i) to be the *worst*
  pipeline of the three, dominated by interpolation bias — worth
  understanding precisely why before assuming a joint inversion avoids the
  same trap. The likely answer: rectified interpolates the *rendered
  detector image* onto a shared grid (a post-hoc smoothing of already-
  spliced rows), while this plan inverts the *underlying state* jointly
  against the real, un-interpolated pixels — a materially different
  operation — but that distinction should be checked, not assumed.
