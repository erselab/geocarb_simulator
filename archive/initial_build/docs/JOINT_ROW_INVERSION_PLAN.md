# Plan: joint multi-row inversion to recover a decoupled spatial/spectral field from the distorted FPA

> **Archived 2026-08-19.** Superseded by `docs/PROJECT_STATUS.md`. This is
> the plan that motivated the pivot away from independent per-row (1D)
> retrieval, in response to `KEYSTONE_SMILE_BIAS_PLAN.md` §11o/§11p (also
> archived) — and unlike that pivot's own precursor, this one WAS carried
> forward: it's the origin of the joint-block/`StateSpec` architecture the
> current design is built on. Kept in place, unedited, for the record of
> that motivation.

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

### 0a. Go/no-go test — built and run, 2026-08-12: **go**

Built `scripts/gd_toy_trace_retrieve.py`. `traced_obs()` reuses
`gd_toy_trace_pixels.py`'s per-pixel evaluation to collect every real
`(row, column)` pixel within `--tolerance-km` of a target `eta0`, pulls
each one's real, unmodified value straight from `band["A"]` (rendered by
`gd_test.py`'s own `_band_setup`, unchanged), and hands the resulting
`(ν, y)` pairs to `gd_test._joint_retrieve()` exactly as `_native_row`
does today — same `GERTRetrieval`/`ForwardModel` call, no new solver, just
a different `obs_grid`. Ran §11p's exact row-910/row-880 hot-spot case,
FPA2, realistic scene, no noise, dispersion order 2, `--tolerance-km 0.5`:

| | pixels used | rows spanned | converged | χ²ᵣ | CO2 bias |
|---|---|---|---|---|---|
| peak (row 910) | 393 | 906–915 (10) | **yes** | 0.068 | −4.30 ppm |
| background (row 880) | 392 | 876–885 (10) | **yes** | 0.067 | −3.70 ppm |

True rise +3.97 ppm; trace-and-select retrieved rise +3.37 ppm —
**84.9% of the true peak enhancement captured**, against native's 44% and
undistorted's 99% (§11p). Both fits converged cleanly (good χ²ᵣ, no
dispersion/nuisance-parameter fragility despite only ~390 real pixels)
using ~40% of native's 1024-column data.

**Go/no-go verdict: go.** Trace-and-select nearly doubles native's
recovered signal without any new solver, Jacobian-gather structure, or
`G`-atmosphere state — just a different, real, un-interpolated pixel
selection fed into existing retrieval code. It doesn't fully close the gap
to undistorted (84.9% vs. 99%), consistent with the fact that
trace-and-select still admits up to 0.5 km of real position error per
pixel while undistorted has none at all — a real, expected, bounded
residual, not a red flag.

### 0b. Whole-slit sweep — built and run, 2026-08-12: real improvement on average, with a real, specific, predictable failure mode

The two-row go/no-go test in §0a can't show whether 84.9% generalizes or
is a property of that one hot spot. Built `scripts/gd_trace_retrieve_sweep.py`
(parallelized the same way `gd_test.py` parallelizes its own row
batteries — `mp.get_context("fork")` + `Pool.imap_unordered`, no new
solver) and ran all 1024 rows, FPA2 realistic scene, no noise,
`--tolerance-km 0.5`: **1024/1024 converged**, 162s wall-clock on 16
workers (~0.16 s/row). Compared against native/undistorted's existing
`results/gd_joint_fpa2.pkl` via `scripts/gd_trace_retrieve_plot.py`
(`plots/gd_trace_vs_native_fpa2.png`).

**Whole-slit robust stats, CO2 bias [ppm]:**

| pipeline | mean | std | median | MAD-σ |
|---|---|---|---|---|
| native | −2.22 | 2.25 | −2.85 | 0.88 |
| undistorted | −0.85 | 1.79 | −1.28 | 1.16 |
| trace-and-select (0.5 km) | **−1.72** | **2.75** | −2.55 | **1.72** |

On average the systematic bias (mean, median) sits between native and
undistorted, consistent with §0a — but scatter is **higher** than both
(std 2.75 vs. native's 2.25; MAD-σ 1.72 vs. native's 0.88). The two-row
test couldn't see this; it's a real cost, not noise in the estimate.

**Where it goes wrong, diagnosed concretely.** The comparison plot shows
a second, broader CO2 feature near row ~400 (a different plume than
§11p's point source) where trace-and-select *overshoots*: row 394 hits
**+9.03 ppm** bias vs. native's +4.21 and undistorted's +3.28, against a
true rise of only ~+4 ppm — worse than native, not better. Checked why:
only **4 distinct rows** contribute pixels there (`rows_crossed≈4.0` at
that row gives a much narrower reach than row 910's ~9.3), and true CO2
barely varies across those 4 rows (416.87–416.94 ppm, 0.07 ppm total) —
so it is *not* a mixing-genuinely-different-truths problem. It matches
the dispersion/nuisance-parameter risk flagged in §0 directly: too few
distinct rows means too little spectral coverage across the band to
constrain the fit properly, even though χ² still looks fine (a real,
underdetermined-fit failure mode, not a divergence one).

**Quantified across the whole sweep**: 230/1024 rows are meaningfully
worse than native (|bias| higher by >0.5 ppm), 353/1024 meaningfully
better, the rest comparable. Of the 230 worse rows, **224 have ≤6
distinct contributing rows** (the sweep's own median is 6) — the failure
mode is real but specific and predictable, not scattered randomly across
the slit. It shows up exactly where the theory said it would.

**Revised verdict**: still a genuine net positive (353 better vs. 230
worse, and the mean bias improves), but not a clean win everywhere —
trace-and-select needs a minimum-contributing-rows floor (widen the
tolerance or window adaptively when too few rows would otherwise
contribute, rather than accepting a narrow, poorly-covering trace) before
it's a safe drop-in replacement for native. `results/gd_trace_fpa2_tol0.5.pkl`
(gitignored) and `plots/gd_trace_vs_native_fpa2_tol0.5.png` have the full
data; `plots/gd_trace_all_pipelines_fpa2_tol0.5.png`
(`scripts/gd_trace_all_pipelines_plot.py`) adds rectified for the full
four-pipeline picture (own panel/scale — its bias here is ~28x native's
magnitude, §9l/§9m's already-known interpolation artifact, unrelated to
this experiment).

**0.2 km tolerance, run and compared the same way, 2026-08-12**: also
1024/1024 converged (73s, even faster than 0.5 km — fewer pixels per row
on average). Tightening tolerance made things *worse*, not better —
confirms the mechanism directly rather than just plausibly explaining it:

| tolerance | mean | std | median | MAD-σ |
|---|---|---|---|---|
| 0.5 km | −1.72 | 2.75 | −2.55 | 1.72 |
| 0.2 km | −1.76 | **3.33** | −2.46 | **2.20** |

Same mean (no systematic change), but materially higher scatter. The
row~400 overshoot gets worse (+12.5 ppm vs. 0.5 km's +9.0 ppm), and a new
large excursion appears near rows 130-220 (down to −7.5 ppm) that wasn't
prominent at 0.5 km. Consistent with the diagnosis above: a tighter
tolerance means fewer rows clear it on average, which means more
underdetermined fits, not more accurate ones — tightening the position
constraint without also ensuring enough rows contribute makes the failure
mode more common, it doesn't fix it. `results/gd_trace_fpa2_tol0.2.pkl`,
`plots/gd_trace_vs_native_fpa2_tol0.2.png`,
`plots/gd_trace_all_pipelines_fpa2_tol0.2.png`.

**Housekeeping note**: the first pass of these two plotting scripts wrote
to a fixed filename regardless of `--tolerance-km`, so rerunning at 0.2 km
silently overwrote the 0.5 km figures (caught immediately — git still had
the 0.5 km version staged, nothing was actually lost, but it's exactly
the kind of un-parameterized-output-filename bug this project has hit
before, e.g. the autocorrelation script in §11o of
`KEYSTONE_SMILE_BIAS_PLAN.md`). Both scripts now suffix their output with
`_tol{tolerance_km}`, matching `gd_trace_retrieve_sweep.py`'s own
`.pkl` naming, which already did this correctly.

**Not yet done**: implement the minimum-row floor and rerun; check
whether the joint block (§1-§4) — which uses *all* pixels with proper
weighting rather than a binary include/exclude threshold — avoids this
specific failure mode by construction, which would be a concrete,
evidence-backed reason to still build it rather than stop at
trace-and-select.

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

**GERT inspection, 2026-08-12 — Phase 0's own question answered directly
from the source, without needing to build and run it first.** Read
`gert/retrieval.py` (`StateVector`, `GERTRetrieval`, `DecoupledGERTRetrieval`)
rather than guess:

- `StateVector.apply(prior_atm, prior_albedo, prior_albedo_slope)` maps
  the flat state array onto exactly **one** `AtmosphericProfile` — every
  transform (`{mol}_scale`, `p_scale`, `albedo_{b}`, ...) is a scalar
  multiplier/offset on that one profile. No indexing, no per-location
  structure, anywhere in the class (~1300 lines).
- `GERTRetrieval.__init__` takes exactly one `fm_prior: ForwardModel` and
  one `state_vector: StateVector`. A large, mature single-location engine
  (backtracking, per-transform analytical Jacobians, aerosol, EOF
  empirical corrections) — built around one atmosphere throughout.
- `DecoupledGERTRetrieval` (name looked promising) turned out to be a
  *different* decoupling axis — atmosphere-vs-BRDF-surface for
  reflectance imaging (EMIT/AVIRIS-style, cached path terms reused across
  surface-only iterations), same theme as `decoupled_forward_model.py`.
  Not applicable to multi-location.
- Checked whether the Gauss-Newton step is factored out separately from
  the single-atmosphere plumbing (`_forward`, `_jacobian_fd`,
  `_jacobian_mixed`, `run()`): it isn't. The GN math lives directly inside
  `run()`'s iteration loop, tightly coupled to `self.sv`/`self.fm_prior`.
  No standalone "step given `(K, y, Sy_inv, Sa)`" utility to lift out.

**Answer: no in-place extension.** `GERTRetrieval` does not generalize to
a multi-atmosphere state, and subclassing/monkey-patching its internals to
fake one would be invasive and fragile against future GERT changes. What
*is* cleanly reusable, unchanged, called once per bin: `ForwardModel`
(atmosphere-agnostic at construction) and `StateVector.gas_scaling()` +
`.apply()` — exactly what `_joint_retrieve` already calls per row today.

**Revised first step** (supersedes "build and run Phase 0 to find out"):
write a small, standalone Gauss-Newton/optimal-estimation loop — new code
in `geocarb_gert`/`scripts`, **not** a `gert` modification — that builds
`G` independent `(ForwardModel, StateVector)` pairs, assembles the
block-sparse Jacobian/residual across all real pixels by hand (the
pixel→bin gather from §2), and does its own Rodgers-form update
`dx = (KᵀSy⁻¹K + Sa⁻¹)⁻¹(KᵀSy⁻¹·resid − Sa⁻¹·(x−x_a))` — standard,
well-understood linear algebra, not a re-derivation of anything GERT
doesn't already provide per-bin. Phase 0's `G=2` synthetic test (below)
is still the right first thing to build and run — it now has a known
answer for *how* to build it, not just *whether* it's buildable.

**Phase 0 result, 2026-08-12 — built and run: pass.**
`scripts/gd_toy_joint_g2_test.py` implements exactly the revised design
above: a standalone finite-difference Gauss-Newton loop (no `gert`
modification, `GERTRetrieval` not involved) built around two reusable,
unmodified pieces — `StateVector.gas_scaling()`+`.apply()` to turn a
scalar `co2_scale` into a modified `AtmosphericProfile`, and a fresh
`ForwardModel(...).run()` per evaluation, exactly `GERTRetrieval.
_forward()`'s own recipe. A `local_render()` helper replicates `gd_render.
image()`'s per-row loop (`xy_to_wavelength_slit` → per-row ILS convolution
→ `gaussian_blur_rows`) restricted to a padded window, serving as a
working draft of Phase 1's own `predict_neighborhood()`.

Setup: FPA2, row 512 ± 2 (5 core rows, padded ±4 rows for correct PSF-blur
edge handling), a synthetic hard truth edge at `η(row=512, col=512)`
(`geocarb_gert.focalplane.edge_scene`, `softness=0` — all edge blur in the
rendered image comes from the real 1.5-px spatial PSF, not the synthetic
scene), true `co2_scale` = 0.95 (A, η<edge) and 1.05 (B, η>edge) against
the shared prior, no noise. This window genuinely exercises joint/coupled
machinery rather than trivially decomposing into two independent
single-bin fits: at row 512, `rows_crossed≈5.24` is comparable to the
window width, so individual rows' own column ranges straddle the eta_edge
boundary (within-row keystone splicing), and the real PSF blur further
mixes rows on both sides of the boundary.

Results:
- **G=2** (joint fit of `co2_scale_A`, `co2_scale_B`): converged in 2
  Gauss-Newton iterations, recovered `co2_scale_A=0.95000` (true 0.95,
  error 0.00000) and `co2_scale_B=1.05000` (true 1.05, error -0.00000) —
  essentially exact recovery (residual rms → 8e-8 by the final iterate,
  floating-point noise floor for a noise-free synthetic test).
- **G=1 control** (single shared `co2_scale` fit to the identical
  pixels): converged to `co2_scale=1.00559` — near the pixel-fraction-
  weighted dilution expectation (~0.999, from the core window's 49.0% of
  pixels falling on the B side of the edge), far from *both* true values
  (0.95 and 1.05) and nowhere near either endpoint. This is the same
  structural failure mode native's per-row-independent retrieval already
  exhibits at real hot spots (§11p) — confirmed here in a controlled,
  closed-form setting.
- **Verdict: both success criteria met.** `G=2` recovers both true states
  to well within ordinary retrieval noise; `G=1` on the same data degrades
  to a diluted value, confirming `G=2` is doing something native/G=1
  structurally cannot, not just a reparameterization of the same fit.

This closes Phase 0. The standalone GN-loop approach (not a `GERTRetrieval`
extension) is validated as buildable and correct on the simplest possible
case; Phase 1's `predict_neighborhood()` + real Jacobian assembly can now
proceed with confidence in the underlying mechanism.

Only once Phase 0 passes:

1. **Forward-operator function**: `predict_neighborhood(fpa, rows, states)`
   — given a list of `G` atmospheric states on an `η` grid and a row
   range, produce the predicted (PSF-blurred) sub-image, using
   `xy_to_wavelength_slit`, `ForwardModel`, `_diagonal_ils_convolve`,
   `gaussian_blur_rows` exactly as `gd_render.image()` does today. Sanity
   check: with `G=1` state covering the whole neighborhood's `η` range,
   this should reduce close to (not identical to, since real `η` varies
   continuously) native's own per-row rendering.
2. **Jacobian assembly**: finite-difference (simplest, always available)
   or reuse GERT's own per-atmosphere analytic Jacobian machinery
   *per bin* (each bin's own `GERTRetrieval`-style Jacobian column block,
   computed independently) — replicated per `η`-bin and gathered through
   the known `η(i,j)` weights into the combined block-sparse `K`, then fed
   to the standalone GN loop above, not to `GERTRetrieval` itself.
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

**Phase 1-3 result, 2026-08-12 — built and run: real, substantial
improvement, comparable to trace-and-select.**

Implementation, split at the same architecture boundary the rest of this
codebase already uses (`geocarb_gert` = rendering/geometry, `gert` =
untouched, `scripts/` = retrieval logic):

- `geocarb_gert.nearest_bin_scene(bin_centers, spectra)` — added to
  `focalplane.py` alongside `edge_scene`/`barcode_scene`. `G`-way
  generalization of `edge_scene`: hard nearest-bin assignment, never a
  blend of two bins' spectra (the §9l/§9m trap §2 already warned about).
- `geocarb_gert.gd_render.predict_neighborhood(fpa, rows, wn_hires,
  radiance, ils, pad=4)` — added to `gd_render.py` alongside `image()`.
  Item 1's own forward operator, exactly `image()`'s per-row loop
  restricted to a padded row window. **Sanity check passed exactly**: on
  a uniform scene, `predict_neighborhood(rows=505..520)` matches
  `image()`'s own output for those rows to `0.000e+00` max abs
  difference (bit-identical) — `pad=4` is more than sufficient for the
  1.5-px PSF's kernel half-width.
- `scripts/gd_joint_block_retrieve.py` — items 2-3, the standalone GN/OE
  loop (not `GERTRetrieval`), reusing `StateVector.gas_scaling()`+
  `.apply()` and `ForwardModel` per bin exactly as Phase 0 validated, now
  with `G` bins and a first-difference Tikhonov smoothness prior on
  adjacent bins' `co2_scale` (`Sa⁻¹ = γ·LᵀL + I/σ_abs²`, §3's own
  regularization, needed because `G` bins this large is underdetermined
  by construction).

**A real bug caught and fixed before the first realistic result.** The
first attempt held every bin at one *shared* prior atmosphere
(`atm_center = atmosphere_at(0.0)`, Phase 0's own convention) with only
`co2_scale` free. On the real scene this failed badly: a small 5-bin/
11-row trial converged to a >1000% "peak-enhancement captured" number —
obviously unphysical, and the residual stalled around 0.5-0.7 (barely
better than the `x=1` starting guess) regardless of `G=1` vs `G=5`. Cause:
unlike Phase 0's synthetic scene, the *real* along-slit truth
(`als.atmosphere_at(x_km)`) varies CH4/H2O/CO/surface-pressure
continuously too, not just CO2 — holding those at one shared reference
value left the fit no way to explain that real variation except by
distorting CO2 instead. **Fix**: give each bin its own local *true*
nuisance-gas atmosphere, `als.atmosphere_at(x_km_bin)` — the same
function that generated the real image's per-row truth — with only that
bin's `co2_scale` retrieved relative to it. This isolates exactly the
question this test asks (does bin-splicing recover CO2's keystone-diluted
gradient) from a separate, larger question (jointly retrieving nuisance
gases too, deferred). After the fix, residuals converge properly (rms
~0.001, not stuck at 0.5) and results are physically sensible. Documented
in the script's own docstring, not just here.

**Result, rows 890-935 (46 rows), `G=15` bins, `γ=3.0`:**

| | peak-enhancement captured |
|---|---|
| native (§11p) | 44% |
| joint block, `G=15` | **81.7%** |
| trace-and-select, tol=0.5km (§0a) | 84.9% |
| undistorted (§11p) | 99% |

Converged in **2 Gauss-Newton iterations**, ~81s wall time. Per-bin
recovered CO2 matched local truth to <0.1 ppm at every one of the 15
bins except right at the hot spot itself, where the discrete `G=15`
grid under-resolves the peak's own narrow width (true rise +3.973 ppm,
retrieved rise +3.245 ppm — the residual gap from 100%). The `G=1`
control on the identical pixels recovered 413.77 ppm — diluted, far
below the true peak (416.98 ppm), the same structural failure native
exhibits, reproduced here as the requested control.

**Regularization-strength check**: reran at `γ=0.3` and `γ=30` (100x
range) — captured fraction unchanged at 81.7-81.8% either way. In this
noise-free test, `G=15` over a 46-row/47k-pixel window is apparently not
close enough to the underdetermined regime for the smoothness prior to
matter much — the real pixel count (even after accounting for
correlation) is ample at zero noise. §3's warned-about degeneracy would
likely need either realistic noise added or `G` pushed substantially
higher to actually bite; not yet tested.

**Verdict**: the joint block is a real, working improvement over
native's own per-row retrieval, closing most of the gap to undistorted
and landing close to trace-and-select's own already-validated number —
on this specific hot spot, at comparable cost (2 GN iterations, ~81s, vs.
trace-and-select's near-instant per-row solve). It does *not* yet clearly
beat trace-and-select (81.7% vs 84.9%) enough to justify the much larger
engineering investment of a fully free per-bin nuisance state and a
whole-slit sweep — that comparison is the natural next step before
investing further here (§6 already lists "trace-and-select vs. joint
block, head to head" as an open question; this is now partially
answered, at one hot spot, with nuisance parameters idealized away on
the joint-block side).

**Not yet done**: whole-slit sweep (only one hot spot tested so far);
realistic-noise version (this and everything above is still the
noise-free convention this whole study has used); jointly-retrieved
nuisance gases (currently idealized to local truth); averaging-kernel
characterization (item 4 above); `G` sweep to find where regularization
starts to matter.

**Diagnostics, 2026-08-12 — built `scripts/gd_joint_block_diagnostics.py`,
requested before running the whole-slit sweep.** Two additions:

- **Spectral-residual inspection**: reshape `resid = y_true - forward(x)`
  back to `(rows, cols)` image space and look at its structure directly,
  plus a per-bin reduced chi-square (grouping real pixels by their own
  nearest-bin assignment, same convention as `_chi2`/`chi2_outlier_mask`
  elsewhere in this codebase). This is the diagnostic that survives once
  real, noisy, truth-unknown data eventually replaces this synthetic
  scene — unlike the capture-fraction number, which needs known truth and
  won't be available then. On the row 890-935/`G=15` case: no sharp
  staircase pattern at bin boundaries (the quantization-error signature
  originally hypothesized) — residuals show a smooth diagonal band
  instead, more consistent with a line-shape/dispersion residual than
  under-resolved `G`.
- **Bin placement**: the original `np.linspace`-in-eta uniform grid was
  replaced with bin centers placed at quantiles of the *actual per-pixel
  eta distribution* in the window (`pixel_density_bin_centers`) — not a
  `rows_crossed`-based proxy (an earlier version of this function used
  one, routed through each row's own center-column eta; it needed an
  extra floor to handle a near-null row's real internal wiggle reading as
  zero keystone, and only approximated what's already directly
  available). Wherever keystone is large, more rows' column ranges
  overlap a given eta interval, so real pixel density is genuinely
  higher there — quantile spacing on the real per-pixel data reflects
  this directly, no proxy needed.

**Result on the same row 890-935/`G=15` case**: pixel-density bins beat
uniform on every metric — capture 90.9% vs 81.7%, mean per-bin chi2
1.87e-8 vs 2.58e-8 (lower/better), minimum pixel count per bin 890 vs 269
(much better-conditioned). But the two schemes' bin *placements* are
nearly indistinguishable in this window (visible in the diagnostic
figure) — because `rows_crossed` only spans 9.09→9.55 across these 46
rows, i.e. this window sits entirely in FPA2's high-keystone regime with
no real keystone *contrast* for density-weighting to respond to. The
scheme is working (better chi2/conditioning even from a small placement
shift), but this window is not a real test of the "few bins near-null,
many bins high-keystone" hypothesis — that needs a window (or the
whole-slit sweep itself) spanning both regimes. Not yet run.

Figure: `plots/gd_joint_block_diagnostics_fpa2_row890-935_G15.png`.

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
