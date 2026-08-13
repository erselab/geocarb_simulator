# Plan: migrating the joint multi-atmosphere block into `geocarb_gert`, and what that migration needs to accommodate

Sketch only — not started. Written 2026-08-13, after the whole-slit sweep
(`scripts/gd_joint_block_whole_slit_sweep.py`), the hi-res forward model
(`scripts/gd_joint_block_hires_test.py`), and the keystone-free ceiling
test (`scripts/gd_joint_block_keystone_free_sweep.py`) — see
`JOINT_BIN_RETRIEVAL_ATBD.html` for the method itself — had all
accumulated as one-off scripts in `scripts/`, each importing from the
others via `sys.path.insert(scripts/)` rather than real package imports.
This document plans (a) which pieces move into `geocarb_gert` and how
they're organized, and (b) how that organization needs to anticipate
three follow-on goals — noise, an expanded state vector, and multi-FPA
joint retrievals — so the migration isn't immediately obsoleted by the
next round of work.

## 1. What exists today, and why it needs a home in `geocarb_gert`

Following the precedent already set: `nearest_bin_scene` lives in
`geocarb_gert/focalplane.py` and `predict_neighborhood` lives in
`geocarb_gert/gd_render.py` — general forward-model primitives already go
in `geocarb_gert`; only orchestration (`main()`, argparse, which rows/G
to run, figures) stays in `scripts/`. Today's joint-block code hasn't
followed that split: `gd_joint_block_hires_test.py` imports from
`gd_joint_block_retrieve.py`, `gd_joint_block_whole_slit_sweep.py` imports
from both, and `gd_joint_block_keystone_free_sweep.py` imports from
`gd_joint_block_retrieve.py` again — an informal, ad hoc dependency graph
between files that all live in the "one-off scripts" folder.

**New module: `geocarb_gert/joint_block.py`** (flat file, matching the
package's existing one-file-per-concern layout: `focalplane.py`,
`gd_render.py`, `gd_polynomials.py`, `along_slit_scene.py` all coexist as
flat modules), to hold:

- `band_basics`, `make_spectrum_fn` — spectrum-generating infrastructure
- `build_forward`, `build_forward_hires`, and the two `_no_keystone`
  variants — the forward-model constructors
- `gauss_newton_regularized` — the standalone GN solver
- `pixel_density_bin_centers`, `uniform_bin_centers`, `bin_assign`,
  `per_bin_chi2` — bin placement/scoring
- `build_window_tiles` — adaptive window tiling
- `stitch_window_results` — the "reconstruct a whole-slit profile/bias
  from a sweep results dict" logic, currently copy-pasted three times
  (`gd_joint_block_whole_slit_plot.py`, `gd_joint_block_vs_native_plot.py`,
  and again inline in `gd_joint_block_keystone_free_sweep.py`)

`predict_neighborhood_no_keystone` goes into `gd_render.py` itself, next
to `predict_neighborhood` — same family, same file.

Everything that stays in `scripts/`: `main()`, argparse, multiprocessing
`Pool` setup, pickle I/O, and all plotting/figure code — now importing
cleanly from `geocarb_gert.joint_block` instead of each other.

One small real cleanup falls out of the move: `_eta_of` (col=512
convention, used everywhere in this line of work) and
`gd_render.real_row_eta` (col=511.5 convention) compute the same quantity
slightly differently — worth reconciling on one canonical function during
the move rather than carrying two conventions forward.

**Risk, and how to bound it.** This touches ~7 already-validated files.
The risk isn't design risk, it's silent behavior drift during the
mechanical move — exactly the class of bug this project has hit before
(wrong column convention, unparameterized output filenames). Mitigation:
treat the move as pure cut-and-paste-and-reimport first, then re-run the
row 890-935, G=15 case afterward and diff the printed numbers against
what's already recorded, before building any of §2-§4 below on top of it.

## 2. Why the module's shape matters now, not later

The three goals below aren't independent add-ons — they change what the
*state vector*, the *forward model*, and the *regularizer* need to look
like structurally. Building `joint_block.py` narrow (today's co2-only,
single-FPA, noise-free path) and generalizing later means either breaking
the published API immediately after writing it, or carrying two competing
signatures. Better to size the core abstractions correctly once, even
though only the narrowest path through them is exercised today. The
regularizer in particular should be a single generic mechanism driven by
a per-state-element table (§4.0), not a set of hand-picked groups grown
one at a time as new goals arrive.

## 3. Goal 1 — adding noise

Cheapest of the three; mostly plumbing that already exists elsewhere in
the codebase, not new machinery.

- `gd_test.py`'s `_band_setup` already has the real per-pixel noise model:
  `geocarb_noise_model(fpa)` (a `LinearShotNoise` calibrated from
  `geocarb_gert.radiometry.RADIOMETRIC_SPEC_BY_FPA`, `sigma =
  sqrt(N0**2 + N1*|A|)`) and `rng.normal(0, sigma_map)` injection. The
  joint block currently uses a flat, scene-independent `Sy_inv_diag =
  1/mean(|y|)**2` — a known, explicitly-flagged placeholder in
  `gd_joint_block_retrieve.py`'s own docstring, not the real model.
- Swapping in the real per-pixel sigma (and the real
  `geocarb_noise_model_multi` for the eventual multi-FPA case, §5) is a
  direct substitution — no new architecture.
- The real consequence is downstream: once noise is nonzero, a single
  deterministic solve no longer characterizes bias on its own — separating
  bias from variance needs multiple noise realizations per window (this is
  the ATBD's own flagged-but-undone open item). So the sweep runner needs
  a realization axis (`--n-realizations` / seed loop), and the results
  schema needs to store either a list of solves per window or a
  realization index folded into the results dict key, not just one
  `x_coarse`/`x_hires` per window as today.
- Once noise is real, `gamma` (the smoothness-regularization strength)
  stops being a free lunch — the earlier gamma-insensitivity result
  (`0.3/3.0/30 -> 81.7/81.7/81.8% capture`, ATBD §8.1) was a smooth-truth,
  noise-free artifact, not a general property. A real bias-variance
  tradeoff opens up once noise is present, and `gamma` will likely need
  re-tuning (e.g. an L-curve or cross-validation sweep) once this lands —
  a methodology task, not a plumbing task, flagged here so it isn't a
  surprise later.

## 4. Goal 2 — expanding the state vector (nuisance gases, albedo, surface pressure, dispersion)

### 4.0 Regularization as a per-element property, not per-group code

**Correction (2026-08-13, user)**: the first draft of this section handled
regularization by sorting parameters into hand-picked groups (gases get a
smoothness prior with one shared `gamma`; albedo needed a special-cased
"different, weaker prior" carve-out once §4's albedo correction landed;
dispersion gets no smoothness at all). The user's objection: don't hard
code one correlation treatment for one collection of elements and then
bolt on more special cases as new elements arrive. That's the right call
— the fix is a single generic mechanism, not a growing set of buckets.

**The mechanism.** Give every *named* state element its own entry in a
small declarative table, not a hand-picked group:

- a position array in eta (for a per-bin quantity, its bin centers; for a
  quantity shared across a window, a single point; for a quantity shared
  per-FPA, one point per FPA)
- a prior standard deviation `sigma`
- a correlation length `corr_length` (in eta or physical km, not "number
  of adjacent bins" — bin-spacing-independent, unlike the current
  first-difference operator)

One generic function builds a quantity's own prior covariance block from
those three numbers — an exponential correlation kernel,
`Sa_block[i, j] = sigma**2 * exp(-|eta_i - eta_j| / corr_length)` — and
the full prior is the block-diagonal stack of every quantity's own block,
inverted block-by-block (cheap; no giant dense solve, since quantities
are assumed prior-independent of each other for now — a possible later
extension, not needed yet). Today's first-difference Tikhonov `L^T L` is
a special case of this (nearest-neighbor coupling with an implicit
"correlation length" tied to bin spacing rather than a real physical
distance) — this generalizes it rather than replacing it with something
unrelated.

This directly answers the question: yes, every element gets its own
correlation length, and the code has exactly one path — assembling one
block per table entry — rather than one path per parameter *type*. Adding
a new quantity later (a second surface-pressure-like scale, a different
gas, whatever comes after this round) means adding one row to the table,
not a new branch in `gauss_newton_regularized`. It also quietly resolves
something the first draft treated as an open problem: a quantity's own
bin/position *resolution* is just part of its table entry too — albedo
doesn't need a special "finer bins than the gases" carve-out, it's simply
a table row with its own (probably denser) position array and its own
(probably short) `corr_length`, exactly the same as every other row.

The actual numbers — `corr_length` per quantity, in particular albedo's,
per the MODIS/OCO-2 coefficient-of-variation point from the prior
correction — are still unset; this section fixes the *mechanism*, not the
values. Filling in real numbers is future work once that data is in hand.

### 4.0a Extending the mechanism to within-bin correlation (GERT profile retrieval)

Everything above correlates one scalar-per-bin quantity *across* bins
(along the slit). GERT's own profile retrieval
(`StateVector.gas_profiles(corr_length=...)` / `correlated_Sa()`, in
`gert/retrieval.py`) already does the same kind of thing on a different
axis: a gas's concentration *per vertical level*, correlated across
levels via `Sa[i, j] = sigma_i * sigma_j * exp(-|level_i - level_j| /
corr_length)` — the identical exponential-covariance kernel as §4.0's
along-slit mechanism, just indexed by layer instead of eta. Combining the
two (a profile retrieved *per bin*, correlated both along the slit and
vertically within each bin) doesn't need new math, just a generalization
of what a table row *is*.

Instead of one position array per row, a row carries a **list of axes**
(along-slit, vertical level, per-FPA, whatever else comes up), each with
its own positions / `corr_length` / kernel, and the row's full prior
covariance is the **Kronecker product** of its axes' own covariance
matrices — a standard separability assumption (no reason vertical
correlation structure should depend on where you are along the slit, or
vice versa). Concretely:

- today's `co2_scale`: one axis (along-slit bins) — a 1-axis row.
- a profile-retrieved `co2_profile`: **two** axes — along-slit bins (eta
  positions, `corr_length_slit`, §4.0's kernel) times vertical levels
  (layer index, `corr_length_vertical`, *GERT's own `correlated_Sa()`
  kernel, reused directly rather than reimplemented*). Block size
  `G x N_levels`.
- dispersion (shared scalar) and per-FPA-per-bin albedo (§5): still just
  different axis combinations of the same row object — one axis of length
  1 for a shared scalar, two axes (along-slit x per-FPA) for per-FPA
  albedo.

So §4.0's table doesn't need to change shape to accommodate this — "one
position array" generalizes to "one or more axes," and every case
discussed in this document (scalar-shared, per-bin, per-bin-per-FPA,
per-bin-per-level) is a different axis count/combination of the same
object, not a new code path.

**Real cost consequence, not glossed over.** Today, one bin's spectrum
only needs recomputing when that bin's single scalar changes — `G+1`
forward calls per GN iteration (§4.1). A profile-retrieved bin has
`N_levels` free parameters that all require re-running RT together (a
spectrum can't be decomposed per-level), so a profile-gas block costs
`G * N_levels` forward calls per iteration under finite differences — a
real jump, not just a bigger constant under finite differences — see §7
for how this gets avoided: `ForwardModel.run(jacobians=True)`'s own
`tau_gas_layer_hires` gives exact per-level analytic sensitivities on the
hi-res grid, so a profile block never needs finite-differencing at all,
regardless of `N_levels`.

### 4.1 Two different difficulty tiers for the `StateVector` plumbing itself

Confirmed by reading `gert.retrieval` directly rather than assuming.

**Easy tier — gases, albedo, surface pressure.** `StateVector.gas_scaling`
/`.apply()` already carries `p_scale` and `albedo`/`albedo_slope` as full
state elements on *every* call the joint block makes today —
`make_spectrum_fn`'s `spectrum_for` just never sets them to anything but
their prior, only `co2_scale` gets touched before `.apply()`. So making
more gases, albedo, and surface pressure free is a mechanical
generalization of existing plumbing, not new machinery:
`spectrum_for` needs to accept a dict of scale values instead of one
scalar. "Easy" here describes the `StateVector` plumbing only — where
each new parameter *lives* (per-bin vs. shared) is a real design question,
and the original draft of this plan got it wrong for albedo (see below).

Under §4.0's table-driven mechanism, "where a parameter lives" is just
its position array, with no separate design decision or code path needed:

- Gas/pressure scales vary physically along the slit (that's the whole
  point of the along-slit bin structure) — table rows with one position
  per bin, `corr_length` set to whatever along-slit smoothness scale is
  physically reasonable for that gas (H2O's natural scale has no reason
  to match CO2's — different row, different number, same mechanism).
- **Correction (2026-08-13, user)**: an earlier draft of this section put
  albedo in a "shared per window" group with dispersion, reasoning it was
  an instrument/scene property with no particular reason to vary in eta.
  That reasoning was backwards. Real surface albedo is *more* spatially
  variable along the slit than well-mixed gas concentrations, not less —
  coefficients of variation computed from MODIS surface reflectance and
  OCO-2's own per-sounding albedo retrievals show this directly: albedo is
  a surface-cover property (bare soil vs. vegetation vs. water, field
  boundaries) that can change sharply within a single window's footprint,
  unlike a long-lived, well-mixed gas like CO2. Under §4.0's mechanism
  this doesn't need a carve-out — albedo is simply a table row with its
  own position array (likely denser than the gas bins) and its own short
  `corr_length` (likely much shorter than a gas's, and not yet measured —
  real MODIS/OCO-2 numbers should set it directly rather than guessing).
- Dispersion is a pure instrument-calibration property (how the real
  wavelength grid deviates from nominal) with no scene dependence at all
  — a table row with a single position per window (or per FPA, in the
  multi-FPA case, §5) and no meaningful correlation length, since there's
  only one point.

So the state vector becomes block-structured —
`[co2_scale_1..G_co2, ch4_scale_1..G_ch4, ..., albedo_1..G_albedo,
disp_a1, disp_a2]`, each block's own length set by its own table entry —
and `gauss_newton_regularized`'s current single first-difference `Sa_inv`
(sized to one flat `n`, uniform across the whole vector) becomes the
block-diagonal stack of covariance blocks §4.0 describes, one per table
row.

Jacobian cost scales accordingly: today's G+1 forward evals per GN
iteration (thanks to per-bin spectrum caching, only the one perturbed
bin's spectrum is recomputed) becomes roughly `(sum of every table row's
own block length) + 1` — the sum of all the gas blocks' `G_gas`, plus
albedo's own (likely larger) `G_albedo`, plus the handful of single-point
dispersion rows. A real but bounded cost increase, since each Jacobian
column is still only ever one bin's spectrum recompute, however many
table rows there are.

**Hard tier — dispersion.** Not just another `StateVector` element.
Traced through `gert.instrument`/`gert.forward_model`: dispersion shifts
the *output* channel centers (`SpectralWindow.dispersion_centers`),
applied when `ForwardModel.run(dispersion=...)` convolves onto the
*nominal* instrument grid (`win.dispersion_centers(coeffs)` in place of
`wn_instrument`). The joint block's own renderer currently convolves
straight onto the real, exact GD-polynomial wavenumber
(`_diagonal_ils_convolve(wn_hires, S_row, nu_row, ils)`, with `nu_row`
from `xy_to_wavelength_slit` — i.e. it already "knows" the true dispersion
by construction, for both the synthetic truth and its own forward model).
Retrieving a dispersion correction only means something if the forward
model instead convolves onto a *nominal* grid it must correct for — so
this needs a second convolution path (nominal grid + fitted
`dispersion_centers` shift, with the synthetic truth still generated from
the real GD polynomials as today), not just a new free parameter alongside
the gases. Build this last, as its own small design pass, not bundled in
with gases/albedo/pressure. Whether its own Jacobian can stay analytic
too (§7) is a separate, currently-unconfirmed question from this
convolution-path question — both need answering before dispersion is
done, but they're independent.

## 5. Goal 3 — multi-FPA joint retrievals

The biggest structural change of the three, but with a direct template
already in the codebase: `_joint_retrieve` (`gd_test.py`) already does
this for single rows — N `SpectralWindow`s, one concatenated `y_true`,
one shared gas-scale state (physically one atmosphere seen by every
band), per-band albedo/dispersion (`prior_albedo = np.array([b["albedo"]
for b in bands])`).

The joint block's generalization has the same shape, scaled up:
`build_forward` loops over FPAs, renders each one's own row-window via
its own `predict_neighborhood`/`_no_keystone` variant with its own
`wn_hires`/`ils`, and concatenates residuals — the same per-bin gas state
shared across every FPA (one physical atmosphere), dispersion now
**per-FPA** rather than a single window-wide value. Albedo compounds
further per §4's correction: since surface reflectance is itself
wavelength-dependent, albedo needs to be **both per-bin and per-FPA**
(one value per bin per band, not shared across bands the way the gas
state is) — the largest block in the whole state vector once multi-FPA
and the corrected albedo treatment land together.

The part needing real thought is windowing. Today's tiles are row-indexed
against one reference FPA (`build_window_tiles` uses `rows_crossed(fpa,
...)` for a single `fpa`). Physically, eta is the band-independent
coordinate — the same physical atmospheric column lands on a *different*
row on every FPA's own detector. So tiles should be defined **in
eta-space**, and each FPA's own row range for a given eta-window derived
independently via that FPA's own inverse mapping — mirroring how
`_shared_s_grid` already computes the N-way coverage intersection across
bands for the rectified pipeline, rather than inventing a new convention.
Bin placement and regularization stay FPA-agnostic once defined in
eta-space; only the forward model needs to become FPA-list-aware.

## 6. Suggested build order

1. **Migrate** (§1) — mechanical move into `geocarb_gert/joint_block.py`
   plus `gd_render.predict_neighborhood_no_keystone`, validated by
   re-running the row 890-935, G=15 case and diffing against recorded
   numbers.
2. **Noise** (§3) — cheapest, reuses the existing real noise model,
   unlocks the bias/variance characterization work that's been an open
   item since the ATBD.
3. **Gases / albedo / surface pressure** (§4, easy tier) — mechanical
   generalization of existing `StateVector` plumbing already present in
   every call the joint block makes today.
4. **Analytic Jacobians end-to-end** (§7) — restructures the solver from
   one opaque finite-difference loop to a real chain-rule composition.
   Build this right after the migration and before piling further goals
   on top, since noise doesn't touch it but gases/albedo/pressure/profile
   all benefit directly, and it's cheaper to get right once than to retrofit
   after several goals' worth of state-vector growth are already finite-
   differenced.
5. **Dispersion and multi-FPA** (§4 hard tier, §5) — the two pieces that
   need genuinely new architecture (a nominal-grid convolution path; an
   eta-space, FPA-list-aware forward model and window tiling), built last
   so they land on top of an already-solid state-vector/regularizer/
   Jacobian foundation rather than alongside a still-changing one.

## 7. Analytic Jacobians end-to-end — avoiding finite differences where we can help it

**Directive (2026-08-13, user)**: no finite differences anywhere GERT
already gives us the pieces to avoid them.

**Where finite differences are today.** `gauss_newton_regularized`
finite-differences through the *entire* forward chain as one opaque
`forward(x)` callable, once per free parameter per iteration: bin
attribution, `_diagonal_ils_convolve`, and `gaussian_blur_rows` all get
rerun from scratch inside every perturbed call, on top of re-running RT
for the one changed bin. None of that is necessary.

**What GERT already exposes.** Checked `gert/forward_model.py` directly
rather than assuming: `ForwardModel.run(..., jacobians=True)` returns
analytic sensitivities on the *same hi-res grid* `I_hires` already lives
on (`ForwardResult.I_hires` is documented as "before ILS convolution" —
exactly the stage `spectrum_for` already stops at, via `res.I_hires[0]`).
Concretely, on that hi-res grid:

- `K_mol_hires * tau_gas_hires[mol]` gives `dI_hires/d(mol_scale)` per gas
  — the piece needed for every per-bin gas block in §4.
- `K_albedo_hires` / `K_slope_hires` give `dI_hires/d(albedo)` and
  `dI_hires/d(albedo_slope)` directly.
- `dtau_mol_dpscale_lay_hires` combined with `K_ray_lay_hires` /
  `tau_ray_lay_hires` gives `dI_hires/d(p_scale)` (the same Rayleigh +
  molecular assembly `GERTRetrieval._jacobian` already does internally for
  its own channel-grid Jacobian, just read off one stage earlier).
- `tau_gas_layer_hires` supports exact per-level Jacobians via the
  documented "mid-point chain rule" — directly the piece §4.0a needed to
  avoid finite-differencing a profile block `G * N_levels` times.

**Important nuance**: this is not literally reusing
`GERTRetrieval._jacobian`/`_jacobian_mixed` as callable functions — those
are internal to `GERTRetrieval`, coupled to *its own* `Instrument`/
`SpectralWindow` channel-grid convolution and retrieval loop, not exposed
as standalone utilities. What's reusable is `ForwardModel.run(jacobians=
True)` itself (already what `spectrum_for` calls, just with one more
kwarg) plus the *assembly formulas* those internal methods use, documented
directly in `ForwardResult`'s own docstrings — applied one stage earlier,
at the hi-res pre-convolution point, so the joint block can compose its
*own* downstream operators on top instead of GERT's channel-grid ones.

**The chain-rule composition.** Apart from RT (genuinely nonlinear, now
covered analytically above), every other stage in the joint block's own
pipeline is *linear*: bin/anchor attribution (a fixed gather or
interpolation-weight matrix — pixel-to-bin assignment depends only on
geometry, never on the retrieved state `x`), `_diagonal_ils_convolve` (a
fixed per-pixel truncated-Gaussian weighted sum), `gaussian_blur_rows` (a
fixed Gaussian blur across rows). A linear operator's own Jacobian *is*
itself, so the full chain-rule Jacobian is just

```
K_total = [PSF blur] . [ILS convolve] . [attribution] . dI_hires/dx
```

with `dI_hires/dx` supplied analytically per the above, and no finite
differences anywhere in the composition. Practically, this doesn't even
need new "Jacobian versions" of the linear stages written: since
`_diagonal_ils_convolve` and `gaussian_blur_rows` are linear in their
input array, calling the *exact same functions* on a sensitivity array in
place of a spectrum/image array propagates the Jacobian correctly through
the same code path already used for the forward value itself.

**A real efficiency win, not just a correctness one.** Since attribution/
ILS/PSF geometry depends only on `(fpa, rows, bin_centers)`, never on
`x`, those three operators' weights can be computed *once per window* and
reused across every GN iteration — strictly cheaper than today, which
reruns the full convolution + blur pipeline from scratch inside every one
of the `G+1` finite-difference `forward()` calls, every iteration.

**Dispersion, flagged but not confirmed.** `gert/instrument.py` has its
own analytic-Jacobian building block for dispersion coefficients
(`∂R/∂a_k`, near `dispersion_centers`) — suggesting an analytic path
exists there too, but this hasn't been traced through as thoroughly as
the gas/albedo/pressure pieces above and shouldn't be asserted with the
same confidence until §4's dispersion work actually starts. Separately,
note this is orthogonal to dispersion's own architectural fork (§4, hard
tier: convolving onto a nominal grid instead of the real GD-polynomial
one) — that question is about *what* gets convolved onto, this section is
about *how* the Jacobian w.r.t. whatever's being convolved gets computed;
both still apply once dispersion is built.

**Net effect on the solver.** `gauss_newton_regularized` needs to change
shape: not a generic finite-difference loop around an opaque `forward(x)`
callable, but something that (a) asks `ForwardModel` for analytic hi-res
sensitivities per free parameter, and (b) applies the same precomputed
linear attribution/ILS/PSF operators used for the forward evaluation
itself. This is the single biggest change to the *solver* discussed in
this document — bigger than anything in §3-§5 — which is why it's placed
early in the build order (§6) rather than retrofitted after several
goals' worth of state-vector growth are already finite-differenced.
