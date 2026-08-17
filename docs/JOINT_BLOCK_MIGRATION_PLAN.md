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

**Made precise, not just asserted (2026-08-14).** `gauss_newton_
regularized`'s `Sa_inv = gamma*(L^T L) + I/sigma_abs**2` is exactly a
discrete Gauss-Markov (Ornstein-Uhlenbeck-type) prior *precision* matrix
— literally the `Sa^-1` of Rodgers optimal estimation, the same object
this whole section is about generalizing. It has two springs: `gamma`
couples each bin to its immediate *neighbors* only (a random-walk-style
smoothness term — on its own it's rank-deficient, since a uniform shift
of every bin costs nothing, so it constrains differences, never the
absolute level); `1/sigma_abs**2` couples each bin *individually* to the
prior `x_a` (no reference to neighbors at all). Inverting this matrix
(verified numerically, not just by the general AR(1)-precision identity
it follows) shows that, away from the array edges, the implied prior
covariance decays like `Corr(i,j) = exp(-|i-j|/xi)` — an exponential
kernel in *bin index*, exactly this section's own `exp(-|eta_i-eta_j|/
corr_length)` form, just with the correlation length expressed in bins
rather than eta. Both the decorrelation length `xi` and the marginal
prior standard deviation at any one bin are set by the *ratio*
`r = (1/sigma_abs**2) / gamma`, not by either constant in isolation --
worked numerically for the specific values this project has been using,
with the empirical consequences, in §9's own gamma/sigma_abs findings.

This is the concrete reason "one correlation length per quantity" (this
section's own proposal) is a real improvement, not just a tidier API: the
current two-constant parameterization only lets you set `xi` and the
marginal prior std *jointly* and non-obviously (recovering either one
requires inverting the matrix, as done here) -- a table entry giving
`sigma` and `corr_length` directly, in physical eta/km units, would make
both directly legible and independently tunable instead of an emergent,
coupled property of two numbers in unrelated units (a unitless smoothness
strength and a fractional scale-factor uncertainty).

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

## 8. A validation finding worth carrying into the migration: `_diagonal_ils_convolve` and observation-grid numerical fidelity

Found 2026-08-14 while building an independent, non-joint-block validation
of the keystone-free ceiling (`scripts/gd_keystone_free_1d_ceiling.py`):
that script generates its "truth" measurement with the joint block's own
renderer (`_diagonal_ils_convolve`, the ILS-convolution routine behind
`predict_neighborhood`/`predict_neighborhood_no_keystone`, §1's own
migration target), then retrieves against it using GERT's unmodified,
native `GERTRetrieval`/`ForwardModel` machinery — deliberately two
different code paths, so the check is genuinely independent. Even on a
trivial constant-atmosphere (`--uniform`) scene, where the correct answer
is exactly zero everywhere, a small but real and consistent CO2 bias
showed up.

**First cause, found and fixed.** `_diagonal_ils_convolve`'s own docstring
claimed "same truncated-Gaussian kernel and normalisation as
`ILS.convolve(..., exact_center=True)`" — reading the two implementations
side by side turned up two real differences: GERT's own window is one
hi-res point wider on each side (`ceil(half/spacing) + 1` vs.
`ceil(half/spacing)`), and GERT applies a second, exact `|delta| <= half`
cutoff after slicing that window, which `_diagonal_ils_convolve` skipped.
Built `scripts/gd_diagonal_ils_convolve_test.py` as a small, real-spectrum
unit test to check this directly rather than trust the docstring's claim,
confirmed a real (if tiny, ~3e-7 relative in absorption line depth)
mismatch, then fixed `_diagonal_ils_convolve` in `geocarb_gert/gd_render.py`
to match GERT's window and mask exactly. Re-running the test afterward
showed the two routines are now bit-identical (diff ~5e-15, floating-
point noise). This fix carries forward automatically into the migrated
module (§1) since `predict_neighborhood`/`predict_neighborhood_no_keystone`
already live in `gd_render.py`.

**Second cause, chased down but not a joint-block issue.** Fixing the
convolution closed almost the entire gap for a full-nuisance-parameter
retrieval (~200x smaller) but left a small residual (~0.002 ppm) in the
local-truth-nuisance-idealized case essentially untouched. Built
`scripts/gd_rt_reproducibility_test.py` to check the next candidate — the
raw hi-res RT spectrum itself, before any convolution — by running truth's
own RT call and the retrieval's own RT call for the identical atmosphere
and diffing them directly. They were *not* bit-identical (~4e-6 relative,
uniform across the whole spectrum — the signature of a flat scaling
difference, not a spectral-shape one). Isolated the exact cause by varying
one `SpectralWindow`/`Instrument` construction argument at a time: passing
`obs_grid` (an explicit observation-wavenumber array) reproduces the full
discrepancy on its own; removing it while swapping in the real noise model
instead gives an exact match. So `obs_grid` — not noise modeling, not
albedo, not solar-spectrum handling — perturbs `ForwardModel`'s computed
hi-res radiance by a small, structural, atmosphere-independent amount,
even though that radiance is documented as computed before ILS convolution
and shouldn't depend on the output grid at all.

Checked directly whether this touches the real joint-block pipeline: it
does not. `make_spectrum_fn`'s own `spectrum_for` (used throughout
`build_forward`/`build_forward_hires`/every real joint-block retrieval)
builds its `Instrument` via `band_basics`, the same construction truth-
generation itself uses — no `obs_grid` anywhere in that path. Diffed
directly against truth: bit-identical (`max|diff|=0.0`). It *does* touch
`gd_test._joint_retrieve` — the function behind every native/undistorted/
rectified number in this whole project — which builds its own retrieval
`SpectralWindow` with `obs_grid=nu`. That's pre-existing, established
methodology, not something this investigation introduced, and it's ~6
orders of magnitude smaller than the effects those pipelines actually
report (native mean −2.2 ppm vs. this artifact's ~1e-6 relative scale), so
it changes no conclusion already drawn from them — but it's now a
characterized, understood, reproducible detail rather than an unexplained
loose end, which matters given this work is headed for publication.

**Why this shouldn't get worse under more realistic scenes.** The
discrepancy tracks a construction choice (`obs_grid` present or not), not
atmospheric complexity — it was identical to six significant figures
across three different rows/wavelength regions of a uniform scene, and its
flat, wavelength-independent shape is inconsistent with a gas-absorption
or spectral-shape mechanism that could grow with sharper real features.
Not exhaustively stress-tested outside the configurations already run
here (this band, this row range, noise-free) — characterized and
structurally explained, not proven bounded for every future configuration.

## 9. A second validation finding: the plume-region anomaly is a regularization effect, not a keystone or rendering bug

Found 2026-08-14, directly following §8's own methodology (build a
self-consistent test with an exactly-known correct answer, not just an
"independent" one). §8's uniform-scene test proved the joint block's own
geometry/attribution/PSF machinery is exact when the true field has zero
curvature; it could not, by construction, say anything about the still-
open plume-region anomaly (`scripts/gd_joint_block_whole_slit_sweep.py`,
RMS 0.06-0.29 ppm, hi-res not meaningfully better than coarse there),
since that only appears with real curvature.

**The test**: `scripts/gd_ramp_self_consistency_test.py`. CO2 varies
exactly linearly in eta within a window (every other gas held fixed at
`atm_center`, isolating CO2-ramp recovery specifically), rendered through
the joint block's own self-consistent machinery (`predict_neighborhood`
for truth, `build_forward`/`build_forward_hires` for the retrieval's own
model — the identical code path, exactly as §8's uniform test used), at
five window centers spanning near-zero to near-maximum local keystone
(rows 25, 275, 525, 775, 1000; `rows_crossed` 0.005 to 10.2). Two sharp,
exactly-known expectations, not just "small residual expected": hi-res
(piecewise-linear interpolation) should recover a linear ramp almost
exactly, since a line is its own exact piecewise-linear interpolant;
coarse (nearest-bin) should show a *calculable* sawtooth bias, zero at
each bin center and growing linearly to the bin edges.

**Three real bugs found and fixed while building the test itself** (worth
recording, matching this project's own practice of not hiding the
debugging path):

1. *ppm-conversion mismatch*: converted the retrieved `co2_scale` back to
   ppm using the ramp's own per-bin true value instead of the constant
   `atm_center` CO2 value that `x` is actually a scale factor relative to
   — produced spurious ~15-18 ppm "bias" that was pure scoring error.
2. *PSF-padding lookup margin*: the ramp's own precomputed lookup table
   didn't cover the eta range of the PAD rows `predict_neighborhood`
   needs for correct PSF-blur edge handling (found to be ~10x wider than
   a naive 5%-of-window-span margin), silently clamping those rows to a
   boundary value instead of continuing the ramp.
3. *Unphysical extrapolation*: anchoring the ramp globally at eta=0 while
   scaling the local slope to hold a fixed total swing per window (see
   below) implied negative CO2 concentrations for windows far from eta=0
   — fixed by anchoring each window's own ramp at its own center.

None of these three were joint-block bugs — they were bugs in the new
test script itself, each caught by the same discipline used throughout
this investigation: check the actual numbers against a known-exact
expectation rather than trust the code.

**A fourth issue, this one a test-design flaw rather than a bug**: a
fixed *global* ppm-per-eta slope makes narrow, low-keystone windows see
an unrealistically tiny local swing (a 20 ppm/eta slope gives a 9-row
window only ~0.3 ppm of true local variation), easily swamped by
regularization tuned for real several-ppm features. Fixed by scaling the
local slope per window to hold a constant *local* swing (10 ppm, `--swing-ppm`)
across each window's own eta span regardless of width, keeping the
signal-to-regularization ratio comparable across very different keystone
levels — the actual goal of the sweep.

**Result, once the test itself was correct**: both coarse and hi-res
showed a real, non-zero bias — but *flat across the entire keystone
range tested* (~0.9-1.2 ppm RMS at every one of the five windows, from
`rows_crossed` 0.005 to 10.2). Visually, both retrieved profiles show a
systematic S-curve: too high relative to the true ramp at the low-eta end
of each window, too low at the high-eta end — flatter than the true
ramp everywhere, the textbook signature of a regularization prior pulling
a fit toward smoothness/its own prior value at the expense of the full
signal swing. Since the bias doesn't track `rows_crossed` at all, keystone
amplitude is not the driver of this component of the joint block's error.

**Which regularization term, tested directly rather than assumed**:
`gauss_newton_regularized`'s `Sa_inv = gamma*(L^T L) + I/sigma_abs**2` has
two terms — a smoothness prior (`gamma`, linking neighboring bins) and a
loose absolute prior (`sigma_abs`, pulling every bin toward its own prior
of `co2_scale=1`). Tested each in isolation, one variable at a time:

- `gamma`: 3.0 -> 0.3 (10x weaker smoothness prior) changed the bias by
  under 2% at every window — falsifies the smoothness-prior hypothesis
  directly, not just fails to confirm it. (Two resulting plots differing
  by only ~1.9% look visually near-identical at a glance; verified by
  hash, by the underlying printed numbers, and by a pixel-level diff — 1.5%
  of pixels differ, with real contrast where they do — that this was a
  genuine independent rerun, not a duplicate/caching artifact.)
- `sigma_abs`: 0.10 -> 1.0 (10x looser absolute prior) collapsed hi-res
  bias by 13-46x at every window (down to ~0.02-0.08 ppm, right at
  GN-convergence-tolerance noise — hi-res now recovers the ramp almost
  exactly, matching the original theoretical expectation), and reduced
  coarse bias by 30-80% wherever enough bins exist for shrinkage rather
  than genuine quantization to have been the dominant error (the
  smallest, 3-bin window at the keystone-null row barely moved, since
  real quantization already dominates there regardless of the prior).

**Confirmed mechanism, not a hypothesis**: `sigma_abs=0.10` — the value
used throughout the production whole-slit sweep — is measurably too tight
for a feature with a ~10 ppm swing, producing a real, keystone-independent
shrinkage bias. This is a property of the regularization, not a bug in
geometry, attribution, rendering, or the GN solver's own convergence.

**Why gamma didn't matter but sigma_abs did, quantified via §4.0's own
decorrelation-length analysis.** `Sa_inv`'s implied prior covariance is
(away from edges) an exponential-correlation Gauss-Markov prior in bin
index, `Corr(i,j) = exp(-|i-j|/xi)`, with both the decorrelation length
`xi` and the marginal prior std set by the ratio `r = (1/sigma_abs**2) /
gamma`. Computed directly (n=15, a typical window G) for the exact values
tested above:

| gamma | sigma_abs | r | marginal std | corr(lag 1) | xi [bins] |
|---|---|---|---|---|---|
| 3.0 | 0.10 | 33.3 | 0.097 | 0.028 | 0.28 |
| 0.3 | 0.10 | 333 | 0.100 | 0.003 | 0.18 |
| 30 | 0.10 | 3.3 | 0.082 | 0.195 | 0.61 |
| 3.0 | 1.0 | 0.33 | 0.527 | 0.566 | 1.80 |

Across the entire tested `gamma` range (0.3 to 30, `sigma_abs` fixed at
0.10), `xi` never leaves sub-bin territory (0.18-0.61 bins) -- the
absolute-prior spring's own precision (`kappa=1/sigma_abs**2=100`)
dominates the diagonal so completely that neighboring bins stay nearly
independent regardless of `gamma`'s own value, which is exactly why a
100x change in `gamma` moved the bias by under 2%: the smoothness spring
was present but never strong enough, relative to the absolute spring, to
matter. Moving `sigma_abs` to 1.0 instead flips the ratio (`r`: 33 -> 0.33)
and does two things at once -- `xi` grows to ~1.8 bins (neighbors
genuinely coupled for the first time) and the marginal std loosens to
0.527 (looser, but less than a naive "10x" would suggest, since the two
springs reinforce each other rather than acting independently -- exactly
the joint, non-separable behavior that motivated treating `sigma` and
`corr_length` as the *right* independent knobs in the first place). The
loosened marginal std is the direct mechanism behind the S-curve: bins
far from a window's own center need `co2_scale` well away from 1 to
represent the true local ramp value, and a tight absolute prior resists
that on every bin symmetrically, worst at the edges.

**A second, compounding effect at the window edges**: the ramp profiles
also show growing error right at the ends of each window's eta range,
even at `sigma_abs=1.0`. This is a separate, structural effect from the
S-curve above, not a restatement of it. Every bin's row in `L^T L`
(the first-difference smoothness term) gets one `+gamma` contribution
per *adjacent* bin it's coupled to -- interior bins have two neighbors
(one on each side), but the two edge bins of any window have only one.
This is the free/natural boundary condition on a first-difference
operator, not a bug: nothing in the regularizer tells an edge bin what
lies just outside the window, so it gets less "spring" pulling it
toward its neighbor's value and is correspondingly freer to be pulled
by, or resist, the data term alone.

Checked directly by computing the diagonal of `Sa_inv` and the marginal
std (`sqrt(diag(Sa))`, `Sa = Sa_inv^-1`) at every bin position, not just
the center, for n=15:

  - `gamma=3.0, sigma_abs=0.10`: `Sa_inv` diagonal is 103 at the two edge
    bins vs. 106 for interior bins, giving marginal std 0.0986 (edge) vs.
    0.0972 (interior) -- an edge/center ratio of only 1.014 (1.4%).
  - `gamma=3.0, sigma_abs=1.0`: `Sa_inv` diagonal drops to 4 (edge) vs. 7
    (interior), giving marginal std 0.6590 (edge) vs. 0.5267 (center) --
    an edge/center ratio of 1.251 (25%).

The edge effect exists at both settings (it's structural, not caused by
loosening `sigma_abs`), but it only becomes visually/numerically
prominent once `sigma_abs` is loosened. The absolute-prior term
`I/sigma_abs**2` adds identically to every bin's diagonal regardless of
position, so when it's the dominant term (tight `sigma_abs`) it swamps
the 1-vs-2-neighbor asymmetry and every bin looks nearly equally
constrained. Loosening it removes that masking floor rather than fixing
the underlying weakness -- the boundary condition was always there.

This directly confirms the mechanism behind the "check whether
excursions align with window boundaries" item from the original
five-point debugging plan: production windows tile the whole slit
non-overlapping, so every window's edge bins sit exactly at
inter-window boundaries. It also means the upcoming real-plume
`sigma_abs` test (below) needs to be read with this in mind -- whatever
`sigma_abs` value helps shrink the S-curve bias will simultaneously make
this edge weakness more visible at every window boundary, not just in
the controlled ramp test.

**Why this looked like a strong candidate for (part of) the plume
anomaly**: the real plume feature this whole investigation has been
chasing has almost exactly this scale — baseline ~413 ppm to peak ~423
ppm, a ~10 ppm swing, the same order as this test's own `--swing-ppm
10.0` default. `sigma_abs` was never varied in any of the production
sweeps or the original single-hot-spot validation (which used a smaller,
~4 ppm feature, where this same shrinkage would be proportionally
smaller). This motivated testing it directly rather than assuming the
mechanism transfers — see below.

**Tested directly on the real production windows — the hypothesis does
NOT transfer, and the direction of the effect is opposite what shrinkage
would predict.** `scripts/gd_plume_sigma_abs_sweep.py` solves three real
windows from the production whole-slit sweep (`results/gd_joint_block_
whole_slit_fpa2.pkl`, the `gamma=3.0`/`sigma_abs=0.10` baseline already
on disk) directly against the real rendered detector image, across a
100x range of `sigma_abs` (0.05 to 10.0, gamma fixed at 3.0) and a 100x
range of `gamma` (0.3 to 30, sigma_abs fixed at 0.10) — the same single-
variable-at-a-time isolation methodology as the ramp test above, but on
real data instead of the synthetic ramp:

  - rows 331-345 (G=5): the broad plume's own peak, x_km [-510,-472].
  - rows 884-924 (G=14): the original single-hot-spot ATBD test window
    (row 890-935), CO2 hot spot at x_km=+1050.
  - rows 108-116 (G=3): the worst hi-res bias anywhere in the whole-slit
    sweep (-0.655 ppm at row 110), CO2 hot spot at x_km=-1100.

| window | sigma_abs=0.05 | 0.10 (baseline) | 0.30 | 1.0 | 3.0 | 10.0 |
|---|---|---|---|---|---|---|
| 331-345 hi-res rms | 0.036 | 0.047 | 0.059 | 0.062 | 0.063 | 0.063 |
| 884-924 hi-res rms | 0.127 | 0.147 | 0.160 | 0.162 | 0.162 | 0.162 |
| 108-116 hi-res rms | 0.399 | 0.427 | 0.442 | 0.444 | 0.444 | 0.444 |

Every window's bias *increases* slightly as `sigma_abs` loosens — the
opposite sign from the ramp test's 13-46x collapse — and saturates
almost immediately past `sigma_abs=0.3` instead of continuing to shrink.
The `gamma` sweep (0.3 to 30 at `sigma_abs=0.10`) reproduces the ramp
test's null result (bias changes by a few percent at most, no consistent
direction), consistent throughout this investigation.

The decisive diagnostic is `max|x_hires-1|`, which was the actual
mechanism in the ramp test (bins pulled back toward the shared prior).
Here it stays inside 0.0002-0.0006 across the *entire* sigma_abs sweep,
including at `sigma_abs=10` — a prior 100x looser than production, which
in the ramp test's own decorrelation-length table (Sec.4.0) corresponds
to a marginal std order-of-magnitude larger than any deviation needed
here. The GN solve simply never wants `x` away from 1 by more than a few
parts in `1e4`, at any prior tightness tested. `resid_hires_rms` (the
actual spectral fit quality) tells the same story: it changes by
single-digit percent across the whole sweep (e.g. window 108-116:
6.57e-4 at sigma_abs=0.10 down to 6.28e-4 at sigma_abs=10 — barely
moving), nothing like the order-of-magnitude drop the ramp test showed
when sigma_abs was loosened. Both signs point the same way: unlike the
ramp test, `x` is not being resisted by the prior here — it is already
close to the least-squares optimum, and loosening the prior has almost
nothing left to give it.

**Why the mechanism is different here, confirmed by the profile plots**
(`plots/joint_block/gd_plume_sigma_abs_sweep_profiles_fpa2.png`): the
ramp test used a single SHARED `atm_center` prior for every bin in a
window, so recovering a ramp genuinely required `x` to move away from 1
bin-by-bin, and a tight `sigma_abs` measurably resisted that. The
production sweep's priors are different by design — `prior_atms[g] =
als.atmosphere_at(x_km_bin_g)`, i.e. already the true LOCAL atmosphere at
each bin's own center (the same "local-truth nuisance-gas idealization"
the whole-slit sweep's own docstring describes). A correct retrieval
therefore does not need `x` to leave 1 to match truth AT the bin
centers — only the piecewise-linear INTERPOLATION between bin centers
needs to represent whatever curvature the true field has in between. The
profile plots show exactly that failure mode at both settings of
`sigma_abs`: the retrieved curve sits almost exactly on the chord
connecting adjacent true-value bin centers, visibly cutting inside every
concave stretch of the true curve (most obviously at the row 884-924 hot
spot's own peak, where the true profile rises to 417.0 ppm but the
piecewise-linear interpolant tops out at 416.5, regardless of
`sigma_abs`). This is a resolution/representation ceiling set by bin
density and placement relative to the feature's own curvature scale, not
a regularization effect — `gamma`/`sigma_abs` were never in a position
to fix it, because the retrieval was never resisting the correct answer
in the first place.

**Net conclusion, stated plainly for the write-up**: the sigma_abs
shrinkage mechanism found in the controlled ramp test is real and
precisely characterized (Sec.4.0, above), but it does NOT explain the
production plume/hot-spot bias — that hypothesis is falsified by direct
test, not merely unconfirmed. The ramp test's shared-prior setup, while
useful for isolating the regularization mechanism cleanly, is not
representative of how the production sweep actually parameterizes a
window (position-dependent local-truth priors), and the two therefore
probe different failure modes. The production bias's real driver is most
consistent with a bin-density/interpolation resolution ceiling.

**Confirmed directly: bin count G, not regularization, is the real
driver.** `scripts/gd_plume_bin_density_sweep.py` holds `gamma`/
`sigma_abs` fixed at the production baseline (3.0, 0.10) and sweeps `G`
instead, on the identical three windows, from below the production
default up to roughly one bin per detector row:

| window (production G) | G=2 | production G | ~2x | ~one bin/row |
|---|---|---|---|---|
| 331-345 (G=5) | 0.362 | 0.047 | 0.031 (G=8) | 0.017 (G=30) |
| 884-924 (G=14) | 1.516 | 0.147 | 0.067 (G=28) | 0.044 (G=41) |
| 108-116 (G=3) | 1.861 | 0.427 | 0.169 (G=5) | 0.059 (G=18) |

Bias falls monotonically and substantially with `G` at every window —
9-11x from `G=2` to the finest `G` tested, and a further 3-10x just from
the *production* `G` to the finest `G` tested (108-116: 0.427 -> 0.059,
7.2x; 884-924: 0.147 -> 0.044, 3.3x; 331-345: 0.047 -> 0.017, 2.8x) —
with no sign of flattening out by the largest `G` tried. This is the
opposite behavior from the sigma_abs sweep (flat-to-slightly-worse, no
matter how far the prior was loosened) and lines up exactly with the
resolution-ceiling explanation: `resid_hires_rms` (spectral fit quality)
stays essentially flat across the *entire* `G` range at every window
(e.g. 331-345: 3.184e-3 to 3.186e-3 across `G=2` to `G=30`, a 0.06%
range) even as the ppm bias drops by an order of magnitude — the fit to
the actual measured radiance was never the limiting factor; only the
number of bins available to represent the true field's own curvature
was. The profile plots
(`plots/joint_block/gd_plume_bin_density_sweep_profiles_fpa2.png`) show
this directly: the finest-`G` curve at each window sits almost exactly
on the true profile, including through the row 884-924 hot spot's own
peak (417.0 ppm) that the production-`G` piecewise-linear interpolant
could only reach 416.5 ppm on.

This closes the loop opened by the ramp test's own speculative
"strong candidate" framing: the mechanism that explains the controlled
ramp's bias (sigma_abs shrinkage) is real but does not explain the
production anomaly; the mechanism that does explain the production
anomaly (bin-density resolution ceiling) was identified only by testing
directly against the real windows rather than assuming the ramp test's
finding would transfer. For the write-up, the honest framing is: the
plume/hot-spot bias in the current production whole-slit sweep is
primarily a resolution artifact of the adaptive window/bin-count formula
(`G = round(width / G_RATIO)`, tied to local keystone amplitude via
window width, not to the true field's own curvature scale), not a
regularization or geometry/rendering bug. The practical implication is
that `G_RATIO` (or an adaptive scheme that also accounts for feature
curvature, not just window width) is the actual lever for reducing this
bias in production, not `gamma`/`sigma_abs`.

**Caveat, checked directly rather than assumed: at high G, bins are not
independent samples — a real, second ceiling exists underneath the
first one.** A window of `W` rows contains `W x 1024` real pixels, not
`W` — bin placement quantiles the full per-pixel eta distribution (see
`gd_plume_bin_density_sweep.py`'s own docstring), which is why `G` was
able to exceed the row count at all in the sweep above. But that raises
the obvious follow-up: with `G` approaching or exceeding the row count,
are bins resolving genuinely new information, or mostly re-hitting the
same physical rows through different column slices? Checked directly,
not assumed, with `scripts/gd_plume_bin_placement_diagnostic.py` (pure
geometry, no RT — bin centers and nearest-bin pixel *counts*, the same
assignment `gd_joint_block_diagnostics.py`'s own per-bin chi2 diagnostic
uses, at the finest `G` tested in each window above):

| window | width | finest G | mean rows spanned per bin | bins confined to 1 row |
|---|---|---|---|---|
| 331-345 | 15 | 30 | 3.57 | 2/30 |
| 884-924 | 41 | 41 | 9.41 | 0/41 |
| 108-116 | 9 | 18 | 1.44 | **10/18** |

No bin was empty or exactly duplicated another's pixel set at any window
(`plots/joint_block/gd_plume_bin_placement_fpa2.png`, min pixel count per
bin 123-302, none zero) — so every bin genuinely carries some new
information, confirming the `G` sweep's result wasn't an artifact of
degenerate bins. But at 108-116 (the lowest-keystone window of the
three), over half the finest-tested-G bins live entirely within a single
detector row, meaning most of that window's G increase was subdividing
one row's own narrow keystone-driven column spread rather than resolving
new physical along-slit positions. The bin-count ceiling is therefore not
uniform across the slit: at high-keystone windows (884-924, mean 9.4 rows
spanned per bin even at G=41) there is real headroom for still more bins
before hitting this second ceiling; at low-keystone windows like 108-116,
the ceiling is much closer, set by how much column spread keystone
actually produces there (i.e. by `rows_crossed` at that location), not
by an arbitrary resolution choice. The same asymmetry is visible in the
`gd_plume_bin_density_sweep.py` results above on reflection: 108-116
needed a full 6x its own row count in `G` (production 3 -> tested 18) to
get the bias down to 0.059 ppm, while 884-924 needed only ~3x (production
14 -> tested 41) to reach a comparable order of improvement — consistent
with 108-116 running out of genuinely independent column-spread
information sooner. Practically, this means any fix along the `G_RATIO`
line (previous paragraph) needs to account for keystone amplitude at the
bin-placement stage too, not just scale up `G` uniformly — a fixed
multiple of window width will over-promise resolution at low-keystone
locations and under-deliver it at high-keystone ones.

**Correction, checked directly: the analysis above describes the COARSE
posterior's bin partition, not the hi-res forward model's real pixel
resolution — worth recording as a real correction, not silently fixed,
matching this project's own practice.** The "column ownership"/"10 of 18
bins confined to a single row" analysis above used
`gd_joint_block_diagnostics.pixel_density_bin_centers`/`bin_assign`
applied directly to the retrieval's `G`-dimensional state bins. That
scheme is exactly right for the COARSE posterior (`build_forward` in
`gd_joint_block_retrieve.py` really does use it, via
`nearest_bin_scene(bin_centers, ...)`, to assign every pixel to one of
the `G` bins). But `build_forward_hires`
(`gd_joint_block_hires_test.py:63`) — the forward model behind every
"hi-res" bias number quoted in this whole document — does NOT assign
pixels to the `G` state bins at all. It first interpolates the `G`-dim
state (in ppm/co2_scale space) onto `G_eff` ROW-LEVEL anchors, one per
detector row in the padded window (`anchor_etas = _eta_of(fpa,
col=512, anchor_rows)` — a single eta per row, at column 512 only), where
`G_eff = width + 2*PAD` is fixed by the window's own row count and
completely independent of `G`. Only THEN does `nearest_bin_scene` assign
each real pixel to its nearest anchor. So `G`'s only role in the hi-res
model is controlling how much freedom the interpolation onto those fixed
`G_eff` anchors has — it never touches pixel assignment directly, and
the earlier bin-level "sharing" analysis, while an accurate description
of the coarse-posterior partition, was answering the wrong question for
the hi-res numbers.

Checked directly (pure geometry, no RT) what the REAL hi-res anchor
assignment looks like, at `G_eff` for each of the three windows
(`PAD=4`, so `G_eff = width + 8`):

| window | width | G_eff | mean % of a row's own pixels -> its own anchor | distinct anchors one row's pixels touch | `rows_crossed` at center |
|---|---|---|---|---|---|
| 108-116 | 9 | 17 | 97.2% | 2 (±1 row) | 0.96 |
| 331-345 | 15 | 23 | 33.2% | 5 (±2 rows) | 3.41 |
| 884-924 | 41 | 49 | 10.9% | 10 (±4/+5 rows) | 9.23 |

This is the opposite ranking from the earlier (wrong-scheme) analysis,
and it makes complete physical sense once framed correctly: the number
of distinct anchors a single row's own pixels spread across scales
directly with `rows_crossed` at that row — exactly the quantity that
already defines window/bin sizing everywhere else in this investigation.
108-116 (the window that looked "worst" for coarse-bin sharing) actually
has the LEAST real anchor-sharing, because its keystone amplitude is
near the row-25 null; 884-924 has the MOST, because its keystone is the
highest of the three windows tested. This is by design, not a flaw — it
is literally the mechanism `build_forward_hires`'s own docstring
describes ("resolving within-row keystone splicing at native row
granularity"): a row with large keystone genuinely straddles several
neighboring rows' along-slit positions, and letting its own pixels draw
on those neighbors' independently-computed RT anchors is more accurate
than forcing every pixel in a row to use that one row's own single
center-column RT call.

**Directly answering the question that prompted this: no, no pixel is
ever assigned to two anchors (or two bins) at once, in either scheme.**
Both `bin_assign` (coarse) and `nearest_bin_scene` (hi-res, at `G_eff`
resolution) are hard nearest-neighbor partitions — `np.searchsorted`
against sorted bin/anchor edges gives exactly one index per pixel by
construction. What varies between windows is not whether pixels get
double-counted (they never do) but how many of a ROW's own pixels end up
under a NEIGHBORING row's anchor instead of its own — real information
sharing between physically adjacent rows, appropriately scaled to local
keystone, never a duplicate assignment of the same pixel.

**What this means for the `G` sweep's own result and the practical
recommendation.** Since `G` only controls interpolation freedom onto a
FIXED `G_eff`-anchor grid for the hi-res model, the correct target for
`G` is `G_eff` itself (`width + 2*PAD`), not an arbitrary multiple of
window width: once `G >= G_eff`, the interpolation can give (almost)
every anchor its own independent `co2_scale`, and no further hi-res
resolution is available from `G` alone, because pixel-to-RT-call
assignment is capped at `G_eff` regardless of how large `G` gets. Cross-
checked against the `G`-sweep bias numbers (`gd_plume_bin_density_
sweep.py`) by computing the relative bias drop at each step:

| window | G_eff | relative hi-res bias drop at each successive G tested |
|---|---|---|
| 108-116 | 17 | 77% (2->3), 61% (3->5), 50% (5->9), **29% (9->18, straddles G_eff)** |
| 331-345 | 23 | 87% (2->5), 34% (5->8), 25% (8->15), 28% (15->30, past G_eff) |
| 884-924 | 49 | 85% (2->7), 36% (7->14), 54% (14->28), 35% (28->41, still short of G_eff) |

108-116 shows the cleanest signal — a clear, monotonic deceleration that
lands right where `G` crosses its own `G_eff=17`. 331-345 is broadly
consistent (big early gains, smaller later ones) but doesn't show a sharp
knee at `G_eff=23` — only 5 `G` values were tested, and real per-window
noise (exactly where quantile bin/anchor edges happen to fall relative to
the feature's own shape) is clearly still a factor at this sampling
density. 884-924 never reached its own `G_eff=49` in the sweep (tested
only to `G=41`), and correspondingly shows no deceleration at all — if
anything the 14->28 step improved more than 7->14, consistent with still
being well short of the ceiling. The overall picture is consistent with
the `G_eff` mechanism without being a clean, sharply-resolved proof of it
at every window — worth a finer `G` sweep bracketing each window's own
`G_eff` directly (particularly re-running 884-924 out to `G=49`) before
treating "set `G = G_eff`" as a settled production recommendation rather
than a well-motivated one.

**One picture tying bins, anchors, rows, columns, and eta together**,
for the 108-116 window at `G=18` (`scripts/gd_plume_bins_vs_anchors_
diagram.py`, `plots/joint_block/gd_plume_bins_vs_anchors_fpa2_r108-
116_G18.png`): per-pixel eta across the full padded row range with the
`G=18` state-bin centers (orange lines, clustered tightly since they
only quantile the 9 core rows) and the `G_eff=17` anchors (black stars,
one per row, spanning the full padded range including the rows added
just for PSF-edge handling) plotted together, alongside the coarse
bin-partition and hi-res anchor-partition pixel-ownership maps side by
side over the same 9 rows x 1024 columns. The contrast is immediate: the
coarse map carves each row into 2-3 unevenly-sized, differently-colored
segments that jump around row to row; the hi-res/anchor map is almost
entirely one solid color per row, with only a thin sliver bleeding into
each neighbor — the same near-97%-own-row result from the table above,
now visible directly rather than just tabulated.

**Should `G` ever exceed `G_eff`? Checked exactly, not just by analogy
to the diminishing-returns curve above — for the hi-res model, no, and
it is a harder boundary than "no further benefit."** `co2_anchors =
np.interp(anchor_etas_sorted, bin_centers, x)` in `build_forward_hires`
is, for fixed anchor/bin positions, an EXACTLY linear map `co2_anchors =
M @ x` with `M` a fixed `(G_eff, G)` weight matrix — and every downstream
step (RT, pixel assignment, ILS convolution, spatial PSF blur) depends on
`x` only through this product. So whenever `G > G_eff`, `M` is a wide
matrix with a nontrivial null space, and any perturbation `dx` in that
null space satisfies `M @ (x + dx) = M @ x` EXACTLY — not approximately,
not "small effect on the fit" — meaning `forward_hires(x + dx)` is
bit-identical to `forward_hires(x)` for any such `dx`. Verified directly
by constructing `M` and checking its rank (not merely asserted from the
dimension-counting argument, since it undersells the effect):

| window | G | G_eff | rank(M) | null-space dimension |
|---|---|---|---|---|
| 331-345 | 30 | 23 | 19 | **11** |
| 108-116 | 18 | 17 | 11 | **7** |
| 884-924 | 41 | 49 | 41 | 0 (G still < G_eff — full column rank) |

A random nonzero `dx` drawn from the null space changed `M @ x` by
2e-16 (floating-point noise) when checked numerically — confirming the
claim exactly, not just in principle. The null space is bigger than the
naive `G - G_eff` count in both cases tested (11, not 7, at 331-345; 7,
not 1, at 108-116) — bin/anchor interleaving (multiple bin centers
falling between the same two anchors, or several anchors clamping to the
same boundary bin) creates additional rank deficiency beyond simply
having more columns than rows. 884-924, the one window whose `G` sweep
never crossed its own `G_eff`, correspondingly has `M` at full column
rank — no null space — consistent with why it was the one window still
improving strongly at the largest `G` tested, no plateau in sight.

The practical reading: once `G > G_eff`, those extra state dimensions
are invisible to the Gauss-Newton data term specifically (`K^T Sy^-1 K`
is exactly singular along them too, since `K` inherits the same null
space via the chain rule) — the regularizer (mainly the `sigma_abs`
floor, which is generically full-rank) is the ONLY thing keeping the
solve well-posed in those directions, so the solve still converges to a
unique answer, but its fine structure below `G_eff` resolution is purely
prior-determined, not measured. That is a stronger and more precise
statement than "no further benefit was observed in the sweep" — it is an
exact statement about what the data can and cannot constrain, provable
from the forward model's own linear-algebra structure without needing to
run RT at all.

Two caveats against over-generalizing this into "always cap `G` at
`G_eff`" project-wide:
- This null space is specific to the HI-RES model's `np.interp`-onto-
  fixed-anchors construction. The COARSE model has no analogous ceiling
  — `bin_assign` determines pixel assignment directly from the `G` bins
  themselves, so its real ceiling is the per-pixel column-density limit
  from the earlier caveat (which can exceed `G_eff` at high-keystone
  windows like 884-924, where mean rows-per-bin was still ~9.4 even at
  `G=41`).
- `G_eff` itself is a fairly blunt, keystone-INDEPENDENT quantity — one
  anchor per row, at column 512 only, regardless of local keystone
  amplitude — so "cap `G` at `G_eff`" really means "cap `G` at whatever
  this particular anchor-placement convention happens to deliver," not
  necessarily the true informational ceiling the pixels could support.
  A keystone-aware anchor scheme (more than one anchor per row where
  keystone is large, mirroring how `pixel_density_bin_centers` already
  handles the coarse model's own bin placement) would raise `G_eff`
  itself at high-keystone windows, and the sensible `G` ceiling would
  rise with it. Whether that extra hi-res resolution would actually be
  used well by the current padding/PSF-blur machinery, and how much
  spectral RT cost it would add, is untested — flagged as a natural
  follow-on, not yet investigated.

**What information a bigger `G_eff` would actually add, and exactly how
keystone gates it — checked directly, not asserted.** The one-anchor-
per-row scheme evaluates every anchor's eta at column 512 only, then
stamps that single RT result across every pixel of the row assigned to
it (before ILS convolution). But a row's own 1024 columns don't all sit
at that one eta — keystone spreads them across a real range. Measured
directly at each window's center row:

| window (center row) | within-row eta span (1024 px) | row-to-row anchor eta step | ratio | `rows_crossed()` |
|---|---|---|---|---|
| 108-116 (row 112) | 0.00189 | 0.00192 | 0.99 | 0.96 |
| 331-345 (row 338) | 0.00664 | 0.00193 | 3.45 | 3.40 |
| 884-924 (row 904) | 0.01797 | 0.00191 | 9.39 | 9.23 |

The ratio and `rows_crossed` agree to within measurement noise — not a
coincidence, since both are measuring the same thing: `rows_crossed` IS
this ratio, already computed and already used everywhere else in this
investigation for window/bin sizing. This gives it a second, very
concrete meaning here: it is exactly how many anchor-spacings' worth of
along-slit position a single row's own columns already sample, that a
single row-level anchor discards. At row 904, the row's own columns
physically look at ~9 rows' worth of along-slit position, but the
current scheme extracts one RT sample from all of it. If the true field
has real structure at that scale (a plume edge, a hot spot — exactly
what this whole investigation has been probing), that structure is
present in the measured pixels and currently unavailable to the
retrieval at anything finer than row resolution, recoverable only
indirectly through the crude row-to-row spillover characterized above,
not through a genuinely independent RT sample at the column's own eta.
At zero keystone this gap doesn't exist (ratio -> 1, every column of a
row truly does sit at the same eta, and a single anchor is already
exactly correct) — the potential gain from a finer-than-row anchor grid
is `rows_crossed`-proportional, essentially by definition. This is also
exactly why 884-924 was the one window whose `G` sweep never plateaued
and whose `M` matrix stayed full column rank even at `G=41` (above): its
real informational ceiling, set by `rows_crossed~9`, sits far above what
a row-count-based `G_eff=49` can express with one anchor per row.

**Towards a keystone-aware anchor selection strategy.** The natural fix
follows directly: place anchors as quantiles of the REAL per-pixel eta
distribution over the padded window — literally reusing
`pixel_density_bin_centers` unchanged, just applied to the padded row
range's full `(row, col)` pixel population (every column of every padded
row) instead of one column-512 sample per row. This is precisely the
mechanism that already lets the COARSE model's `bin_assign` support far
more bins than `G_eff` at high-keystone windows (884-924 was still at
mean 9.4 rows/bin even at `G=41`, i.e. still not exhausted) — the coarse
model already gets this for free because `bin_assign` was always
pixel-density-aware; only the hi-res model's anchor scheme was left at
a flat, keystone-blind one-per-row convention.

Concretely, this would replace `build_forward_hires`'s `anchor_etas =
_eta_of(fpa, col=512, anchor_rows)` with `anchor_etas =
pixel_density_bin_centers(eta_flat_over_padded_window, G_eff)`, letting
anchor DENSITY track real keystone-driven pixel density automatically
rather than fixing anchor COUNT to row count a priori. Two consequences
worth flagging, both untested:

- **A natural place to also resolve the `G > G_eff` null space.** Since
  the null space traced directly to `G` state bins being interpolated
  onto a MISMATCHED, coarser `G_eff`-anchor grid, using the SAME
  pixel-density placement for both — and simply setting `G = G_eff`
  directly — would collapse the two-tier "state bins interpolated onto
  separate anchors" design into one tier: `G` parameters, each with its
  own independently-computed RT, placed once by real pixel density. No
  interpolation matrix `M`, no rank deficiency to create in the first
  place, and the "coarse vs. hi-res" comparison that has run through
  this whole document would need rethinking, since the coarse posterior
  would then already sit at what is currently called hi-res resolution.
  This is a real simplification, not just a resolution fix, but it
  changes what "coarse" means in every plot in this document — worth
  treating as a deliberate design decision, not a drop-in patch.
- **Self-scaling, not free, compute cost.** RT cost for the hi-res model
  is presently `~G_eff` calls per window (`~one per row`); a pixel-
  density-weighted scheme would spend more of a fixed total anchor
  budget exactly where `rows_crossed` is large and less where it's
  small — a strictly better allocation of a given budget than uniform
  one-per-row, but if the total anchor count is allowed to grow with
  `rows_crossed` rather than staying budget-capped, RT cost at high-
  keystone windows (already the slowest windows in every sweep run in
  this document, e.g. 884-924's ~99-140s per solve vs. 108-116's ~10-15s)
  would grow further. Choosing the total anchor count is still an
  empirical question, not one this ratio table answers by itself — the
  same sweep-and-check-`M`'s-rank methodology already established above
  is the natural way to tune it, now against a properly keystone-shaped
  anchor grid instead of a row-count one.

Not implemented or tested — a well-motivated proposal grounded directly
in the `rows_crossed`/null-space findings above, not a change made in
this investigation. The natural next step, if pursued, is a small
standalone unit test analogous to `gd_diagonal_ils_convolve_test.py`:
build the pixel-density anchor scheme for one window, confirm `M` (now
square by construction, `G=G_eff`) is full rank, and compare bias
against the current row-uniform scheme at matched RT cost (same total
anchor count) before touching the production sweep.

**What "minimum resolvable spatial signal" precisely means, checked
directly before running that test.** The detector can only capture
along-slit structure as finely as the eta STEP between neighboring
pixels resolves it (before any PSF smearing) — and that step must be
measured locally, in both directions a pixel grid has: `d_col` (eta step
between adjacent COLUMNS of the same row) and `d_row` (eta step between
adjacent ROWS at fixed column). Measured directly at each window's own
padded rows:

| window | `d_col` (typical) | `d_row` (row-to-row) | `d_col` range across ONE row's own band (wavelength-dependence) |
|---|---|---|---|
| 108-116 | ~1.5e-6 | ~0.00192 | 8.4x (row 112: min at ~2.067 um, max at band edge ~2.086 um) |
| 331-345 | ~6e-6 | ~0.00193 | 2.0x (row 338) |
| 884-924 | ~1.8e-5 | ~0.00191 | 1.5x (row 904) |

`d_row` is essentially constant everywhere (varies <1% across every row
tested) — it's set by the physical row pitch, independent of keystone.
`d_col` is the one that moves, by orders of magnitude with keystone
(correlates with `rows_crossed` at r=+1.000 within each window — a row's
total eta span is `d_col x 1024`, so more keystone stretching that span
over the same 1024 columns mechanically means a larger average step) AND
by up to 8x with wavelength alone, within a single row. This sharpens
Sec.9's earlier point about the spatial PSF only blurring rows, never
columns: `d_row` being fixed means the row-to-row axis is already close
to uniformly resolved regardless of keystone; essentially all of the
keystone/wavelength-dependent structure lives in `d_col`, i.e. in exactly
the direction the current row-uniform anchor scheme cannot see at all
(one eta sample per row, always at column 512).

This also sharpens what pixel-DENSITY placement is and isn't measuring.
Density (pixels per unit eta) is high wherever `d_col` is small — which
is LOW keystone, where a row's total eta span is narrow to begin with,
not high keystone, where the real long-range structure actually lives.
So `pixel_density_bin_centers` quantiling the padded population doesn't
straightforwardly target "where keystone creates real resolvable
structure" — it targets "where many pixels sit close together in eta,"
which is a related but distinct thing. Worth having flagged before
running the test below, not just after.

**The decoupled test, done properly.** The first attempt at this test
(not written up in detail here — it conflated two changes: swapping
anchor placement AND collapsing the two-tier `G`/`G_eff` architecture
into one, `G_eff := G`, at once, which independently throws away the
implicit smoothing the old scheme's interpolation-onto-a-larger-fixed-
grid provides. That confound, not anchor placement, was almost certainly
what made the first attempt look uniformly worse.) The corrected version
(`scripts/gd_plume_anchor_density_sweep_v2.py`) keeps the OLD two-tier
architecture for BOTH schemes — `G` state values interpolated onto a
FIXED `G_eff = width + 2*PAD` anchor grid, exactly `build_forward_hires`'s
own structure — changing ONLY how those `G_eff` anchors are placed (row-
uniform vs. pixel-density over the padded population, both computed once
per window, not per `G`). `G` itself is swept from 5 up through `G_eff+1`
per window, to watch bias cross the exact null space proven above (via
the same interpolation-matrix rank check) for both schemes side by side.

Full results:

| window | G | G_eff | old bias rms | new bias rms | rank(M) old | rank(M) new | nullity |
|---|---|---|---|---|---|---|---|
| 108-116 | 5 | 17 | 0.1687 | 0.1569 | 5 | 5 | 0 |
| 108-116 | 8 | 17 | 0.1129 | 0.1082 | 8 | 8 | 0 |
| 108-116 | 14 | 17 | 0.0707 | 0.0665 | 11 | 11 | 3 |
| 108-116 | 17 | 17 | 0.0662 | 0.0633 | 11 | 11 | 6 |
| 108-116 | 18 | 17 | 0.0593 | 0.0551 | 11 | 11 | 7 |
| 331-345 | 5 | 23 | 0.0471 | 0.0538 | 5 | 5 | 0 |
| 331-345 | 12 | 23 | 0.0257 | 0.0334 | 12 | 12 | 0 |
| 331-345 | 18 | 23 | 0.0219 | 0.0281 | 17 | 17 | 1 |
| 331-345 | 23 | 23 | 0.0212 | 0.0261 | 18 | 17 | 6 |
| 331-345 | 24 | 23 | 0.0205 | 0.0251 | 19 | 17 | 7 |
| 884-924 | 5 | 49 | 0.9528 | 0.9528 | 5 | 5 | 0 |
| 884-924 | 24 | 49 | 0.0762 | 0.0762 | 24 | 24 | 0 |
| 884-924 | 39 | 49 | 0.0519 | 0.0519 | 39 | 39 | 0 |
| 884-924 | 49 | 49 | 0.0420 | 0.0421 | 45 | 43 | 6 |
| 884-924 | 50 | 49 | 0.0413 | 0.0413 | 45 | 43 | 7 |

**Result: mixed, not a clear win — and mixed in a way that's partly, but
not fully, explained.**

- 108-116 (lowest keystone, `rows_crossed` 0.88-1.05 within the padded
  window): NEW beats OLD consistently, ~4-8% at every `G` tested,
  including past `G_eff`.
- 331-345 (mid keystone, `rows_crossed` 3.29-3.52): NEW is consistently
  WORSE than OLD, ~15-30%, at every `G` tested.
- 884-924 (highest keystone, `rows_crossed` 8.99-9.47): OLD and NEW are
  indistinguishable — agree to 3+ significant figures at every `G`.

The high-keystone wash follows directly from the `d_col`/`d_row` table
above: at 884-924, within-row spread (`d_col x 1024 ~ 0.018`) dominates
the padded window's total eta range roughly 10x over the row-to-row
spacing (`d_row ~ 0.0019`) — so quantile placement over the padded pixel
population, being dominated by the within-row contribution either way,
ends up nearly identical to row-uniform placement regardless. The low-
keystone improvement is consistent too: there, within-row spread
(`~0.0015`) is actually SMALLER than the row-to-row step, so both
schemes are already effectively "one anchor per row-cluster" — density
placement just gets to fine-tune exactly where within/between those
clusters, for a small, consistent win. 331-345's consistent WORSENING,
in the intermediate regime (within-row spread ~3.4x the row spacing),
does not have a clean explanation from this same story — flagged
explicitly as unresolved, not force-fit to a tidy narrative.

**A partial lead on the 331-345 anomaly, found by directly plotting
where each scheme's anchors actually land**
(`scripts/gd_plume_anchor_placement_diagram.py`,
`plots/joint_block/gd_plume_anchor_placement_fpa2.png` — per-pixel eta
scatter with both anchor schemes overlaid, plus a consecutive-anchor-
spacing panel, for all three windows). NEW's anchor-to-anchor spacing
is NOT uniform the way OLD's is by construction — and the two higher-
keystone windows share a distinct pattern: spacing sits close to OLD's
through almost the entire interior, then widens sharply at the very
edges:

| window | interior `Δη` (NEW, typical) | OLD `Δη` (constant) | edge `Δη` (NEW, first/last gap) | edge widening factor |
|---|---|---|---|---|
| 331-345 | ~0.0020 | ~0.0019 | ~0.0044 / ~0.0041 | ~2.2x |
| 884-924 | ~0.0020 | ~0.0019 | ~0.0079 / ~0.0071 | ~4x |
| 108-116 | ~0.00196 (a shallow U, densest at center) | ~0.00192 | ~0.00223 / ~0.00207 | ~1.15x |

331-345 and 884-924 share the same qualitative shape (OLD-like through
the interior, a sharp outlier at each edge) — but only 331-345 got
worse; 884-924 was a wash. The difference is plausibly just where the
true feature sits relative to that sparser edge coverage (884-924's own
hot spot peak sits well inside the window, away from the thinned edges;
331-345's broad plume peak is closer to one edge of its own window —
not confirmed here, but consistent with both this spacing pattern and
the earlier profile plots). 108-116 has a visibly different spacing
*shape* (a shallow U, not a sharp edge spike), consistent with it being
the one window where NEW's interior spacing is actually tighter than
OLD's rather than matching it. This narrows the open question from "no
explanation at all" to "likely an edge-coverage effect specific to where
each window's own feature sits," but does not fully resolve it — the
feature-position hypothesis is not independently verified here.

One more observation, checked rather than assumed: despite the null
space being mathematically exact (proven above via the interpolation
matrix's rank), none of the three bias-vs-`G` curves show a sharp kink
right at `G_eff` — bias keeps improving smoothly through it in every
panel. Most likely explanation: the regularizer's own smooth
continuation of the now data-blind extra parameters still happens to
track a smooth true field reasonably well in these particular scenes, so
the data ceasing to constrain those directions doesn't necessarily
produce a visible discontinuity in this bias metric, even though the
underlying rank-deficiency is exact and unconditional.

**Net conclusion**: properly isolated from the tier-collapse confound,
keystone-aware (pixel-density) anchor placement is a real but small,
non-uniform effect on these three windows — helps modestly at low
keystone, hurts at mid keystone, does nothing at high keystone. Not
grounds for adopting it in production as tested. The mechanism proposed
above (place anchors where keystone/wavelength genuinely creates
resolvable structure) is directionally motivated and the `d_col`/`d_row`
analysis explains two of the three outcomes cleanly, but raw pixel
density is evidently not the same thing as "where keystone creates real
resolvable structure," and the mid-keystone result shows the current
implementation isn't simply capturing it either. A cleaner placement
criterion — weighting quantiles by local `d_col` directly (or
equivalently by keystone amplitude) rather than by raw pixel count —
is the natural next refinement, not yet built or tested.

## 10. Whole-slit production validation: G tied to the ground-pixel footprint

Everything in Sec.9 characterized the `G`-resolution mechanism on three
representative windows. This closes the loop with a real, full-slit,
realistic-scene production run — the actual deliverable, not another
diagnostic.

**Design** (unchanged from Sec.9's own conclusions, deliberately not
re-litigated here): anchors stay row-uniform (`build_forward_hires`,
`G_eff = width + 2*PAD`) — the pixel-density anchor scheme's mixed Sec.9
results didn't clear the bar for adoption. `G` is tied to the along-slit
ground-pixel footprint (2.7344 km/row = 2 x `SLIT_HALF_KM` / 1024) rather
than the previous arbitrary `G_RATIO=3` convention: `G = width` (one bin
per row's own native resolution), via a new `--g-ratio 1.0` flag on
`gd_joint_block_whole_slit_sweep.py` (default unchanged at `G_RATIO=3`,
fully backward compatible). Checked before running, not assumed: `G_eff`
(the hard ceiling proven in Sec.9) is fixed at `width+2*PAD` regardless
of `G`, and summing over all 58 windows in the existing adaptive tiling,
`G=width` sums to 1024 against `G_eff` summing to 1488 — `G/G_eff` is
0.53-0.85 at every window, so this choice never touches the null space
anywhere on the slit.

**Execution**: parallelized across the SLURM `atmos` partition as one
array task per window (58 tasks total — the simplest possible split,
array index maps directly to a window, no load-balancing logic needed).
Each task independently rebuilds the realistic-scene band from scratch
and writes its own output file; no shared state or cross-task I/O during
the run (`scripts/submit_whole_slit_1x.sbatch`), merged afterward with a
small standalone script (`gd_joint_block_whole_slit_merge.py`) that
sanity-checks all parts share the same run parameters and no window was
solved twice before combining them into the same pickle format the
original single-process script produces — `gd_joint_block_whole_slit_
plot.py` needed only a path argument added, no other changes. Measured
task walltimes (not estimated): narrowest window (width=9) 9s, a mid
window (width=15) 35s, the widest window on the slit (width=45) 106s,
all including the per-task band-rebuild overhead — the array's
`--time=00:30:00` budget has ample margin.

**Result**, scored identically (same `gd_joint_block_whole_slit_plot.py`)
against the existing `G=width/3` production baseline
(`results/gd_joint_block_whole_slit_fpa2.pkl`) for a direct, apples-to-
apples comparison:

| | coarse rms | coarse max\|bias\| | hi-res rms | hi-res max\|bias\| |
|---|---|---|---|---|
| `G=width/3` (old production default) | 0.2616 ppm | 1.4320 ppm | 0.0602 ppm | 0.6553 ppm |
| `G=width` (ground-footprint, this run) | 0.0409 ppm | 0.4076 ppm | 0.0166 ppm | 0.1285 ppm |
| **improvement** | **6.4x** | **3.5x** | **3.6x** | **5.1x** |

A dramatic, whole-slit confirmation of the Sec.9 resolution-ceiling
finding — not a modest refinement. The worst hi-res bias location also
moved, from row 110 (the 108-116 west hot spot this whole investigation
spent the most time on) at the old default to row 911 (the 884-924 east
hot spot) at the new one — consistent with Sec.9's own `G`-sweep numbers,
where 108-116 improved fastest with `G` (already near its own resolution
ceiling at modest `G`) while 884-924 kept improving all the way out to
the largest `G` tested there without plateauing.

**Not yet done, left for a deliberate follow-up rather than folded in
here**: the `G_eff` sensitivity point (does pushing `G` all the way to
the proven ceiling on real production windows help further, or replicate
the exact-null-space plateau from Sec.9's synthetic tests) was scoped to
run on a small, stratified subset of windows rather than the full 58, to
keep compute bounded — not yet executed.
