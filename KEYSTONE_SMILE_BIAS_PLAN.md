# Keystone & Line-Curvature Bias Study — Plan

**Goal.** Quantify the bias in **XCO₂, XCH₄, and XCO** *along the slit* induced by the
GeoCarb focal-plane distortions — **keystone** and **line curvature (smile)** — and
examine the accompanying **systematic spectral residuals**. Built on the
`geocarb_gert` focal-plane model + the `gert` retrieval library. Approached in phases
of increasing scene complexity, non-scattering first to save computation.

(SIF is deferred — it needs a new fluorescence forward-model term + state element in
`gert`; it will be added as a later target. Aerosol scattering enters only in Phase 4.)

---

## 1. Real GeoCarb detector geometry

The focal-plane model is scale-free in physical length: `slit_length_mm`,
`magnification`, and `detector_pitch_um` enter **only** through
`n_spatial = slit_length_mm · magnification / pitch_mm`. Distortions are all in pixels.

| quantity | value |
|---|---|
| slit → detector | 3000 km ground → **1024 spatial px** and **1024 spectral px** |
| ground sampling (GSD) | 3000 / 1024 ≈ **2.93 km/pixel** (set by the telescope fore-optics, *upstream* of this model) |
| keystone | slit **image** grows **20 px** blue→red = **1.95 %** (`keystone_px=20`); modeled symmetric about slit centre → each end grows 10 px. **Magnitude confirmed by ground-test GD characterization** (up to ~10 rows crossed at the slit ends, §9) — but the real curve is **not symmetric about centre**; each FPA's zero-keystone "sweet row" is offset by clocking, worst for FPA2 (see §9) |
| N/S PSF | **1.5 px** FWHM (`spatial_psf_fwhm_px=1.5`) ≈ 4.4 km ground blur. **Confirmed** — ground test reports the same ~1.5 px FWHM (§9) |
| smile | "pronounced", parabolic opening toward long-wave. **Confirmed from the real GD polynomials** (`geocarb_gert.gd_polynomials`, §9f): **~33–39 px** at each band's centre wavelength — a **15–20× correction** to the `SMILE_PX=2.0` placeholder. Not even constant across a band: e.g. FPA0 ~37 px at the short-wavelength edge vs. ~48 px at the long-wavelength edge |

Physical translations at 2.93 km/px: keystone shifts a sounding's footprint **~29 km**
between the blue and red band ends at the slit ends (band-to-band co-registration
error); an edge crosses up to **~10 detector rows** across the band at the slit ends,
~0 at centre (`rows crossed = η_edge · keystone_px/2`).

A physical triple for `n_spatial=1024` (any equivalent works): e.g. 18 µm pitch /
mag 1 → 18.4 mm slit, or the notebook's 50 µm / 51.2 mm.

---

## 2. Key physics established (and verified)

### 2a. Keystone alone is null on a uniform slit; smile×keystone coupling is not
With a scene **identical along the slit**, pure keystone (`smile=0`) produces **exactly
zero** change (verified: `max|Δ| = 0`). It only remaps spatially-uniform content.

**But smile is a slit-position aberration**, and keystone remaps slit→detector-row
*wavelength-dependently* (`row = η_slit·stretch(λ)`). So at a fixed detector pixel the
effective smile is that of `η_slit = row/stretch(λ)` — keystone **modulates** smile by
`1/stretch(λ)²`. The combination differs from smile alone, growing as ~η²:

| slit position | `(smile+keystone) − (smile alone)`, uniform scene |
|---|---|
| η = 0.3 | 2.8 % of band signal |
| η = 0.6 | 9.7 % |
| η = 0.9 | **25.3 %** |

`FocalPlaneModel` already captures this (smile is evaluated at `η_obj = η/stretch`), so
**no new distortion code is needed**. This coupling is a first-class effect of the
study, not a footnote.

### 2b. On a uniform scene the whole distortion is a pure per-row dispersion
Each pixel samples `L` at `ν_eff(u) = ν_j − smile·dnu·η²·f/(1+kf·f)²`, with
`f` the in-band fraction and `u = 2f−1` the dispersion coordinate. Expanded:

```
δν ≈ −smile·dnu·η² · (f − 2·kf·f² + …)
       └ smile: linear in f  → dispersion order 1
                └ smile×keystone coupling: f² → dispersion order 2
```

So the uniform-scene distortion is exactly a **per-row dispersion polynomial**:
order-1 = smile, order-2 = the coupling. The leftover beyond order-2 is O(kf²) ≈
0.008 px — negligible.

---

## 3. Methodology

**Bias = retrieve a distorted truth with a nominal model.** For each slit position η:
1. Generate the **truth** radiance *with* the focal-plane distortion (smile shifts each
   row's dispersion; keystone mixes along-slit scene content once the scene varies).
2. Retrieve with the **nominal** `gert` forward model, non-scattering
   (`solver_kind='ss'`, Lambertian), **with dispersion retrieved**.
3. Bias(η) = x_retrieved − x_true for XCO₂/XCH₄/XCO; **systematic residual**(η) =
   y_obs − y_model(x_ret).

### Retrieve dispersion — order 2, loose prior
The retrieval **includes** a per-window dispersion polynomial
(`include_dispersion=True`, `dispersion_order=2`) with a **loose prior**
(`dispersion_uncert` large) so it moves freely to minimise bias. Rationale: order-1
absorbs smile but leaves the order-2 coupling (the ~25 % residual); order-2 absorbs
both. **DOF loss is acceptable; posterior bias is what we minimise.**

**Sub-study — bias vs dispersion order (1, 2, 3) per phase**, watching for
**over-fitting** at high order, where the dispersion polynomial becomes degenerate with
the gas Jacobian and can re-introduce bias/variance. The bias-vs-order curve finds the
sweet spot.

### Nonlinear-primary; linear only where validated
The problem is **not** assumed linear — the 25 % coupling is already outside a
small-perturbation regime, and dispersion-in-state + row-crossing + scattering make it
worse. So:
- **Two bias engines:** full nonlinear `GERTRetrieval`, and a linear estimate
  `Δx = G·Δy` (gain `G = (KᵀSy⁻¹K + Sa⁻¹)⁻¹KᵀSy⁻¹`, `K` computed once as in
  `run_design_sweep`; systematic residual `r = (I − KG)·Δy`).
- **Validate first:** at a few η (0, ±0.5, ±0.9) compute bias **both** ways and compare
  per gas, across dispersion order 1/2/3, for smile-only and smile+keystone.
- **Production runs use nonlinear.** It is cheap in Phases 1–3 (single-scatter), so a
  dense η sweep of full retrievals is affordable. Linear is a fast diagnostic, used for
  dense sampling only where the comparison shows it holds (likely small |η|, smile-only).
- Phases with **row-crossing (2–3)** and **scattering (4)** are nonlinear throughout.

---

## 4. Phases

Split along "is the distortion a pure dispersion?":

**Phase 1 — uniform scene (smile + coupling; keystone-null check).**
Standard atmosphere, uniform albedo. Run three configs and difference them: smile-only,
smile+keystone, keystone-only (null check). Because the distortion is a pure per-row
dispersion here, order-2 retrieval should drive residual gas bias → **~0**. So Phase 1
**calibrates the dispersion order needed** (scan order → watch bias collapse at order 2)
and validates the machinery. Deliverables: bias(η) for each gas per config per order;
systematic-residual spectra vs η.

**Phase 2 — albedo varies along slit (keystone turns on for real).**
Same atmosphere; along-slit albedo structure (edges/gradients, per band). Keystone now
blends *different continuum levels* across the band footprints — **not** a wavelength
remapping, so dispersion retrieval **cannot** fully absorb it. The surviving bias is the
**irreducible keystone-heterogeneity bias**. Deliverables: bias(η) decomposed
smile/keystone/combined; residuals showing the keystone continuum signature vs the smile
derivative signature; residual bias vs dispersion order (should *not* collapse).

**Phase 3 — trace-gas amounts vary along slit (line-depth variation).**
Add along-slit variation in CO₂/CH₄/CO/H₂O so absorption depth varies. Keystone now
mixes *different line depths* across bands → gas-specific biases and cross-band
cross-talk. Deliverables: bias(η) per gas with realistic gas gradients; XCO₂ sensitivity
to XCH₄/H₂O cross-talk under keystone.

**Phase 4 — add aerosol scattering from an atmospheric simulation.**
Turn on scattering (`ms`/`xrtm`), aerosol profiles sampled from the atmospheric model
along the slit. Deliverables: the full realistic bias(η) budget with scattering, and how
aerosol-induced along-slit radiance structure amplifies keystone leakage. Expensive —
compute `K` once per aerosol state where the linear diagnostic is used.

---

## 5. Infrastructure to build

1. **GERT-grid truth generator** (`geocarb_gert`): produce per-sounding (per detector
   row) distorted truth on the **instrument's channel grid with the instrument's ILS**,
   for all bands.
   - *Smile (all phases):* use `gert`'s native `ForwardModel.run(dispersion={band:
     coeffs})` — per row, smile+coupling is an order-2 dispersion, so this is exact and
     grid-consistent; no 2-D render needed for uniform scenes.
   - *Keystone with scene variation (Phase 2+):* a multi-band render sampling the
     along-slit scene `S(η, ν)` at `η_obj = η/stretch(λ)`, convolved with the instrument
     ILS — the `FocalPlaneModel` logic generalised to 4 bands + the gert grid.
2. **Bias driver**: loop η → build distorted `Observation` → retrieve (nominal FM,
   `ss`, dispersion order-2 loose prior) → extract XCO₂/XCH₄/XCO bias (scale → column
   via `column_xgas`) + post-fit residual. Runs both engines for the validation step.
3. **Along-slit scene builder**: `albedo(η)`, `gas_scale(η)`, `aerosol(η)` and the
   per-node hi-res radiances to blend (reuse `edge_scene`/`random_scene`).
4. **(Later) SIF forward-model term + `sif` state element** in `gert`.
5. ~~Discrete row-crossing truth generator~~ — **built**, see §9h:
   `geocarb_gert.gd_render.image()` renders the full focal-plane image from the real
   GD polynomials at native 1024×1024 resolution, non-separable (every pixel at its
   own true wavelength/slit-position, no cross-row borrowing except the real N/S PSF
   blur). Uncovered a new mechanism in the process — irreducible PSF×smile coupling
   bias, §9h — not just the row-averaging effect originally anticipated.

---

## 6. Deliverables (consistent per phase)

- `bias(η)` curves: XCO₂ [ppm], XCH₄ [ppb], XCO [ppb], split smile-only / keystone-only
  / combined, and per dispersion order (1/2/3).
- `residual(η)` heatmaps (channel × η) per band — the systematic fingerprint.
- Summary table: peak bias (η = ±1), mid-slit bias, fraction of soundings exceeding
  mission bias thresholds, and the dispersion order that minimises residual bias.
- `notebooks/keystone_smile_bias.ipynb` + a driver in `geocarb_gert`/`scripts`.

---

## 7. Open decisions

1. ~~Per-band smile amplitude~~ — **resolved**, see §9f: ~33–39 px at band centre from
   the real GD polynomials, a 15–20× correction to `SMILE_PX=2.0`. Keystone magnitude
   (`keystone_px=20`) confirmed by ground test (§9b). Remaining implementation work:
   the truth generator still needs the per-band **offset** (clocking, §9c) and the
   **wavelength-dependence of the smile amplitude itself** (§9f), not just a bigger
   constant.
2. **η sampling**: dense nonlinear sweep (e.g. 32–64 η) + linear-vs-nonlinear validation
   at ~5 η; revisit after the Phase-1 comparison.
3. **Dispersion order to carry into Phases 2–4** (from the Phase-1/order sub-study).
4. ~~Row-crossing non-convergence mechanism~~ — **resolved**, see §9b: confirmed as
   pixel-grid aliasing during spectrum extraction, not a software indexing issue
   (independently documented in `keystone_report.pdf` §4.8.2).
5. **Smile-vs-wavelength shape** (§9f): confirmed the amplitude grows toward band
   edges and ruled out clocking as the cause, but haven't yet checked whether it's a
   pure amplitude scaling or an actual curve-shape change across the band — would
   affect how faithfully a single per-band dispersion polynomial (vs. a
   wavelength-dependent one) can represent it.

---

## 8. First concrete step

Build the GERT-grid smile truth generator + the retrieval driver **with dispersion
retrieval (order-2, loose prior)**, then for **Phase 1** run the **5-η
linear-vs-nonlinear** comparison at two configs (smile-only, smile+keystone) **across
dispersion order 1/2/3** — expecting order-2 to zero the uniform-scene bias, and
confirming where the linear estimate diverges near the slit ends.

---

## 9. Real-data check: confirmed mechanisms (2026-07-21 → 2026-07-23)

**Motivation.** The Phase-1 uniform-scene results (`compute_keystone_bias.ipynb`) are
much more optimistic than retrievals against real GeoCarb upward-looking ground-test
data, where the slit was filled with uniform light — nominally a Phase-1-like
scenario. Several hypotheses were tested against this gap; ground-test documentation
(`keystone_report.pdf`, GeoCarb geometric-distortion characterization) has now
**confirmed** the leading one and ruled out another.

### 9a. Tested — along-slit radiometric non-uniformity (small effect, real but minor)
Residual slit-thickness brightness variation, diffuser non-uniformity, and
radiometric cal artifacts all leave a small `albedo(η)` structure behind even a
"uniform" scene. Unlike a spatially-constant brightness offset (degenerate with the
retrieval's own per-band albedo state, absorbed trivially), a *gradient* is seen
differently by different wavelengths within a band because keystone samples a true
slit position `η_obj(f) = η / stretch(f)` that varies across the band — this is the
actual mechanism behind the Phase-2 "irreducible keystone-heterogeneity bias" (§4),
and it cannot be absorbed by dispersion retrieval regardless of prior looseness.
Built via gert's native `SurfaceBasis`/`surface_basis=` mechanism (exact
per-wavenumber albedo correction, not an approximation):
`ρ(ν) = albedo[b]·(1 + amp·η_obj(ν))`. Result: even a generous 5% peak-to-peak
residual gives only ~−0.25 ppm CO₂ bias at η=0.9, scaling linearly down to
~−0.005 ppm at 0.1%. **Real, correctly-scaling, but an order of magnitude too small
to be the primary explanation.**

### 9b. Confirmed — row-crossing / pixel-grid aliasing (the leading mechanism)
`keystone_report.pdf` §4.8.2 independently derives the same mechanism, in almost the
same words: *"This presents an image processing challenge to rectify spectra when
image contrast features are **aliased by the pixel grid**."* Ground-test GD
characterization (Figs. 36–38, 41) makes it concrete and quantified:

- Near each FPA's minimal-keystone slit position, extracting one spectrum means
  averaging **~2 detector rows** (e.g. rows 595–596 for FPA0 near +18% slit).
- Near the slit ends, it means averaging **up to ~7–10 rows** (e.g. rows 925–931,
  7 rows, for FPA0 at +85% slit) — quoting the report directly: *"averaging rows
  925 through 931 would add contributions from nearby slit positions, increasing
  the effective ground footprint size."*
- This happens **inside ordinary spectrum extraction**, not as a separate
  scene-variation test — it is the literal, row-averaging version of the Phase-2
  keystone-heterogeneity mechanism, present every time keystone is large enough
  that a "one row = one sounding" extraction has to blend several true rows.
- Peak row-crossing across all four FPAs is **~10 rows** at the slit ends (Fig. 38),
  which **matches the plan's `keystone_px=20` placeholder magnitude** (§1) — the
  amplitude assumption was reasonable; what was missing was modeling the row-mixing
  consequence at all.

Nothing built in this repo so far reproduces this: every truth generator to date
(`smile_disp_coeffs`, and the `SurfaceBasis` gradient in §9a) constructs the per-row
distortion as a single **smooth** polynomial fit for one nominal row — by
construction it has no discontinuities and no row-mixing, so a polynomial dispersion
retrieval always fits it cleanly. The bias/convergence results in
`compute_keystone_bias.ipynb` are for a regime the ground-test data shows is not
representative of the real hardware once |η| is more than modest.

### 9c. Confirmed — clocking is a per-band offset of the keystone curve, not a separate effect
Ground-test Figs. 38/41 show keystone (rows crossed) vs. slit position is **not**
symmetric about slit centre the way `smile_disp_coeffs`'s `η²` term assumes. Each
FPA's zero-keystone "sweet row" is offset by small rotational misalignment
(clocking) between the FPA and grating axes:

| FPA (band) | sweet row (of ~1024) |
|---|---|
| FPA0 (O₂ A-band) | 600 |
| FPA1 (weak CO₂) | 570 |
| FPA2 (strong CO₂) | **25** (nearly outside the slit) |
| FPA3 (CH₄/CO) | 266 |

FPA2 is the extreme case — its keystone-null point is almost at the slit edge, so
**nearly the entire FPA2 slit range has significant keystone**, unlike the other
three bands where it's modest near centre. This is exactly the band from the
residual-optimization images in this conversation. Model implication: the truth
generator's keystone term needs a **per-band offset**, not just a per-band
amplitude — `η²` centred on 0 is wrong for FPA2 in particular.

### 9d. Confirmed — FPA2 has an additional, independent calibration-coverage gap
Separately from clocking/row-crossing, ground test could not tune the calibration
laser across FPA2's full wavelength range, leaving part of the image with **no laser
calibration control points**. The report: round-trip geometric self-consistency
error is ~0.025–0.03 px for FPA0/1/3 but **~0.079 px mean / ~0.30 px peak for FPA2**,
concentrated exactly in the uncalibrated region, and — critically — *"there did
remain detectable wavelength errors even after the polynomial coefficients were
refined [via the EM27/SUN method]... as evidenced by error terms obtained in
retrievals over that part of the spectrum."* This is a strong, independently
documented explanation for the FPA2 (strong CO₂) residual-optimization images shown
in this conversation — likely compounding with, not replacing, the row-crossing
mechanism (9b) and the clocking offset (9c) for that specific band.

### 9e. Ruled out — geometric self-consistency / round-trip error
The report explicitly calls the baseline round-trip error (a few hundredths of a
pixel in dispersion, a few ten-thousandths along-slit) "easily absorbed in L2
retrieval processing." Not the driver, except in FPA2's uncalibrated region (9d).

### Calibration chain (for reference, `keystone_report.pdf` §4.8.1–4.8.6)
1. **Laser-spot GD test**: 55 control points per FPA (11 wavelengths × 5 slit
   positions), 2D Gaussian centroid fit to sub-pixel precision.
2. **4th-degree 2D polynomial** (not bicubic — 15 coefficients, `(x,y)→(λ,s)` and 7
   other transforms) fit to the 55 points per FPA by pseudo-inverse least squares.
   Degree was tuned (N=3,4,5 tested) — N=4 gave the best round-trip-residual balance;
   higher degrees over-fit and diverge between control points.
3. **EM27/SUN refinement**: an independent sun-viewing FTIR spectrum builds a
   synthetic "ideal" rectified spectrogram (solar lines vertical, slit-width
   artifacts horizontal); the 30 polynomial coefficients are then refined by
   Nelder-Mead minimization of the squared residual between synthetic and measured
   spectrograms — this is the "before/after" comparison in the FPA2 images
   discussed earlier in this conversation.

### 9f. Confirmed — real smile amplitude (~33–39 px), and why it varies across the band
`geocarb_gert.gd_polynomials` (added 2026-07-22) wraps the actual EM27/SUN-refined GD
coefficients (`gcmap_em27.csv`, from `keystone_report.pdf` §4.8.1/4.8.6) for both
directions: `xy_to_wavelength_slit` (rectification, A/B) and `wavelength_slit_to_xy`
(projection, C/D). Coordinate conventions (x,y in pixels; wavelength in microns;
`s` = slit angle in degrees, ~[-2, 2]) were validated by reproducing the report's own
numbers from the C/D-derived keystone curve: sweet rows 599.7/569.8/266.2 for
FPA0/1/3 (report: 600/570/266, all <1 px off) and FPA2's minimum landing at the edge
of the tested slit range, matching "falls slightly outside the slit."

Using the C/D (projection) polynomials to trace a fixed-wavelength line's x-position
across the full slit gives the real smile amplitude for the first time:
**~33–39 px at each band's centre wavelength** — a 15–20× correction to the
`SMILE_PX=2.0` placeholder used in every `compute_keystone_bias.ipynb` run to date.
It is also **not constant across a band**: FPA0 is ~37 px at the short-wavelength
edge vs. ~48 px at the long-wavelength edge.

**Is the within-band variation a clocking effect? No.** Clocking (§9c) is a rigid
*rotation* of the FPA axes relative to the grating's dispersion axis — a single angle
that is essentially wavelength-independent. It explains the keystone curve's offset
zero-point (a shift of the whole curve), not a change in curvature *magnitude*
between one part of a band and another. Smile amplitude growing toward the band
edges instead points to **field- and wavelength-dependent optical aberration in the
spectrometer's imaging path**: the grating diffracts different wavelengths at
different angles, sending them through different parts of the downstream camera
optics, and residual aberrations (coma, astigmatism, field curvature) there are
rarely uniform across the full design bandpass — commonly best-corrected near band
centre and worse toward the edges. This is a design/manufacturing-tolerance property
of the optics, not an alignment (clocking) effect. Not yet checked: whether this is a
pure amplitude scaling of the same curve shape, or an actual shape change across the
band (open decision §7 item 5) — would matter for how well a single per-band
dispersion polynomial can represent it vs. needing a wavelength-dependent one.

### 9g. What the residual-optimization result implies about model fidelity
The EM27/SUN refinement (9d, §4.8.6) reduces FPA2's summed residual 4933.65 → 2128.96
— a 57% reduction, not to zero — and this separates two different kinds of error
rather than pointing to one:

- **Under-sampling (fixable):** the initial laser-spot fit uses only 55 discrete
  control points per FPA; between them "the fit is uncontrolled" (§4.8.5). Going to
  an effectively ~1M-point reference (every pixel, via the EM27/SUN spectrum) closes
  most of the gap — this part of the error is about calibration density, not the
  model.
- **Structural (not fixable by more data):** the residual that *survives*
  refinement is not measurement noise — Method A's self-consistency check
  (Table 6) shows the laser centroids themselves are precise to ~1/20 px, so the
  input data is clean. What remains is a genuine mismatch between the 4th-degree
  global polynomial's functional form and the true optical mapping. The degree
  study (§4.8.5: N=3 underfits, N=5 "excursions dominate" between control points,
  N=4 is only the best *balance*) is an explicit admission that **no single global
  polynomial degree exactly reproduces the true distortion** — some real structure
  is finer-grained than any reasonable global polynomial can follow without
  overfitting elsewhere. The persistent post-refinement pattern (line-correlated
  striping across nearly the whole slit, not just the edges) is consistent with the
  same fine-scale, sub-polynomial structure the row-crossing/aliasing mechanism
  (9b) already implicates — an independent line of evidence for the same
  conclusion.

**Implication for the simulator:** `gd_polynomials.py` is a strong, validated
description of the *smooth, large-scale* component of the real distortion — good
enough to replace the placeholders with real per-band amplitude, offset, and shape
(9b/9c/9f). But the ground-test data's own residual says this smooth description is
not the complete truth. A truth generator built purely on these coefficients should
be read as a **lower bound** on real bias/residual, not an exact match — real
hardware likely shows somewhat more than a smooth-polynomial-based truth predicts.
FPA2's unusually large residual is a mix of both categories above: a known, fixable
data gap (insufficient laser wavelength coverage, 9d) plus whatever baseline
structural residual is present on all four FPAs.

### 9h. Built — `gd_render.py`, and a new confirmed mechanism: irreducible PSF-driven bias on a uniform scene
The discrete row-crossing truth generator from §5 item 5 is now built:
`geocarb_gert.gd_render.image()` renders the full 1024×1024 focal-plane image
directly from the real GD polynomials (not `FocalPlaneModel`'s analytic
formulas), for any scene function. It connects to retrieval via a per-row
`gert.instrument.SpectralWindow` with `obs_grid` set to that row's real
per-pixel channel centres (`A(x, row)`) — model and truth are evaluated on
the *identical* grid, so there is no interpolation or dispersion-polynomial
approximation standing between them; any remaining mismatch is real physics,
not a grid artifact.

**A real implementation bug had to be found and fixed first.** The initial
version mirrored `FocalPlaneModel`'s architecture: a per-row spectral step,
then a cross-row spatial resample for keystone. That reintroduces error even
on a *perfectly uniform* scene — because real smile varies (slowly) with row,
the cross-row resample borrows content computed with a neighbouring row's
slightly different dispersion curve, silently mixing two wavelength
calibrations together. Diagnosed via a PSF on/off test: disabling the N/S PSF
should make keystone (and this bug) exactly null on a uniform scene, but a
~0.7–1 ppm CO₂ residual survived regardless — a smoking gun that something
*other than* the PSF was mixing rows. Rebuilt non-separable: every pixel is
evaluated at its own true `(wavelength, slit-position)` pair directly, with
no cross-row borrowing anywhere except the final N/S PSF blur — the one step
where cross-row mixing is physically correct, not an artifact.

**Result, FPA2, uniform desert scene, dispersion order 2 vs. 4:**
- **PSF disabled:** bias converges to the *identical* value (-0.0005 ppm, at
  the level of the retrieval's convergence tolerance) at every row tested
  (25, 100, 300, 512, 700, 900, 950) — order-2 dispersion retrieval fully
  absorbs smile+keystone on a uniform scene, exactly reproducing the
  idealized Phase-1 conclusion, now with the real curves. Confirms the
  separable-version artifact is gone.
- **PSF enabled (1.5 px FWHM):** a real, order-*independent* residual bias
  survives — order-2 and order-4 agree to ~0.003 ppm at every row, so it is
  not an under-fitting effect. Bias ranges from near-zero (row 512/700) to
  -5.9 ppm (row 100), non-monotonic and asymmetric — row 100 is *worse* than
  row 25 despite row 25 being closer to FPA2's real keystone-null point
  (§9c) — reflecting genuine complexity in the real curve, not an idealized
  symmetric bowl.

**Mechanism:** the along-slit PSF genuinely blurs together detector rows
that each have a slightly different real smile calibration. The signal read
out at any one row is therefore a blend of several different wavelength
calibrations, which no single per-row dispersion polynomial — of any order —
can represent, because the error isn't in *that row's* dispersion, it's in
having borrowed neighbours' dispersion too. This is a **new, previously
undocumented bias mechanism**, distinct from both the classical
keystone-heterogeneity mechanism (§4, needs along-slit scene structure) and
the row-crossing/aliasing extraction mechanism (§9b, about *which* physical
rows get combined during readout) — this one is a pure smile×PSF coupling
effect, present even on a perfectly uniform scene, and it does not respond
to floating a higher dispersion order.

Two smaller fixes landed alongside this: `barcode_scene` (originally
alternated two different spectra) now correctly alternates the *brightness*
of one identical spectrum — matching a diffuser-panel illumination pattern
where the atmospheric column, and hence spectral shape, doesn't change along
the slit, only reflectivity does — with independently specifiable bar widths
and brightness levels. And the per-pixel ILS convolution (`gd_render.
_diagonal_ils_convolve`) is windowed (direct index slicing on the uniformly-
spaced hi-res grid, instead of masking the full array ~1e6 times per render)
for a 2× speedup; a further speedup would need compiled code; see the
performance note in `gd_render.py`.

### 9i. The smile-slope-null row is not the keystone-null row
§9h noted row 100 shows *worse* PSF×smile bias than row 25, despite row 25
being FPA2's real keystone-null point (§9c) — surprising if you expect one
"sweet row" to govern everything. It doesn't, and the reason is precise:
keystone-null (row 25) and the PSF×smile mechanism depend on two different
derivatives of the same GD polynomials.

- **Keystone-null (row 25):** where `B(x, row)` has zero spread *across a
  row's own columns* — driven by clocking (the FPA/grating rotational
  misalignment, §9c).
- **Smile-slope-null (row 519, at FPA2's actual band centre):** where
  `∂A(x, row)/∂row` — how fast the *wavelength calibration itself* shifts
  between adjacent rows — is smallest. This is what governs the PSF×smile
  mechanism (§9h): the PSF blends adjacent rows' dispersion curves, and that
  blend is most benign wherever those curves are locally flattest with
  respect to row, independent of where keystone happens to be null.

Verified directly from the real coefficients: `∂ν/∂row` (evaluated at the
band-centre column) is minimised at row 519 for FPA2, and this holds within
1 row across the *entire* spectral range (checked at columns 4, 256, 512,
768, 1019 — all give row 518–519) — not an artifact of picking one column.
Keystone and smile are described as orthogonal distortions in this plan
(§1), and this is a concrete instance of that orthogonality: their
respective critical points don't have to coincide, and for FPA2 they don't.
The near-symmetric bowl centred close to slit centre in §9h's PSF-enabled
result is therefore the genuine signature of the smile-slope mechanism, not
a discrepancy to explain away.

### 9j. Barcode/transition tests: keystone-heterogeneity is real but narrowly localized
Three scene tests beyond the uniform case, all FPA2, dispersion order 2
unless noted:

- **Barcode (32 bars, brightness-only, same spectrum)**: matches the
  uniform-scene PSF×smile baseline (§9h) almost everywhere, **except a sharp,
  localized spike at row 100** — CO₂ bias -12.4 ppm vs. -2.1 ppm baseline (an
  order of magnitude worse), chi2 also degrading badly there (0.62 vs.
  ~0.003–0.005 at neighbouring rows) — because row 100 happens to land only
  ~4 rows from a bar boundary (bars are 32 rows wide). This is the classical
  keystone-heterogeneity mechanism (§4) showing up concretely for the first
  time with real curves — and it is **highly localized**, not a smooth
  function of distance from any sweet row.
- **Realistic (6 broad desert/forest/grass/water segments)**: matched the
  uniform baseline almost exactly at all 7 originally-tested rows — not
  because heterogeneity doesn't matter, but because none of those rows
  (chosen by keystone-crossing amount) happened to land near one of the 5
  segment boundaries.
- **Systematic 12-transition test**: all 4×3 ordered surface-type pairs
  (desert/forest/grass/water), boundaries placed at 6 rows spanning the full
  keystone range (100–850, `rows_crossed` 0.83–8.69), tested at each
  boundary ±10 rows (never exactly on a boundary — with `softness=0` that's
  a mathematical discontinuity, and an order-0 retrieval diverged into an
  unphysical atmospheric state there on the first attempt). Result: a clean
  **null** — all 6 surface pairs collapse onto the identical PSF×smile-only
  curve at every row (spread ~0.0001–0.0002 ppm, pure numerical noise),
  regardless of which two surface types were on either side. So the
  keystone-heterogeneity contamination footprint is narrower than 10 rows —
  bounded between roughly 4 rows (where the barcode test's row 100 sat from
  its nearest boundary) and 10 rows (where this test found nothing) — but
  not yet pinned down precisely (§10).
- **Residual-shape check on the null result**: re-ran the highest-contrast
  pair (desert-water) at the two highest-keystone boundaries (700, 850,
  ±10 rows), this time saving the full post-fit spectral residual, not just
  the aggregate bias/chi2 (`scripts/gd_transition_residuals.py`, output in
  `results/gd_transitions_residuals.pkl`). The dispersion-corrected residual
  shows the same "beat"-envelope pattern as the uniform-scene case (§9g/9h)
  — smallest near band centre, growing toward both edges — at all four rows,
  with **no additional feature distinguishing the two rows on either side of
  a boundary** from each other. This confirms the null result at the level
  of residual *shape*, not just the retrieved gas amount: at ±10 rows, there
  is no detectable trace of the scene transition anywhere in the fit.

**Performance fix found along the way:** `random_scene`/`edge_scene`'s
multi-segment blend originally did one full `(n_pixels, n_hires)` array add
*per boundary*, so cost scaled with segment count — a 6-segment scene took
~240s to render vs. ~40s for a 32-bar `barcode_scene` (which is O(1) in
segment count, since brightness blending is scalar-per-pixel). Refactored
into a shared `_segment_blend` engine: compute small per-segment blend
weights (shape `eta.shape + (n_seg,)`), then one matrix multiply against the
stacked segment spectra — ~5× faster, cost no longer scales with segment
count.

### 9k. Known assumptions in the `gd_render` retrieval pipeline
Everything in §9h–9j shares the same test harness, which carries assumptions
worth stating explicitly rather than leaving implicit:

- **Radiative transfer:** `SingleScatterSolver` only — no aerosols, no
  multiple scattering (Phase 4 in this plan's original scope, not yet
  engaged by any of this work). Fixed US Standard Atmosphere
  (`reference_atmosphere()`), identical for every row and every scene test —
  only albedo varies along the slit, gas columns never do. Fixed single
  geometry (`sample_geometries(..., n=1, seed=0)`) for every row, even
  though a real N/S slit scan has genuinely different viewing geometry at
  different slit positions — this harness isolates the geometric-distortion
  effect specifically, by design.
- **Instrument model:** one Gaussian ILS (FWHM from nominal resolving power)
  for every row/pixel — not a real measured/tabulated ILS, and not spatially
  varying. Gaussian PSF fixed at 1.5 px FWHM — spatial blur only, no stray
  light or ghosting (§4.9 of `keystone_report.pdf`). Truth includes the real
  geometric distortion and PSF only — not FPA2's separate calibration-gap
  (§9d) or any detector-level effects (nonlinearity, dark current,
  flat-field residuals).
- **Retrieval configuration:** noise model is an ad hoc flat 0.3%-of-peak
  floor (`sigma = max(0.003·|y|.max(), 1e-6)`), not a physically-derived
  SNR/shot-noise model (unlike `FlatSNR(300.)` in the earlier
  `compute_keystone_bias.ipynb` pipeline) — so chi2 values here are not on
  the same scale as that pipeline's. Prior albedo is always desert
  regardless of true scene content. Convergence uses `dx_norm`,
  `dx_tol=0.01`, `max_iter=14` (established earlier this session), not
  re-validated against tighter tolerances. **Single-band retrieval only**
  (FPA2 in isolation) — never GeoCarb's real multi-band joint retrieval
  (O2-A + weak CO₂ + strong CO₂ + CH₄/CO together), so these ppm-level
  numbers are not directly comparable to a real L2 product. Every row
  retrieved independently, no cross-row regularization. The retrieval's only
  miscalibration nuisance parameter is the generic dispersion polynomial —
  it has no spatial/keystone-correction mechanism at all (presumably
  realistic, but an assumption about what "the retrieval" means here).
- **Scene construction:** barcode/transition scenes use sharp edges
  (`softness=0`); the "realistic" scene is 6 broad segments built from 4
  fixed placeholder albedo values that `geocarb_gert/scene.py` itself
  documents as "typical clear-sky nadir reflectances, not a specific
  measured spectrum" — not real satellite imagery or measured land-cover
  statistics.
- **Scope:** everything in §9h–9j is **FPA2 only** — the band with the worst
  clocking offset (§9c). None of FPA0/1/3 have been run through this
  pipeline yet, so it isn't yet known whether the PSF×smile bias magnitude
  found here is FPA2-specific or general to all four bands.

**Status:** row-crossing/aliasing (9b), the FPA2 calibration gap (9d), the
real smile amplitude (9f), and the PSF×smile coupling bias (9h/9i) are the
best-supported, quantified corrections to the Phase-1 truth generator, with
clocking (9c) explaining *where* row-crossing is worst per band and 9g
setting expectations for how far a smooth-polynomial truth generator can go.
Round-trip self-consistency (9e) is ruled out. The truth generator itself is
built and validated on a uniform scene (9h) and partially validated on
non-uniform scenes (9j, one clean null result plus one strong positive) —
remaining work is pinning down the keystone-heterogeneity contamination
footprint width precisely, extending beyond FPA2, and working through §9k's
assumption list where it matters most (§10).

### 9l. Built — rectify-then-retrieve verification test: rectification-interpolation bias dominates over the GD-curve effects found so far

Everything in §9h–9k retrieves directly on a row's own *native* per-pixel
grid (`obs_grid` = that row's true per-column wavelengths from
`xy_to_wavelength_slit`) — deliberately, to isolate the GD-curve mechanisms
from any grid-mismatch confound. But the real ground-test pipeline doesn't
retrieve on the raw native grid: it first **rectifies** the raw detector
image onto a regular (slit, wavenumber) grid via the inverse polynomial
mapping (`keystone_report.pdf` §4.8.1, the C/D pair), *then* retrieves. This
section asks whether running the same two-stage process through this
simulator reproduces the qualitatively severe bias/non-convergence gap
between the Phase-1 idealized result and real ground-test data (§9's
motivation) — the test the user proposed as the infrastructure's realism
check for the planned OSSEs.

**Built:** `gd_render.rectify(fpa, A, s_grid, wn_grid)` — for each point on a
regular output (slit, wavenumber) grid, uses `wavelength_slit_to_xy` (the
C/D pair) to find where it came from in the raw rendered image, then
interpolates (`scipy.ndimage.map_coordinates`, bilinear by default) —
standard inverse-mapping image resampling, matching real L1B rectification.
`scripts/gd_rectify_retrieve.py` renders a uniform-desert FPA2 scene,
rectifies it onto a 1024-slit x 1075-wavenumber grid (the standard nominal
FPA2 window from `build_geocarb_instrument()`), then retrieves at 7 rows
(25, 100, 300, 512, 700, 900, 950) with dispersion order 0 and 2.
`scripts/gd_rectify_plot.py` produces `plots/gd_raw_vs_rectified_fpa2.png`.

**Bug caught and fixed:** the first run diverged on every one of the 14
row/order combinations (NaN in the ABSCO lookup, singular Jacobian,
non-increasing pressure levels). Root cause: `gert.ForwardModel` always
returns `y` in ascending-*wavelength* order (`wl_instrument =
wn_instrument[::-1]`, "reverses to wavelength order" in
`forward_model.py`), regardless of what order `obs_grid` is supplied in —
but the rectified row was extracted in ascending-*wavenumber* order
(matching `wn_grid`, which was built ascending). That's an exact
index-for-index reversal: channel 0 of the data was being compared against
channel *N* of the model, channel 1 against *N-1*, etc. Iteration 0 chi2 was
~12,700 as a result, and the linearized Gauss-Newton step immediately
overshot into an unphysical atmospheric state (negative gas scale, negative
pressure) on every row. This is a bug specific to this script — every
earlier native-grid script in §9h–9j happened to extract `A[i, :]` in raw
column order, which for this instrument's real dispersion direction already
came out in ascending-wavelength order, so the mismatch never showed up
before. Fixed by reversing the extracted row (`row[valid][::-1]`) before
handing it to `GERTRetrieval`.

**After the fix:** order=2 (dispersion floated) converges cleanly
(chi2_reduced ≈ 3.2–4.4) at 6 of 7 test rows; order=0 also mostly converges
but is far noisier. Row 25 (near the slit edge, `rows_crossed≈0`) diverges
under *both* orders — order=0 converges to an unphysical state (chi2 ≈
10^117, `conv=False`); order=2 crashes outright ("p_levels must be strictly
increasing"). The underlying rectified data at row 25 was checked directly
and is well-behaved (no NaN/garbage among the 1052 valid channels, sane
radiance range) — so this looks like generic Gauss-Newton fragility near
the slit edge rather than a data problem, consistent with the divergence
pattern already seen in §9j's boundary tests. Not chased further this
session.

| order | row 100 | row 300 | row 512 | row 700 | row 900 | row 950 |
|---|---|---|---|---|---|---|
| 0 | −115.6 | −40.7 | −104.8 | −42.0 | −55.7 | −88.3 |
| 2 | −52.5  | −62.3 | −54.4  | −62.9 | −61.2 | −55.7 |

(CO₂ bias, ppm; row 25 excluded — diverged/unphysical both orders.)

**Isolating the cause:** to check whether this bias is a GD-curve effect
(like §9h/9i) or a retrieval artifact, the rectified row 512 was compared
directly against the *true* spectrum at that exact slit position — computed
independently, without ever constructing the raw 1024×1024 detector image
or calling `rectify()` at all (same `radiance(eta)` callable and
`_diagonal_ils_convolve` helper `gd_render.image()` uses internally, just
evaluated once at the fixed slit position instead of per-row). The residual
is essentially identical (mean |resid| ≈ 0.82 against a mean signal of 6.54,
~13%) to what the retrieval sees against the forward model at the prior
state. **This means the CO₂ bias is not primarily a retrieval or GD-curve
artifact — it is bilinear-interpolation smoothing/aliasing introduced by
`rectify()` regridding the raw detector image onto a regular grid.** Visibly
the two curves nearly overlap (`plots/gd_raw_vs_rectified_fpa2.png`, panel
3) — the ~13% residual is a subtle, sub-pixel line-registration effect, not
a gross distortion, which is consistent with column-retrieval sensitivity
being high enough to turn a barely-visible spectral perturbation into a
tens-of-ppm gas-column bias.

This is a **new mechanism**, distinct from and much larger than both
mechanisms found so far: the PSF×smile coupling bias (§9h, up to ~−5.9 ppm)
and the keystone-heterogeneity localization (§9j, a few ppm within ~4–10
rows of a scene boundary). Unlike those two, it does **not** go to zero at
the smile-slope-null row (~519, §9i) — with order=2, the bias is a roughly
uniform −50 to −65 ppm at *every* converged row, including 512 (right next
to the null row). That's consistent with it being a broad, generic
regridding effect rather than one localized to keystone-crossing or
smile-slope-extremum regions.

**Interpretation for the OSSE-realism question:** this test reproduces both
halves of the qualitative real-data signature that motivated §9 — large
bias (tens of ppm, not the few-ppm level found in every native-grid test)
and poor/failed convergence (row 25, and order=0's row-to-row instability)
— once the simulator is run through the same two-stage rectify-then-retrieve
process the real pipeline uses. That's evidence this infrastructure is
realistic enough for the planned OSSEs, provided the rectification step is
included and not skipped in favor of native-grid retrievals.

**Not yet done:** compare against real ground-test data directly (this
section is simulation-only, like 9h/9j before their real-data checks were
filled in at §10); test whether higher-order interpolation
(`gd_render.rectify(..., order=3)`, cubic) meaningfully reduces this bias,
which would indicate real L1B processing's exact resampling scheme matters;
extend beyond FPA2; and diagnose the row-25-style divergence pattern
directly rather than just recording it as "diverged."

### 9m. Dense along-slit sweep (1024 rows): §9l's bias confirmed non-flat, a real `gert` bug found and fixed, and honest failure-rate statistics

§9l sampled only 7 rows. This section reruns native vs. rectified retrieval
at **every** row (1024, both dispersion orders) to look for systematic
structure in the rectification-interpolation residual, per the user's
request to "find the cause of the errors from the rectification by looking
at systematic patterns in the residuals." Two infrastructure problems had to
be fixed first; both are documented here because they change how earlier
results in this document should be read.

**Performance.** A dense sweep needs ~4× the earlier row count, twice
(native + rectified), twice again (2 dispersion orders) — profiling showed
each retrieval spent ~62% of its wall time in `gert.instrument.ILS.
convolve()`, which scans the *entire* hi-res wavenumber grid per instrument
channel (an unwindowed `np.argmin` plus a full-array delta/mask) rather than
using the local support the Gaussian kernel actually needs — the same
anti-pattern already fixed in this repo's own `gd_render._diagonal_ils_
convolve` earlier this session. Fixed in `gert/instrument.py` with a
windowed lookup (index arithmetic on the known-uniform hi-res grid, same
technique). Verified bit-exact against the original algorithm on synthetic
gaussian/tabulated/top_hat cases (1D and 2D, both `exact_center` modes,
non-unit `width_scale`) and via `sanity_checks/dispersion_checks.py` (18/18
pass, unchanged) — this is a pure performance fix, not a behavior change.
Gives roughly a 1.7× per-retrieval speedup. Combined with running rows in
parallel (they're fully independent retrievals) via `multiprocessing`'s
`fork` start method — the ~2.2GB ABSCO table is loaded once in the parent
before the worker pool starts, so every worker shares it via copy-on-write
instead of re-loading it — 16 workers took the full 4096-retrieval sweep
from an estimated multi-hour serial runtime down to ~26 minutes.

**A real `gert` bug, found by this sweep and fixed at the user's request.**
The first full sweep showed 555 rows on the rectified pipeline reporting
`converged=True` with `chisq_reduced=nan` and wildly unphysical bias
(sometimes billions of ppm) — `GERTRetrieval`'s `dx_norm` convergence
criterion measures *step size*, not chi2 validity, so a Gauss-Newton step
that overshoots into an unphysical state (negative gas scale driving
pressure/ABSCO lookups out of range) could still register as "converged" if
the resulting step size happened to be small. Root cause had two parts, both
fixed in `gert/retrieval.py` (both `GERTRetrieval.run()` and
`DecoupledGERTRetrieval.run()`):

1. An in-loop check right after each iteration's chi2 is computed — catches
   a non-finite chi2 immediately rather than computing a Jacobian (which can
   itself raise) at an already-bad state.
2. A second, initially-missed check on the *final* forward call after the
   loop exits — necessary because `dx_norm` accepts a step and breaks the
   loop *before* that step's own chi2 is ever tested inside the loop; the
   post-loop "final forward call" was the first place that state's chi2 was
   evaluated, and it wasn't being checked either. (Confirmed by reproducing
   the exact failure: row 4 initially still showed `converged=True,
   diverged=False, chi2=nan` after the first fix — only the second closed
   the gap.)

Both paths now set `converged=False` and a new `RetrievalResult.diverged =
True`, and skip re-evaluating the forward model at the diverged state
(which is exactly what raises the `ValueError`/`p_levels not increasing`-
style crashes seen elsewhere in this document — re-running it after
detecting divergence would just risk hitting the same crash). Verified: the
row-4 case now correctly reports `diverged=True`; row 512 (a normal case)
is bit-identical to its pre-fix result (chi2=3.2299, co2_scale=0.8690);
`dispersion_checks.py` still 18/18. **This means the "555 falsely
converged" number from the first pass of this sweep was itself partly
wrong** — re-analysis after the fix shows most of those were order=0 rows
with a large but *finite* chi2 (legitimately converged by step size, just a
poor fit — expected, since order=0 has no dispersion term to absorb
geometric distortion) that an overly strict ad hoc filter had misclassified
as bugged. The real bug count was smaller; it is now exactly zero after the
fix (confirmed: 0 rows with `converged=True` and non-finite chi2 in the
corrected sweep).

**Results (corrected sweep, `results/gd_dense_sweep.pkl`, `plots/
gd_dense_sweep_fpa2.png`):**

- **Native pipeline: converges cleanly at all 1024 rows, both orders** —
  chi2 ≤ 1.7 (order 0) / ≤ 0.004 (order 2) everywhere. Zero divergences.
  This confirms (again) that the native per-row grid is not itself a source
  of instability; every failure mode below is specific to the rectified
  pipeline.
- **The rectification bias is not flat.** §9l's 7-point sample looked like
  a roughly constant −50 to −65 ppm. At full density it's a real U-shaped
  envelope (least negative near the slit centre, most negative toward the
  edges) with oscillatory fine structure on top — visible directly in the
  dense bias-vs-row curve.
- **The residual heatmap shows the mechanism directly**: dense vertical
  striping locked to individual absorption-line positions, not smooth
  broadband error — consistent with a sub-pixel *line-registration* error
  (worst where the spectral gradient is steepest) rather than a gross
  distortion, matching the single-row finding in §9l.
- **Residual magnitude vs. keystone severity is only weakly correlated.**
  Mean |residual| has a "floor" (~0.048) present even at rows with near-zero
  `rows_crossed`, growing only ~25% toward the highest-keystone rows. The
  bias is present almost everywhere, with keystone amount as a secondary
  modulator — not the primary driver.
- **Honest failure-rate breakdown** (now trustworthy, post-fix): rectified
  order=0 fails on 139/1024 rows (38 `diverged` + 101 stalled at
  `max_iter` without converging, ~13.6%), scattered fairly broadly across
  the slit rather than only at the edges, and even its "successful" rows
  have wildly variable chi2 (9.7 to 1338) since nothing is absorbing the
  geometric distortion. Rectified order=2 fails on 112/1024 rows (91
  diverged + 21 stalled, ~10.9%) — a similar *count* to order=0, but
  concentrated much more tightly near the slit edges, and its converged
  chi2 is tight (3.2–4.7) rather than wildly variable. Order=2 isn't
  dramatically more reliable by raw failure count than order=0 on this
  pipeline, but it is far more trustworthy *when* it converges. 17 rows are
  off-detector near one slit edge for both orders (matches §9l).

**Interpretation.** The core §9l finding survives full-density scrutiny and
gets sharper: the rectification-interpolation bias is a broad,
line-registration-driven effect present almost everywhere on the slit, only
weakly modulated by keystone amount — not a smooth function of geometric
distortion severity the way the earlier native-grid mechanisms (§9h's
PSF×smile, §9j's keystone-heterogeneity) were. That, plus a failure rate
above 10% for both dispersion orders on the rectified pipeline (with order=0
producing many "successful" but essentially meaningless fits), is a second,
independent piece of evidence — alongside §9l's original one — that
rectify-then-retrieve reproduces the qualitatively severe bias/convergence-
failure signature this whole investigation set out to explain (§9's
motivation).

---

## 10. Real-data observations (fill in as evidence is pulled)

**Purpose.** The forward model only needs *just enough* fidelity to reproduce the
artifacts actually seen in ground-test data gathering/analysis — not full physical
realism. Each row below should tie a specific, concrete real-data symptom to the
model feature it implies, so this table doubles as the requirements list for §5's
infrastructure items. Prefer pulling an actual plot/log/number from the test archive
over a remembered description — a vague "it didn't converge well" doesn't tell us
what to build; "chi2 stalled at 40 for scenes with slit position |η|>0.6, spectral
residual had a sharp step at channel 512 in band 2" does.

| # | Symptom (what was observed) | Evidence (plot / log / dataset ref) | Slit position(s) / row(s) / band(s) | Implied model requirement | Status |
|---|---|---|---|---|---|
| 1 | Retrievals against real upward-looking ground-test data failed to converge, **except** in scenes where the dispersion trace stayed within one detector row | `keystone_report.pdf` §4.8.2, Figs. 36–38, 41 — "aliased by the pixel grid"; row-averaging spans ~2 rows (sweet spot) to ~7–10 rows (slit ends) | All FPAs; up to ~10 rows crossed at slit ends, matching `keystone_px=20` | Discrete row-crossing truth generator (§5 item 5): model extraction as averaging `N(η)` true rows, not a smooth per-band polynomial | **Confirmed**, see §9b |
| 2 | Real keystone-vs-slit-position curve is asymmetric; each FPA's zero-keystone point is offset from slit centre, worst for FPA2 (row 25 of ~1024, nearly at the slit edge) | `keystone_report.pdf` Fig. 38/41 | FPA0 sweet row 600, FPA1 570, FPA2 25, FPA3 266 | Per-band **offset**, not just amplitude, in the keystone term — `η²` centred on 0 is wrong for FPA2 | **Confirmed**, see §9c |
| 3 | FPA2 (strong CO₂) residual-optimization image showed vertical striping surviving even after FTIR-based polynomial refinement | `keystone_report.pdf` §4.8.6, Fig. 45 (round-trip error 0.079 px mean / 0.30 px peak for FPA2 vs. 0.025–0.03 px others); user-provided FPA2 "square of the difference" images (2026-07-22, summed residual 4933.65 → 2128.96) | FPA2 (strong CO₂), region with insufficient laser wavelength coverage during ground test | A calibration-coverage-gap term for FPA2 specifically, independent of row-crossing | **Confirmed** (separate, compounding mechanism), see §9d |
| 4 | Geometric self-consistency (round-trip x,y→λ,s→x,y) error | `keystone_report.pdf` §4.8.4 — "easily absorbed in L2 retrieval processing" | All FPAs (except FPA2's uncalibrated region) | None — not a significant driver | **Ruled out**, see §9e |
| 5 | Real smile amplitude is ~33–39 px at band centre (15–20× the `SMILE_PX=2.0` placeholder), and grows toward band edges (FPA0: ~37 px short-wavelength edge vs. ~48 px long-wavelength edge) | `gcmap_em27.csv` C/D polynomials via `geocarb_gert.gd_polynomials` (2026-07-22) | All FPAs; within-band variation checked for FPA0 | Real, per-band, wavelength-dependent smile amplitude in the truth generator — not a constant placeholder | **Confirmed**, see §9f |
| 6 | On a *uniform* scene, real (non-separable) rendering + real N/S PSF leaves an order-independent CO₂ bias of up to ~-5.9 ppm that no dispersion order absorbs; with the PSF disabled the same setup converges to exactly 0 at every row | Synthetic (`gd_render.py`), not yet checked against real data | FPA2, rows 25–950, orders 2 and 4 agree to ~0.003 ppm | A real-data prediction to test: does actual GeoCarb data show an order-independent residual bias pattern under uniform illumination that scales with PSF/smile coupling rather than scene structure? | **Confirmed in simulation** (2026-07-22, see §9h); **not yet checked against real data** |
| 7 | On a barcode scene, keystone-heterogeneity bias is a sharp, narrowly localized spike (-12.4 ppm vs. -2.1 ppm baseline) only within a few rows of a scene boundary — a systematic 12-transition test at ±10 rows from boundaries found *zero* signal (all 4 surface types, spread ~0.0001 ppm) | Synthetic (`gd_render.py`), not yet checked against real data | FPA2; footprint bounded between ~4 rows (signal) and ~10 rows (null) | A real-data prediction: keystone-heterogeneity bias in real scenes should appear only within a narrow row-distance of genuine scene edges (coastlines, field boundaries), not as a smooth function of keystone amount | **Confirmed in simulation** (2026-07-22, see §9j); footprint width not yet pinned down; **not yet checked against real data** |
| 8 | Retrieving on the raw native per-row grid (§9h–9j) gives few-ppm bias that vanishes at the smile-null row; retrieving on a *rectified* (regridded) spectrum — mirroring the real L1B rectify-then-retrieve pipeline — gives a much larger (tens-of-ppm), roughly row-independent bias plus one outright-diverging row, closer in *severity* to the "much more optimistic than real data" gap this whole study started from (§9's motivation) | Synthetic (`gd_render.rectify` + `gd_rectify_retrieve.py`, `plots/gd_raw_vs_rectified_fpa2.png`), not yet checked against real data | FPA2; rows 100–950 give order=2 bias −41 to −63 ppm (not zero at smile-null row 519); row 25 diverges under both dispersion orders | A real-data prediction: if real ground-test bias/non-convergence is dominated by the rectification-interpolation step rather than the underlying GD curves, the effect should NOT vanish near the calibrated smile-null row the way the PSF×smile bias (row 6) does | **Confirmed in simulation** (2026-07-23, see §9l); **not yet checked against real data** |
| 9 | At full row density (not just 7 sparse points), the rectified-pipeline bias is a real U-shaped envelope with oscillatory fine structure (not flat); the post-fit residual is locked to individual absorption-line positions (sub-pixel line-registration error); and both dispersion orders fail on >10% of rows on the rectified pipeline (order=0: 13.6%, scattered broadly; order=2: 10.9%, concentrated near slit edges) even though the *native* pipeline converges cleanly at all 1024 rows | Synthetic (`gd_dense_sweep.py`, `plots/gd_dense_sweep_fpa2.png`), not yet checked against real data | FPA2, all 1024 rows, both dispersion orders | A real-data prediction: real ground-test non-convergence should be far more common on rectified/regridded data than on any per-row-native-grid analysis, at a rate order 10%+, not just occasional edge cases — and should show line-locked (not smooth) residual structure | **Confirmed in simulation** (2026-07-23, see §9m); **not yet checked against real data** |

**Still open / worth checking the archive for:**
- **Spectral residual shapes from real (even non-converged) retrievals** — any
  systematic (not noise-like) `y_obs − y_model` pattern, especially how it differs
  between row-crossing and non-row-crossing scenes, to check against what the new
  truth generator (§5 item 5) predicts.
- **Whether smile's within-band amplitude growth is a pure scaling or a shape
  change** (§7 item 5, §9f) — check across all four FPAs, not just FPA0.
- **Does real ground-test data show the PSF×smile coupling signature from §9h** —
  an order-independent residual bias under uniform illumination — separately from
  the row-crossing/aliasing and FPA2-calibration-gap mechanisms already matched to
  evidence?
- **Pin down the keystone-heterogeneity contamination footprint width** (§9j) —
  test offsets between 2 and 10 rows from a boundary to find where the effect
  actually turns on, rather than the current 4–10 row bracket.
- **Extend §9h–9j beyond FPA2** — confirm whether the PSF×smile bias magnitude and
  the contamination footprint width are FPA2-specific (worst clocking offset, §9c)
  or general to all four bands.
- **Does real ground-test data show the rectification-interpolation bias signature
  from §9l** — a bias that stays roughly constant across the slit rather than
  vanishing at the smile-null row — separately from the PSF×smile and
  keystone-heterogeneity mechanisms already matched to evidence?
- **Test interpolation-order sensitivity in `gd_render.rectify`** (§9l) — does
  cubic (`order=3`) meaningfully reduce the ~13% residual found with the default
  bilinear regrid, and if so, does real L1B processing's resampling scheme matter
  for the magnitude of this bias?
- **Diagnose the row-25-style divergence pattern directly** (§9l) rather than just
  recording it as diverged — is it generic Gauss-Newton fragility near the slit
  edge, or something specific to that row's rectified data?
- **Replace the ad hoc flat noise model** (§9k) with a physically-derived one
  (e.g. `FlatSNR`, matching the earlier `compute_keystone_bias.ipynb` pipeline) and
  check whether any §9h–9j conclusions change.

**How this feeds back:** once a row here has real evidence attached, it either
confirms an existing hypothesis (§9) or becomes a new one — either way it turns
directly into a truth-generator requirement in §5 and a target pattern the
`compute_keystone_bias.ipynb`-style diagnostics should be checked against, rather
than the model driving toward whatever's easiest to build next.

---

## 11. Plan: two-band (O2-A + strong CO2) joint retrieval (2026-07-23)

**Motivation.** Everything through §9 retrieves a single band (FPA2) in
isolation. The real instrument doesn't: GeoCarb's L2 algorithm fits O2-A
jointly with the CO2/CH4 bands specifically because O2-A constrains photon
path length (Rayleigh + aerosol scattering), which is otherwise degenerate
with the gas column in a single SWIR band alone. This section plans a
two-band (O2-A = FPA0, CO2_strong = FPA2) joint retrieval with a floated
aerosol state, to test whether that degeneracy-breaking behavior shows up in
this simulator, and whether it interacts with the rectification-
interpolation bias found in §9l.

### 11a. What's already built — no new work needed

- `build_geocarb_instrument()` (`geocarb_gert/instrument.py`) already builds
  a **multi-window** `gert.Instrument` — FPA0 (`O2_A`) and FPA2
  (`CO2_strong`) are just two of its four windows; a 2-band instrument is a
  subset, not a new construction path.
- `gert.retrieval.StateVector.gas_scaling` already sizes `albedo_{b}` /
  `albedo_slope_{b}` / dispersion polynomials **per band** from
  `len(prior_albedo)`, and already has first-class aerosol elements
  (`include_tau_aerosol`, `include_height_aerosol`,
  `include_thickness_aerosol`) — the joint gas+aerosol+per-band-nuisance
  state vector this test needs is an existing, exercised code path, not
  something to build.
- `geocarb_gert/scene.py`'s `albedo_for(instrument, surface)` already
  returns per-band albedo in window order, keyed by band label — already
  multi-band-aware.
- **The real GD calibration polynomials cover all four FPAs already**,
  confirmed by inspecting `gcmap_em27.csv`: every coefficient (A–H) has four
  columns (`A0..A3`, ..., `H0..H3`), one per FPA. `gd_polynomials.py`
  already dispatches on its `fpa` argument. FPA0 has simply never been
  *exercised* through `gd_render`/`rectify`/retrieval this session — only
  FPA2 has (§9h–9l).

### 11b. The genuinely new problem: cross-band ground co-registration

Both bands look through the *same physical slit*, so the shared coordinate
across bands is **real slit angle `s` [deg]**, not detector row index and
not either band's own normalized `eta` convention (`eta = s / s_max(fpa)`,
§`gd_render` module docstring) — `s_max` itself differs slightly per FPA.

Checked concretely (2026-07-23): at the *same raw pixel* (col=512, row=512),
`xy_to_wavelength_slit` gives:

| FPA | band | real `s` at (512, 512) |
|---|---|---|
| 0 | O2_A | +0.0225° |
| 1 | CO2_weak | −0.0422° |
| 2 | CO2_strong | −0.0355° |
| 3 | CH4_CO | −0.0227° |

FPA0 vs FPA2 alone differ by 0.058° at the identical pixel — with row
spacing ≈ `2·s_max/1024` ≈ 0.0045°/row, that's over **10 rows'** worth of
ground-position error. Pairing "row *k* of FPA0" with "row *k* of FPA2" (the
naive approach) is wrong by a large margin. This is exactly the clocking
effect §9c characterized for FPA2 alone, now shown to also misalign band
to band.

**Fix:** never align bands by row index. Rectify each band independently
onto a **shared physical `s_grid`** (real degrees, not per-band-normalized)
using that band's own C/D polynomial (`wavelength_slit_to_xy`) — this is
exactly the `gd_render.rectify()` machinery already built in §9l, it just
needs to target a common grid instead of each band's own `linspace(-s_max,
s_max, ...)`. `s_max` differs per FPA (checked: 2.2899° / 2.2820° / 2.2970°
/ 2.2858° for FPA0-3), so the joint `s_grid` must be clipped to the
intersection — and each band's own off-detector NaN edges (§9l found ~17
rows off-detector near FPA2's slit ends) will differ per band too, further
narrowing the usable joint range below either band's own valid range.

**Open question that can't be resolved from code alone:** were all four
FPAs' GD polynomials calibrated against the *same* external slit-angle
reference during ground test (`keystone_report.pdf`'s calibration chain,
§9e), so that "s = 0.03°" means the identical physical direction in FPA0's
and FPA2's polynomials? If yes, shared-`s_grid` rectification is sufficient
for co-registration. If each FPA's calibration used its own local/
independent reference, there is an additional, uncalibrated inter-band
boresight offset not captured in these polynomials at all, and the
retrieval would need an extra free "inter-band pointing offset" nuisance
parameter (or an independent geometric cross-calibration step) to absorb
it. **This is the single highest-leverage thing to check before building
further** — check the calibration provenance/report before writing any
cross-band registration code.

### 11c. Phasing

1. **Validate FPA0 individually first.** Nothing in §9h–9l has ever touched
   FPA0. Before trusting a joint retrieval built on top of it, re-run the
   same validation battery already done for FPA2: round-trip
   self-consistency (§9e's method), real smile amplitude and keystone/
   smile-slope-null rows (§9f/9i's method, expect *different* row numbers
   than FPA2's — row 25 keystone-null and row ~519 smile-slope-null are
   FPA2-specific, driven by FPA2's own clocking per §9c), and the uniform-
   scene rectify→retrieve bias check (§9l's method). No new code — these
   are the existing scripts parameterized with `fpa=0` instead of `fpa=2`.
2. **Resolve the shared-reference-frame question** in §11b before writing
   any cross-band registration code — it determines whether an inter-band
   pointing nuisance parameter is needed at all.
3. **Extend rectification to a shared, intersection-clipped `s_grid`**
   across the two bands (small extension of `gd_render.rectify`'s existing
   call pattern, not a rewrite).
4. **Extend scene construction** to a shared atmosphere (now including an
   aerosol layer — confirm/exercise `ForwardModel`'s aerosol Jacobian path,
   already present per `forward_model.py`'s `K_aer_lay` code) with
   per-band albedo (`albedo_for` already supports this; just needs to be
   called for a 2-band instrument instead of 1).
5. **Build the joint retrieval**: 2-window `Instrument`, `StateVector.
   gas_scaling(prior_albedo=[...2 values...], gases=['co2', 'o2', 'h2o'],
   include_tau_aerosol=True, include_dispersion=True, ...)` (note `'o2'`
   must be in `gases` since `GEOCARB_BANDS[0]`'s molecules are
   `['o2', 'h2o']`), single `GERTRetrieval.run()` over the combined
   measurement vector. Start with a **uniform scene** (matching the "null
   test" convention used for every test in §9) before anything more
   elaborate.
6. **Compare CO2-only vs. CO2+O2A+aerosol joint retrieval** under both the
   native-grid and rectified pipelines. Does adding O2-A actually reduce
   bias / break the aerosol-CO2 degeneracy the way it's supposed to?
   Working hypothesis: it should **not** touch the −50 to −65 ppm
   rectification-interpolation bias from §9l, since that's a spectral-
   registration artifact independent of photon path length/aerosol —
   worth confirming rather than assuming, since real data will have both
   effects simultaneously and a real retrieval can't turn one off to
   isolate the other the way this test can.
7. **Extend to non-uniform scenes** (§9j/9l's barcode/transition/realistic
   patchwork scenes) only once the uniform case is understood, to check
   whether cross-band registration error interacts with along-slit scene
   structure the way single-band rectification error did.

**Status:** planning only — no code written yet. §11a inventories what's
reusable; §11b's open question is the next concrete step (check
`keystone_report.pdf` calibration provenance) before phase 1 begins.
