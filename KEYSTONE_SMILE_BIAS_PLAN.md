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

## 9. Real-data check: confirmed mechanisms (2026-07-21 → 2026-07-22)

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

**Status:** row-crossing/aliasing (9b), the FPA2 calibration gap (9d), the
real smile amplitude (9f), and now the PSF×smile coupling bias (9h) are the
best-supported, quantified corrections to the Phase-1 truth generator, with
clocking (9c) explaining *where* row-crossing is worst per band and 9g
setting expectations for how far a smooth-polynomial truth generator can go.
Round-trip self-consistency (9e) is ruled out. The truth generator itself is
now built and validated (9h) — remaining work is applying it to non-uniform
scenes (barcode, realistic) and checking whether smile's within-band
variation is amplitude-only or a shape change (§7 item 5).

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

**How this feeds back:** once a row here has real evidence attached, it either
confirms an existing hypothesis (§9) or becomes a new one — either way it turns
directly into a truth-generator requirement in §5 and a target pattern the
`compute_keystone_bias.ipynb`-style diagnostics should be checked against, rather
than the model driving toward whatever's easiest to build next.
