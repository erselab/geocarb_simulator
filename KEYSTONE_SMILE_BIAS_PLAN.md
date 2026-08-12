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
| N/S PSF | **1.5 px** FWHM (`spatial_psf_fwhm_px=1.5`) ≈ 4.4 km ground blur. **Confirmed** — ground test reports the same ~1.5 px FWHM for all four FPAs (§9), consistent with an along-slit PSF set mainly by the shared telescope/slit image rather than by each arm's own downstream optics (not diffraction-limited — a diffraction-limited system would scale with wavelength, roughly 3× wider for FPA3 than FPA0; see §11b) |
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
   positions), 2D Gaussian centroid fit to sub-pixel precision. Mechanism
   (confirmed directly by the instrument team, 2026-07-27 — see §11b for the
   cross-band implication): co-aligned lasers spanning the relevant
   wavelengths shared one common optical path focused on the slit, so at any
   instant they all illuminated the *same physical point* on the calibration
   scanning mirror. The 5 slit positions were swept by steering that common,
   co-aligned beam with the mirror; the 11 wavelengths were swept separately
   by tuning laser frequency at fixed mirror position. Because every
   wavelength shares one physical beam at every slit-position step, each of
   the 55 lattice points has exactly one true physical slit angle common to
   *all four FPAs at once* — the polynomial fit inherits that shared
   reference directly from how the data was taken, not from an assumption
   made afterward.
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
- **Floating dispersion suppresses the residual by roughly an order of
  magnitude, without removing its structure.** `plots/gd_dense_sweep_fpa2.png`
  now shows the rectified-pipeline row x wavenumber residual heatmap for
  both orders side by side, on the same shared `wn_grid` (no interpolation
  needed for this comparison). The two heatmaps' natural color ranges differ
  by **~9x** (±1.72 for order=0 vs. ±0.19 for order=2) — a direct,
  quantified measure of how much the dispersion polynomial absorbs. Order=0
  also shows solid horizontal streaks at some rows in the heatmap — rows
  that technically converged (chi2 finite, so not excluded) but fit so
  poorly (chi2 up to 1338, above) that the whole row reads as residual,
  distinct from the fine vertical line-locked structure elsewhere. Even at
  the ~9x-suppressed order=2 scale, the line-locked pattern from §9n is
  still present — dispersion reduces the *amplitude* of the rectification
  error, it does not remove the *mechanism*.

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

### 9n. Spectral residual analysis: what the rectification error actually looks like, and a periodicity search that didn't find what it went looking for

Follow-up analysis of the `gd_dense_sweep.pkl` residuals saved in §9m, using
`scripts/gd_native_residual_periodicity.py` (`plots/
gd_native_residual_periodicity_fpa2.png`) plus a series of ad hoc checks.
Two separate questions, both worth recording carefully because the first
answer that seemed right (by eye or by a quick correlation) turned out to
be wrong or incomplete on closer inspection.

**Rectified-pipeline residual mechanism — corrected.** §9l/9m described the
rectification residual loosely as a "sub-pixel line-registration" error.
Quantitatively checking that against `gd_dense_sweep.pkl`'s residuals:

- **Row-to-row coherence is very high**: mean correlation 0.96 between
  adjacent rows' residual vectors (894 adjacent pairs checked, order=2) —
  confirms the pattern is a smooth, physically coherent function of row, not
  noise.
- **But it is *not* a simple sub-pixel wavelength shift.** Correlating each
  row's residual against the local first derivative of the model spectrum
  (`dI/dν`, the signature a pure registration shift would produce) gives
  essentially zero correlation (|r| ≲ 0.05 across every row tested). The
  second derivative (curvature) correlates just as weakly pointwise. Both of
  these directly contradict the "shift" language used in §9l — that
  characterization should be considered superseded by the following.
- **Directly at absorption line centers, the real mechanism is clear and
  one-sided**: comparing the rectified spectrum (`Rimg`, real `rectify()`
  output) against the true spectrum at the same slit position (computed
  independently, bypassing `rectify()` entirely — the same isolation method
  as §9l) at each line's local minimum, the rectified spectrum is
  **systematically brighter — shallower lines — at 98.6% of line centers
  tested** (819/831, 11 rows spanning the slit), never the reverse, by 2–18%
  of local line depth depending on row. Pointwise derivative correlation is
  too local/noisy to see this; the *integrated* effect across each line's
  width is real, consistent, and one-directional. This is bilinear
  interpolation doing exactly what it's expected to do to a sharp, narrow
  feature — averaging across it, pulling the interpolated value up toward
  the brighter surrounding continuum — and it directly explains why the
  retrieved CO₂ bias in §9l/9m is always negative, never positive: a
  measured spectrum with systematically weaker apparent absorption than
  truth reads as less gas than truth.
- Caution for future work: comparing native-pipeline residuals (each row on
  its own true per-pixel grid) against rectified-pipeline residuals (shared
  `wn_grid`) side-by-side would require resampling one onto the other's
  grid — which would itself smooth out exactly the kind of sharp,
  line-center-localized structure this section just characterized, and
  should not be done without accounting for that.

**Native-pipeline residual periodicity — hypothesis not confirmed, different
pattern found instead.** The original question was whether native-grid
residuals show periodic structure whose *period* varies row to row (a
plausible signature of row-dependent local dispersion nonlinearity beating
against the real absorption-line comb). Checking this directly:

- **Order=0** residuals are broadband/noise-like at every row tested — FFT
  peaks are low-power (power fraction ~1–1.5%) and scattered with no
  consistent location, consistent with unmodeled geometric-distortion
  mismatch rather than a clean periodicity.
- **Order=2** residuals do **not** show a row-varying period. Instead, every
  row tested shows a burst of high-frequency oscillation concentrated in
  the *same* spectral location — roughly **4865–4895 cm⁻¹**, the upper ~30
  cm⁻¹ of the FPA2 band — regardless of row. A fixed location, not a
  row-dependent period.
- **The magnitude is real but more modest than it looks by eye**: residual
  std in that zone is only ~12–17% higher than the rest of the band for the
  native pipeline, ~24–38% higher for the rectified pipeline — worth
  explicitly correcting against the visual impression from the plot, which
  overstates it.
- **Ruled out**: local absorption-line density is flat across the band
  (3–6 line-minima per 5 cm⁻¹ bin everywhere, checked via
  `scipy.signal.argrelextrema` on the true spectrum) — denser line packing
  near the band edge is not the explanation.
- **Present in both pipelines**, proportionally larger in the rectified
  one — consistent with a real edge-of-fit-domain effect (a global order-2
  polynomial dispersion correction has least effective constraint at the
  edges of the band it's fit over) rather than an artifact specific to one
  pipeline's grid construction.
- **Not yet checked**: whether this 4865–4895 cm⁻¹ zone coincides with
  FPA2's independently-documented ground-calibration coverage gap (§9d) —
  plausible given §9d is FPA2-specific and involves degraded round-trip
  accuracy in part of the band, but the gap's exact wavenumber extent
  hasn't been pulled from `keystone_report.pdf` and checked against this
  zone. This is the natural next step before treating the two as related.

### 9o. Dispersion direction alternates FPA to FPA — a second wavelength-ordering bug, this time in the native pipeline

Extending the dense-sweep pipeline to FPA1 (`scripts/gd_band_stress_test.py`,
a new along-slit composition/pressure stress test with its own truth
generator in `geocarb_gert/along_slit_scene.py` -- see `scripts/
gd_along_slit_atm_profiles.py`'s plot for the design; distinct from §11's
two-band *joint* retrieval plan) hit the same *class* of bug as
§9l's rectified-pipeline ordering fix, in a new place: **every** native-grid
retrieval failed catastrophically for FPA1 — including at the slit centre,
where the true along-slit atmosphere equals the retrieval's own prior, a
case that should retrieve almost trivially. Verbose Gauss-Newton tracing
showed the state vector diverging (`albedo_0` collapsing to ~0, `p_scale`
blowing past 3.0) within 2 iterations regardless of row.

Root cause, confirmed by the user (2026-07-24): **wavenumber-vs-column
direction alternates FPA to FPA** — internal reflections/beam splitters in
GeoCarb's real optical path flip the dispersion direction for alternating
bands. Checked explicitly via `xy_to_wavelength_slit` at each FPA's own
representative row:

| FPA | band | wavenumber vs. column |
|---|---|---|
| 0 | O2_A | descending |
| 1 | CO2_weak | **ascending** |
| 2 | CO2_strong | descending |
| 3 | CH4_CO | **ascending** (expected; not yet verified end-to-end, blocked on the §9d/§11 ABSCO extension) |

`gert.ForwardModel` always returns `y`/`y_ret` in ascending-*wavelength*
(descending-wavenumber) order, regardless of `obs_grid`'s input order
(`SpectralWindow.wn_instrument` sorts ascending, then `wl_instrument`
reverses it — see §9l). A native-grid retrieval's `y_dist`, built by
indexing the raw rendered row in column order, only lines up with that
convention when the FPA's own dispersion happens to run descending — true
for FPA0/FPA2, false for FPA1/FPA3. Every native-grid script before this
one (§9h–9n) only ever used FPA2, so this asymmetry was invisible until a
second band was exercised — the exact same shape of blind spot as §9l's
original bug (a lucky accident of direction masking a real ordering
requirement), just in the pipeline that had until now always been the
*reliable* one.

**Fixed** in `gd_band_stress_test.py`'s native-pipeline branch by checking
the direction explicitly (`nu_row[0] < nu_row[-1]`) and reversing both
`y_dist` and `nu_row` together when ascending, rather than assuming either
direction — the general-purpose fix, not a per-FPA special case. Verified:
FPA1 row 512 (slit centre) now converges cleanly (chi2=0.20, co2_scale and
h2o_scale both close to 1 as expected); rows 0 and 960 (arid/humid slit
ends) converge to physically sensible h2o_scale (0.55 arid, 1.86 humid,
tracking the designed along-slit humidity gradient correctly). Documented
as `DISPERSION_ASCENDING` in `geocarb_gert/gd_polynomials.py` so this
doesn't need rediscovering by debugging again, though the runtime check
remains the actual safeguard (self-verifying, not dependent on trusting a
hardcoded table).

**Implication for everything in §9h–9n**: those results are FPA2-only and
unaffected by this bug (FPA2 happens to be on the "lucky" side). But it's a
reminder that *every* single-band script in this study to date implicitly
assumed a convention validated on exactly one of the four bands — worth
keeping in mind before generalizing any §9 conclusion across bands without
re-checking it the way this section just had to.

### 9p. A third pipeline — "undistorted" — added to separate fundamental RT/information-content limits from geometric-distortion artifacts

`gd_band_stress_test.py` gained a third retrieval pipeline alongside native
and rectified: **undistorted**, which evaluates the true hi-res spectrum
directly at a row's intended slit position — no keystone, no smile, no
clocking, no spatial PSF blur across rows, bypassing `gd_render.image()`/
`rectify()` entirely. Its purpose is a control: when native or rectified
show a retrieval difficulty, undistorted answers whether the same difficulty
survives with *zero* geometric distortion, isolating "fundamental RT/
information-content limit" from "something distortion makes worse."

It was added specifically to check a large negative H2O bias found co-located
with the along-slit truth scene's pressure-depression feature in the first
FPA1 dense run (native and rectified both showed it). Running undistorted
reproduced the **same dip, in the same place, at the same magnitude**, with
zero geometric distortion in the loop — confirming the H2O/p_scale
degeneracy is a real retrieval/RT limit for this band's spectral content,
not a keystone/smile artifact. (§9r below revisits this with a cleaner
diagnostic and confirms the same conclusion a second way.)

Because undistorted has no real wavelength-calibration error to correct, its
dispersion order is fixed at 0 rather than the order=2 used for native/
rectified — floating unused dispersion parameters there would only risk
spurious degeneracy with h2o_scale/p_scale, contaminating the exact question
the baseline exists to answer. (§9u below revisits *how* undistorted builds
its observation grid — the version described here used a shared nominal
grid, later found to be a confound in its own right.)

### 9q. FPA0/O2_A: o2_scale floating alongside p_scale caused near-total non-convergence — fixed by excluding well-mixed gases from the retrieved state vector

Extending the along-slit stress test to FPA0 (O2_A, molecules `o2`+`h2o`)
produced a real run that finished without crashing but was, on inspection,
almost entirely useless: **7/1024 (0.7%) converged for native, 1/1024 for
rectified, 0/1024 for undistorted.** The script's own summary line ("diverged:
16, off-detector: 11") looked clean because it never counted the dominant
failure mode — "stalled" (hit `max_iter=14` without `dx_norm` settling below
`dx_tol`), which isn't tracked separately from success in that print.

Individual stalled rows showed the signature of a flat/degenerate cost-
function valley, not real non-convergence: chi2 had already bottomed out
(0.006–0.03, well under 1) while the state vector kept drifting slowly
between iterations, and adjacent rows with nearly identical truth converged
(if forced) to wildly different O2 columns (e.g. rows 0 and 1: +3753 ppm and
−14388 ppm bias respectively).

**Root cause.** `gert.retrieval.StateVector.gas_scaling()` always includes a
`p_scale` element (10% default prior uncertainty) regardless of which gases
are requested — a fact this study hadn't been tracking the retrieved value
of. For FPA0, `gases=['o2','h2o']` meant `o2_scale`, `h2o_scale`, and
`p_scale` were all floating simultaneously. O2 is well-mixed and essentially
fixed in the real atmosphere (`xtrue_of_row["o2"]` in `gd_band_stress_test.py`
is a hardcoded constant, `0.2095` everywhere, by design) — so a well-mixed
gas's column and total air-mass path (what `p_scale` scales) are nearly the
same physical quantity for a fixed-VMR species. Floating both simultaneously
is a near-total degeneracy: the optimizer can trade `o2_scale` against
`p_scale` along a ridge that barely changes chi2, exactly matching the
observed flat-valley symptom.

**Fixed** by introducing `WELL_MIXED_GASES = {"o2", "n2o"}` in
`gd_band_stress_test.py` and excluding these from the retrieved `gases` list
even when present in a band's molecule set — `p_scale` alone now carries the
surface-pressure signal, exactly as an operational O2-A retrieval does (O2's
column is a direct proxy for total air-mass, so there's no independent
information in retrieving it separately). `o2` stays in the forward model's
molecule list (real O2 physics is still computed), it just isn't a free
`_scale` parameter. `p_scale`'s own bias is now tracked explicitly (added
`nl["p_surface"]` to `_retrieve()`'s return, computed unconditionally since
`p_scale` is always in the state vector — useful for every band, not just
FPA0).

**Verified**: FPA0 smoke test after the fix converged 16/16 across all three
pipelines, with native/undistorted `p_surface` bias of a few tenths of an
hPa — essentially perfect recovery. At full 1024-row scale (§9v): native
1024/1024, undistorted 1024/1024, rectified 1007/1024, and the run finished
in 2018s versus the original broken run's 5941s — the earlier run had spent
nearly all its time hitting `max_iter` on every single retrieval; fixing the
degeneracy let most retrievals converge in a handful of iterations instead.

### 9r. h2o_scale's default 10% prior uncertainty is unrealistically tight for this scene — raised to a measured 60%, and FPA1's original H2O finding re-examined

After §9q's fix, FPA0 converged cleanly but still showed a large H2O bias
(hundreds to ~1500 ppm). Checking the actual prior-vs-truth mismatch:
`p_scale`'s default 10% prior (1σ ≈ 96 hPa on the ~961 hPa slit-centre prior)
covers the scene's true pressure deviation (up to −201 hPa, §9q's mountain
feature) at **~2.1σ** — comfortably reachable, and indeed p_scale is
recovered almost exactly. `h2o_scale`'s same 10% default, though, covers a
scene where true h2o deviates from the fixed prior by **up to ±56%**
(`h2o_surface_ratio` range `[0.439, 1.561]` across the slit) — **~5.6σ** away
at a 10% prior width. A MAP estimator under a prior that tight will resist
moving `h2o_scale` anywhere near truth almost regardless of how informative
the spectrum is, independent of any real RT/degeneracy limit.

Tested directly on FPA0 (16-row smoke tests, native pipeline, holding
everything else fixed):

| h2o_scale prior 1σ | H2O bias std (ppm) |
|---|---|
| 10% (old default) | 1163.5 |
| 60% | **615.3** |
| 200% | 903.7 |

60% roughly halves the bias spread relative to the 10% default — real, and
not just "loosen it as far as possible": 200% is *worse* than 60%, not
better, because with too little prior regularization the fit starts drifting
on H2O's genuinely weak (but nonzero) sensitivity in the O2-A band, the same
flavor of problem as §9q's o2_scale/p_scale ridge, just softer. **60% is a
measured sweet spot for this scene, not a guess**, and was adopted as the new
default in `_retrieve()`'s `gas_uncerts={"h2o": 0.60}` — applying generically
to every band, not just FPA0.

This raised a scope question: did FPA1's *already-completed* real run (§9o,
using the old 10% default) need to be rerun? Backed out FPA1's actual
retrieved `h2o_scale` from its stored bias values (native pipeline, all 1024
rows) rather than guessing:

- corr(retrieved `h2o_scale`, TRUE h2o ratio) = **0.960**
- corr(retrieved `h2o_scale`, true `p_surface`) = 0.524 (a real but
  secondary cross-talk)
- RMSE(retrieved − true) = 0.255 (scale units) vs. RMSE(prior=1 − true) =
  0.452 if `h2o_scale` had never moved off prior — retrieval is doing
  roughly **2× better than doing nothing**, not stuck.

FPA1's `h2o_scale` was genuinely tracking real signal, not starved by the
tight prior — CO2_weak has real, usable independent H2O sensitivity, unlike
O2_A. **FPA1's original run did not need a rerun**; its H2O/p_scale
degeneracy finding (§9p) stands as a real, secondary RT effect on top of a
retrieval that was mostly working, not evidence of prior-starvation.
(FPA1 *was* rerun anyway as part of §9v's full 8-run suite, to pick up
§9u's separate undistorted-grid fix — the two issues are independent.)

### 9s. FPA2 added to the along-slit stress test, with a with/without-variation comparison mode

FPA2/CO2_strong had been deliberately excluded from `gd_band_stress_test.py`
runs because it already had dense-sweep results (`gd_dense_sweep.py`, §9m) —
but that scene holds atmospheric composition **fixed** (`reference_atmosphere()`
as both truth and prior, only albedo varies row to row; h2o is floated as a
nuisance parameter there but its bias is deliberately never scored, since
truth always equals prior for it in that scene). It tests a different,
narrower question — pure geometric distortion in isolation — and had never
been run through *this* along-slit, composition-varying scene at all. Added
here for consistent 4-band coverage under the same realistic truth
atmosphere as FPA0/1/3.

This also motivated a genuinely useful new diagnostic: `--uniform`, added to
both `along_slit_scene.build_lookup_radiance()` and `gd_band_stress_test.py`.
`uniform=True` collapses the along-slit lookup table to a single sample at
the slit centre (matching the retrieval's own prior exactly), so every row's
truth spectrum is identical and `xtrue_of_row` is evaluated at x_km=0 for
every row — isolating **pure geometric-distortion bias** from **composition-
tracking bias** by running the identical pipeline with the along-slit
variation switched off. (Output filenames get an automatic `_uniform`
suffix so a uniform run never collides with the real along-slit run.)

Run on FPA2 (16-row smoke test, both modes): rectified's CO2 bias was
essentially unchanged with vs. without composition variation (mean −52.20
vs. −52.66 ppm) — direct, quantitative confirmation that this specific
number is a pure bilinear-interpolation/line-depth-dampening artifact (§9n),
not something composition variability contributes to. Native/undistorted's
H2O bias, by contrast, nearly vanished in uniform mode (std ~610 → ~3 ppm) —
confirming that bias really is about tracking real along-slit composition,
with nothing left to explain once there's no real deviation. Both
conclusions were later confirmed again at full 1024-row scale across all
four bands (§9v).

### 9t. FPA3 unblocked: ABSCO gaps extended, then two further bugs found and fixed (undistorted-grid shape mismatch; GEOCARB_BANDS[3]'s nominal band was offset from the real per-pixel band)

FPA3 (CH4_CO) was hard-blocked all session on ABSCO coverage gaps (§9d/§11):
ch4 and h2o's dense blocks stopped at 4350 cm⁻¹ while FPA3 needed up to
~4360.7; co's block stopped at 4360, similarly short. The user extended
ch4/h2o/co's ABSCO blocks to 4400 cm⁻¹ and, at the same time, padded o2's
block from 12745–13245 to 12745–13300 (giving FPA0 headroom against the
razor-thin 1.99 cm⁻¹ margin found earlier). Verified directly against the
rebuilt `absco.h5` (not just the spec file) via `h5py`: all four gases'
blocks now comfortably cover what each band's `real_wavenumber_range(...,
margin_cm1=10.0)` requires.

With FPA3 finally runnable, its first smoke test **failed 100% on the
undistorted pipeline** (native and rectified were fine) with a shape
mismatch inside `gert.retrieval._chisq`: `operands could not be broadcast
together with shapes (929,) (586,)`. Traced (via a temporary full-traceback
capture in `_retrieve()`'s and `_worker()`'s exception handlers, since neither
had ever needed one before) to `ret.run() → self._chisq(y0, x) → self.y_true
- y`: the forward model was silently returning fewer points than the
observation grid requested, because part of that grid fell outside
`[wn_min, wn_max]` (the actual forward-model-computable range). **Root
cause**: undistorted used the fixed nominal `wn_grid` (from
`build_geocarb_instrument()`) unfiltered, unlike native (which uses each
row's own real, always-in-range positions) or rectified (which drops
out-of-footprint pixels as NaN before retrieving) — neither of those two
pipelines had a structural way to hit this, so the bug was invisible until
undistorted was exercised on a band where nominal and real ranges
genuinely diverge.

Investigating *why* they diverge for FPA3 specifically (nominal
`[4208.00, 4318.00]` vs. real-per-row-envelope `[4248.60, 4360.69]`, only
586/929 nominal points inside) turned up something bigger than a filtering
bug: `GEOCARB_BANDS[3]`'s hardcoded nominal band, `("CH4_CO", 4208.0, 4318.0,
...)` in `geocarb_gert/instrument.py`, was centred at 4263.0 cm⁻¹ — about
**41 cm⁻¹ below** the real per-pixel band's centre (~4304.6 cm⁻¹). That's not
the usual "nominal snugly narrower than real" pattern this study already
knew about and treated as expected (`gd_polynomials.py`'s
`real_wavenumber_range()` docstring cites FPA0 as an example: nominal
`[12950,13190]` vs. real `[12957,13233]`, ~40 cm⁻¹ short on a 240 cm⁻¹ band)
— it was a comparable *absolute* offset on a band only 110 cm⁻¹ wide, i.e.
proportionally about 4× worse, and a genuine *shift* rather than a
conservative narrowing. Independently confirmed against the user's
ground-test wavelength data (`[2.2989, 2.3461] µm` = `[4262.39, 4349.91]
cm⁻¹`), which matched the dispersion polynomial's own per-row output almost
exactly. **Confirmed by the user to be a stale/wrong number, not a real
detector limit, and corrected**: `GEOCARB_BANDS[3]` is now
`(4258.60, 4350.69)` — `real_wavenumber_range(3, margin_cm1=0.0)` exactly —
so the nominal grid now fully contains the real per-row range with zero
margin (770/770 points inside, vs. 586/929 before). Before the fix,
rectified and undistorted had been silently discarding roughly a third of
every row's real spectral content at the high-wavenumber end, for every
single row — not an edge-row effect.

(Minor footnote for the record: `real_wavenumber_range()`'s row sampling
steps by 8 (`range(0, N_PX, 8)`), which skips the literal last row, 1023 —
its true per-row max (4350.77) is about 0.08 cm⁻¹ above the corrected band's
upper bound (4350.69). Negligible next to the 10 cm⁻¹ margin used elsewhere,
and not worth chasing, but noted since it's directly relevant to the
precision of that number.)

### 9u. Undistorted's shared nominal grid was ~38% coarser than native's real per-row sampling — redesigned to match native's grid exactly

A follow-up question (given native's real 1024-pixel-per-row grid and
undistorted's shared nominal grid are built from completely different
sources) turned up a second, independent confound in the native-vs-
undistorted comparison, beyond §9t's range mismatch. For FPA3:
undistorted's grid (770 points post-§9t-fix, spanning 92.09 cm⁻¹) samples at
**8.36 points/cm⁻¹**; native's real per-row grid (1024 points spanning
~88.6 cm⁻¹ at row 512) samples at **11.56 points/cm⁻¹** — native has ~38%
denser sampling. The nominal grid's density comes from `channels_per_fwhm=3`
in `build_geocarb_instrument()` (3 samples per ILS FWHM, a resolution-driven
convention), completely independent of the real detector's actual pixel
pitch, which happens to oversample its own ILS more densely than that
convention gives.

This matters because it's a *third*, previously uncounted contributor to
native's consistently lower chi2 relative to undistorted (noted informally
earlier this session): on top of (1) native having real distortion for its
extra order=2 dispersion parameters to legitimately absorb, and (2) those
same parameters risking absorbing residual structure they shouldn't, native
also simply has more independent spectral samples per row to fit against —
real extra constraining power, unrelated to distortion or to state-vector
freedom.

**Fixed** by having undistorted evaluate the truth spectrum at each row's
own real native positions — literally the same `xy_to_wavelength_slit`-
derived `nu_row` (and the same ascending/descending-wavenumber direction
fix-up, §9o) that native uses — instead of the separate shared nominal grid.
This also **simplified** the pipeline: §9t's range-filter patch (restricting
`wn_grid` to `[wn_min, wn_max]`) became unnecessary and was removed, since a
row's own real positions are in-range by construction, the same reason
native never needed such a filter. Native and undistorted now share an
identical grid per row; the only remaining difference between them is
exactly what the baseline is meant to isolate — whether the spectrum itself
carries real keystone/smile/PSF distortion, not also how densely or over
what range it's sampled. Re-validated on both FPA3 and FPA1 smoke tests
after the change (comparable convergence/divergence counts to before);
folded into the full-scale run in §9v.

### 9v. Full 4-band along-slit stress test at scale (1024 rows × 3 pipelines × 4 FPAs × with/without variation, 8 real SLURM runs)

With §9q–9u's fixes in place, `scripts/gd_band_stress_test.slurm` was
updated to submit all four bands (`--array=0,1,2,3`, previously `0,1` only)
plus an opt-in `UNIFORM=1` env var for the §9s comparison mode (`UNIFORM=1
sbatch --array=<N> ...`), and run by the user for all 4 FPAs × both modes —
8 real 1024-row jobs total. Diagnostic plots (`scripts/
gd_band_stress_test_plot.py`) for the four primary with-variation runs:
`plots/gd_band_stress_test_fpa0.png`, `_fpa1.png`, `_fpa2.png`, `_fpa3.png` —
each a 6-panel figure (primary-gas bias, H2O bias, and chi2 vs. along-slit
position; the truth profile for context; a per-pipeline failure-location
map; and H2O bias vs. true surface pressure, the degeneracy plot referenced
throughout this section). The `--uniform` runs that the with/without
comparison below is built from were analyzed directly from their `.pkl`
output rather than plotted — `gd_band_stress_test_plot.py` doesn't yet have
a paired with/without-variation view (the closest precedent is the smoke-
scale `plots/gd_band_stress_test_fpa1_smoketest.png` / `_fpa0_smoketest2.png`
from earlier validation, §9q/§9r, not the full-scale runs this table
covers).

**Convergence, all four bands, native/undistorted:**

| FPA | pipeline | converged | diverged | stalled | off-detector |
|---|---|---|---|---|---|
| 0 | native | 1024/1024 | 0 | 0 | 0 |
| 0 | rectified | 1007/1024 | 0 | 6 | 11 |
| 0 | undistorted | 1024/1024 | 0 | 0 | 0 |
| 1 | native | 1024/1024 | 0 | 0 | 0 |
| 1 | rectified | 689/1024 | 154 | 160 | 21 |
| 1 | undistorted | 1024/1024 | 0 | 0 | 0 |
| 2 | native | 1024/1024 | 0 | 0 | 0 |
| 2 | rectified | 914/1024 | 77 | 16 | 17 |
| 2 | undistorted | 1024/1024 | 0 | 0 | 0 |
| 3 | native | 1021/1024 | 0 | 3 | 0 |
| 3 | rectified | 966/1024 | 41 | 4 | 13 |
| 3 | undistorted | 1019/1024 | 0 | 5 | 0 |

Native and undistorted are essentially fully converged everywhere. Rectified
consistently shows the worst failure rate of the three (10-33% not usable),
across every band — consistent with §9n/§9t's interpolation-error findings,
not an artifact of any one band's fixes.

**The with/without-variation comparison (§9s), now confirmed at full scale,
splits the rectified-bias story into two distinct mechanisms:**

For FPA1/FPA2/FPA3, rectified's primary-gas bias mean is essentially
*identical* with real composition variation on vs. off — FPA1 CO2: −108.89
vs. −108.64 ppm; FPA2 CO2: −52.08 vs. −52.57 ppm; FPA3 CH4: −86.77 vs.
−83.86 ppb. **A pure geometric/bilinear-interpolation artifact**, confirmed
now across three separate bands at full scale, not just FPA2's smoke test.

**FPA0 breaks that pattern.** Rectified's `p_surface` bias standard
deviation is 5.25 hPa with real composition variation vs. 0.35 hPa in
uniform mode — a >10× difference, and `plots/gd_band_stress_test_fpa0.png`
(top-left panel) shows a sharp ±15 hPa excursion localized exactly at the
along-slit truth scene's pressure depression. Unlike the other three bands'
primary-gas rectified bias,
FPA0's is not a fixed geometric artifact independent of the true state — it
is the interpolation error interacting with the true pressure gradient
itself. This makes physical sense given O2-A senses pressure directly
through line broadening (§9q), a mechanism the other three bands' primary
gases don't share.

**The H2O/p_scale degeneracy (§9p, §9r) is universal and strikingly
consistent in shape** across native/undistorted for FPA1, FPA2, and FPA3 —
essentially the same H2O-bias-vs-true-p_surface "loop," swinging roughly
−1200 to +900 ppm regardless of band. FPA0 shows a related but distinctly
larger-amplitude version (up to +1700 ppm) — expected, since FPA0 retrieves
H2O directly alongside `p_scale` rather than through cross-talk with a
differently-sensed primary gas.

**Resolved: the FPA2 scatter anomaly was a single mislabeled row, not a real
effect.** FPA2's rectified CO2 bias appeared to have *higher* row-to-row
scatter in uniform mode (std=10.65) than with real along-slit variation
(std=3.62) — backwards from the naive expectation that removing real signal
should reduce structure, not add it. Traced to one row (uniform mode, row
59, x=−1242 km, at the edge of a cluster of otherwise-correctly-flagged
divergent neighbor rows 54–64): chi2 = 1.65×10²⁰³, co2 bias = −357 ppm,
h2o bias = 6006 ppm — an exploded, unphysical state that nonetheless
reported `converged=True`. Same class of bug §9m already fixed once
(`dx_norm` measures step size, not chi2 validity, so a state that overshoots
into something unphysical can still register a small step and thus
"converged"), but at a boundary that fix didn't cover: chi2 here is
enormous but *technically finite*, so it passes `gert`'s existing
`np.isfinite(chisq_red)` divergence check untouched. Scanned all 8 real
runs × 3 pipelines (~73,000 row/pipeline combinations): exactly one
occurrence — rare, but real, and silent (it single-handedly inflated one
pipeline's reported bias std ~3×). Excluding just that row drops the
uniform-mode std to 3.44, matching the with-variation run almost exactly —
confirming there was never a real with/without difference here.

Rather than widen `gert`'s own divergence check (shared, foundational
code), this is now handled in the analysis layer: `gd_band_stress_test_plot.py`
gained `_chi2_outlier_mask()` (a robust, MAD-based-in-log10-space outlier
test, applied per pipeline's own converged population — no fixed magic
threshold) and `_robust_stats()` (median + 1.4826×MAD, a normal-equivalent
scale estimate that isn't dragged around by a single extreme row the way
mean/std are). Every plot panel and the new printed summary table (median,
robust_std, mean, std, and outlier count side by side, per pipeline/gas)
now excludes chi2-outlier rows, and the failure-location panel gained a
`*` marker counting them explicitly as their own failure category rather
than silently folding them into "converged." All 8 real-run plots and
summaries were regenerated with this in place; FPA2-uniform's rectified row
is now the only nonzero `n_outlier` anywhere in the dataset.

**Runtime, confirming §9q's convergence fix translates to real wall-clock
savings**: FPA0's full run finished in 2018s (32 cores) — down from the
original broken run's 5941s for the same row count, roughly 3× faster,
consistent with most retrievals now converging in a handful of iterations
instead of always exhausting `max_iter=14`.

---

### 9w. Row-crossing ("chunking") at the slit extremes: real, but not the dominant driver everywhere

**The concern.** Keystone distortion means a single detector row's own
dispersion trace is not a horizontal line across the FPA — near the top and
bottom of the slit it cuts diagonally across as many as ~10 physical rows'
worth of along-slit position. Native's forward model, however, always
evaluates every column of a row against one fixed atmosphere (the
slit-centre prior/truth state at that row's *nominal* position — see
`_retrieve()`'s docstring). If the real along-slit scene varies on scales
comparable to that smeared footprint, native's spectrum at an edge row is
really a blend of several distinct true states that the retrieval has no
mechanism to represent, and the concern was that this blindness should
produce bias far larger than the "a few %" found so far.

**Quantifying the smear.** Using `rows_crossed(fpa, row)` (physical rows
spanned by one row's own dispersion trace) confirms the "~10 rows" claim
for every band:

```
FPA0: row0=10.42  null≈row600 (val≈0)  row1023=7.33   max=10.42
FPA1: row0=9.18   null≈row570 (val≈0)  row1023=7.34   max=9.18
FPA2: row0=0.27   null≈row25  (val≈0)  row1023=10.41  max=10.41
FPA3: row0=2.99   null≈row266 (val≈0)  row1023=7.86   max=7.86
```

Converting to real along-slit km smeared into one row's 1024 columns
(`xy_to_wavelength_slit` → x_km), at the worst rows this is a genuine
20–29 km footprint (vs. ~2–4 km at each band's own null row) — a real,
non-negligible chunk of the along-slit scene's own ~10s-of-km-scale
structure. Critically, **the null row (near-zero row-crossing) is not
always at the slit centre**: FPA0/FPA1 have it mid-slit (row ≈570–600,
matching their smile-null row per §9i's finding that these need not
coincide), but FPA2's is near the top (row 25) and FPA3's is off-centre
(row 266) — so a naive "edge vs. centre" split silently mixes a
low-crossing extreme with a high-crossing one for those two bands.

**Does it show up in the actual measured bias?** Correlating each in-family
native row's `rows_crossed` value against its measured `|gas bias %|` and
`log10(chi2)` across the full 1024-row sweep (chi2-outlier rows excluded,
same filter as §9v):

| FPA | gas | corr(rows_crossed, \|bias%\|) | corr(rows_crossed, log10 chi2) |
|---|---|---|---|
| 0 | p_surface | −0.09 | +0.38 |
| 1 | co2 | **+0.63** | +0.56 |
| 2 | co2 | +0.28 | +0.17 |
| 3 | ch4 | −0.20 | −0.11 |

**Verdict: the effect is real, and it's the dominant driver for FPA1 (CO2
weak)** — row-crossing explains a substantial share of the along-slit bias
variance there, confirming the intuition directly: FPA1's edge rows
(|x|>1300 km) show −2.15%±0.49% native CO2 bias vs. −0.08%±0.25% at its
low-crossing centre, a ~30× jump in mean bias exactly where crossing is
highest. FPA2 shows the same sign of effect but weaker (+0.28), consistent
with its own along-slit CO2 gradient being gentler than FPA1's in the
region where crossing peaks. **FPA0 and FPA3 do not show this pattern** —
their bias is governed by something else (FPA0: p_surface is retrieved via
`p_scale` off a well-mixed-gas assumption, §9q, with near-zero overall
bias regardless of position; FPA3: CH4/CO's own absorption-strength
along-slit gradient and prior/noise structure dominate over the
row-crossing term). So the answer to "can you see chunking in the rows at
the extremes" is: **yes, unambiguously for FPA1, partially for FPA2, and
not detectably for FPA0/FPA3** — it is a real, previously-uncharacterized
contributor to native's bias, but it is band-dependent, not a universal
explanation for why "native" biases stay in the low-percent range. The
overall bias magnitudes reported in §9v were not wrong, but for FPA1 in
particular, part of the along-slit bias curve's shape is now understood to
be a direct signature of keystone row-crossing rather than pure
interpolation/geometric-distortion error — this is a distinct mechanism
from the rectification-interpolation bias in §9l/§9m, additive to it, and
specific to native (rectified/undistorted don't have this failure mode by
construction, since rectified resamples onto a shared grid and undistorted
has no distortion to cross rows with).

---

### 9x. Why native's chi2 looks so much smaller than undistorted's -- a dispersion-order artifact, not a fit-quality difference

**The question.** Every realistic-scene figure in §9v/§9w shows native's
chi2_reduced running 1-2 orders of magnitude below undistorted's, at
essentially every row (e.g. FPA1: native ~1e-3-1e-2, undistorted
~0.15-0.3). Taken at face value this looks like native is simply the
better fit -- counterintuitive, since undistorted is the zero-distortion
baseline and might be expected to fit at least as well as native, not
dramatically worse.

**Isolating the cause.** Reran `_retrieve()` by hand on FPA1 at several
rows, crossing both data sources against both dispersion orders (native/
undistorted's own order choice, order=2 vs order=0, alongside the
opposite order each doesn't normally get):

| row | native, order=2 (actual) | undistorted, order=0 (actual) | native, order=0 | undistorted, order=2 |
|---|---|---|---|---|
| 0 | 0.0028 | 0.163 | 0.653 | 0.0010 |
| 512 | 0.0003 | 0.161 | 0.162 | 0.0001 |
| 1023 | 0.0035 | 0.177 | 0.470 | 0.0028 |

At the orders each pipeline actually runs with, the gap is the ~50-500x
seen in every figure. **Matched at the same order, the gap nearly
vanishes** -- undistorted at order=0 tracks native at order=0 almost
exactly (and is slightly *better* at slit centre and mid-slit, consistent
with it being the cleaner baseline); undistorted at order=2 lands right on
top of native. So this was never a difference in how well each pipeline's
underlying data can be fit -- it's that native (and rectified) are allowed
to float a low-order wavelength-dispersion correction
(`disp_a0`/`disp_a1`/`disp_a2`) that undistorted is deliberately denied
(§9k/module docstring: undistorted has no genuine wavelength-calibration
error, so floating dispersion there risks spurious degeneracy instead of
correcting anything real).

**What that correction is actually absorbing.** Native's fitted `disp_a0`
(constant wavelength shift) at FPA1:

| row | disp_a0 | posterior significance |
|---|---|---|
| 0 | -0.0048 cm-1 | ~18 sigma |
| 512 | +0.0000026 cm-1 | ~0 sigma |
| 1023 | -0.0038 cm-1 | ~15 sigma |

A real, highly-significant, but physically tiny shift (thousandths of a
cm-1, far below one resolution element) that is essentially zero at slit
centre and grows toward both edges -- tracking the same row-crossing
pattern from §9w: near-zero at the low-crossing row, largest where
keystone mixes the most true along-slit positions into one row. The
dispersion term is finding and absorbing a small, real, keystone-
correlated effective wavelength-registration error, exactly the kind of
residual calibration nudge it exists to fit.

**Conclusion.** Native's much lower chi2 is a direct, designed consequence
of which pipelines get a wavelength-calibration escape valve, not evidence
that native's fit is fundamentally better or that undistorted's data is
somehow worse. Denying undistorted that same nuisance parameter is
intentional -- it's what keeps the zero-distortion baseline honest about
how much residual a genuinely well-behaved detector would still leave,
rather than letting it silently absorb the very geometric-distortion
signature it exists to isolate. (Rectified gets the same order=2 freedom
as native and still stays worst of all three, per §9v/§9w -- its own
residual comes from real interpolation corruption, §9l/§9m, which isn't
the smooth, low-order shape a dispersion polynomial can absorb.) No code
changed as a result of this investigation -- it's a read on existing
behavior, not a bug.

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

### 11b. Cross-band ground co-registration — resolved: shared physical reference confirmed

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
effect §9c characterized for FPA2 alone, now shown to also misalign band to
band — but see below: this disagreement at a fixed *pixel* is expected and
is not, by itself, evidence against a shared reference at a fixed *angle*.

**Never align bands by row index; the shared coordinate is real slit
angle `s`.** That much was already clear. What was genuinely open was
whether `s` itself means the same physical direction across all four FPAs'
independently-fit polynomials — resolved below.

**Instrument optical architecture** (from the instrument team, 2026-07-27).
All four bands share one physical slit. Just inside the slit, the light
splits into two arms — one shortwave, one longwave — each with its own
optical path, until a final beam splitter *within* each arm separates that
arm's own two bands onto their own two detectors. So the shared front end
(telescope + slit) is common to all four bands; each arm's relay/grating
optics are common to only two of the four; each detector is unique to one
band. The instrument team's own working assumption is that the smile/
keystone *differences* between bands come from each individual detector's
own mounting (decenter/tilt/rotation in its focal plane), not from
wavelength-dependent aberration in the shared telescope, slit, or either
arm's relay optics.

**Build-time alignment.** Everything upstream of the detectors — telescope,
slit, both arm splits, both arms' relay optics — was verified at
integration using a chief-ray ("gut ray") alignment strategy: a laser at a
representative wavelength for *each* band was used to confirm that light
from slit centre lands on that band's own optical boresight. This was done
independently per band, in both arms, not once for the whole instrument —
so there is no reason to expect an *arm-level* boresight difference between
a same-arm band pair and a cross-arm pair (e.g. O2-A + CO2_strong, §11's
proposed test pair, likely sit in different arms given their wavelengths).
Combined with the detector-only misalignment assumption above, this pins
any real cross-band registration error to a single, well-defined degree of
freedom per detector — a rigid offset (plus possibly a small rotation/scale
if that detector is also tipped), not a distributed or wavelength-dependent
effect spread across the shared optics.

**Ground-test calibration methodology closes the loop.** The build alignment
above is a design/integration-time guarantee; the actual GD polynomials in
`gcmap_em27.csv` come from a separate ground-test data-collection step (the
"Laser-spot GD test," calibration-chain bullet 1 above), and it was that
step — not the build alignment — that could in principle have reintroduced
a per-FPA reference ambiguity if it had been done independently per band.
It wasn't: the calibration lasers spanning all four bands' wavelengths were
co-aligned onto one common optical path focused on the slit, so at every
lattice point they illuminated the *same physical spot* on the calibration
scanning mirror simultaneously. Slit position was swept by steering that
one shared beam with the mirror; wavelength was swept separately by tuning
laser frequency. **Conclusion: every one of the 55 calibration lattice
points has exactly one true physical slit angle common to all four FPAs at
once, and each FPA's polynomial was fit against that same shared stimulus.**
`s = 0.03°` in FPA0's polynomial and `s = 0.03°` in FPA2's polynomial mean
the same physical direction, by construction of the calibration data itself
— not by an assumption this study had to make. The FPA0-vs-FPA2 disagreement
at a fixed *pixel* shown above is exactly what real, independent per-detector
clocking differences at a shared true angle should look like, and is no
longer evidence of an unresolved reference problem.

**One narrower residual question, not yet checked.** The coefficients this
codebase actually uses (`gcmap_em27.csv`) are not the raw laser-lattice fit
— they are that fit *refined* by a separate EM27/SUN sun-viewing step
(calibration-chain bullet 3, §4.8.6), a per-FPA Nelder-Mead optimization
against each FPA's own measured solar spectrogram. If that refinement's
solar observations were taken through the same shared telescope/slit (the
instrument's normal viewing path, and the same underlying reason the laser
test preserved a shared reference), it should inherit the same cross-band
consistency for the same reason. But that hasn't been confirmed the way the
laser-test mechanism now has — worth a direct check of the EM27/SUN
refinement's own observation setup before treating the *refined*
coefficients' cross-band consistency as fully closed, as opposed to just
the initial fit's.

**Practical effect:** shared-`s_grid` co-registration (§11c) no longer needs
to carry a mandatory, unconstrained inter-band pointing-offset nuisance
parameter as insurance against an unknown-shape reference mismatch — the
mechanism above rules out anything but a small, specifically-shaped
(rigid, per-detector) residual. Keeping that offset as a free but
*expected-to-be-near-zero* parameter in the first joint-retrieval test is
still worthwhile: cheap, and a direct empirical check on both the residual
build-alignment tolerance and the still-open EM27/SUN question above.

### 11c. Co-registration mechanism: risks and alternatives to plain rectification

With §11b's reference question resolved, the remaining design choice is
*how* to combine each band's real per-pixel data at a shared slit angle —
and the obvious first answer (rectify each band onto a shared `s_grid`)
carries a real, already-quantified risk.

**Rectification risk.** The mechanism §11b originally proposed — retarget
`gd_render.rectify()` at a grid shared across bands instead of each band's
own `linspace(-s_max, s_max, ...)` — is the *same* interpolation-based
resampling already shown, repeatedly (§9l/§9m, and every "rectified" curve
in §9v/§9w's figures), to be the single largest bias source found in this
whole study, independent of and on top of geometric distortion itself.
Reusing it for cross-band co-registration would run that same interpolation
error twice (once per band) plus whatever residual cross-band mismatch
remains post-§11b, right at the step meant to make the two bands
comparable. `s_max` and each band's off-detector edge rows also differ
slightly per FPA (checked: `s_max` = 2.2899° / 2.2820° / 2.2970° / 2.2858°
for FPA0-3; §9l found ~17 rows off-detector near FPA2's slit ends), so the
joint grid would also need clipping to the cross-band intersection, on top
of the interpolation-bias problem.

**Alternative 1 — nearest native row.** For a target shared slit angle, look
up each band's own nearest *real* row (no interpolation of the spectrum
itself) and retrieve both bands' rows jointly (shared aerosol/atmosphere
state, stacked into one cost function), each on its own real per-pixel
`obs_grid` — exactly how native already operates today, just paired across
bands instead of resampled. The only new error is row-quantization: the two
bands' nearest real rows sit at most half a row's spacing apart in true
angle, a small, bounded discretization error rather than an interpolation
blend across whatever scene structure falls inside an arbitrary grid cell.
Coverage is irregular (whatever discrete angles happen to be close between
the two bands) rather than a clean shared axis, which is likely an
acceptable tradeoff for a retrieval.

**Alternative 2 — PSF-area-weighted average.** Rather than picking one
nearest row, combine the few real rows nearest the target angle, weighted
by the real along-slit PSF — physically motivated, since a real detector
row is already the true scene convolved once with that same PSF
(`spatial_psf_fwhm_px=1.5`, §1/§9h — the one place cross-row mixing is
physically correct rather than an artifact). Confined to combining *rows*
only, with each contributing row's own real wavelength grid left untouched,
this avoids rectification's core problem (a geometric kernel with no
physical basis mixing across wavelength *and* position at once). The one
subtlety to get right: each real row is already the result of one PSF
convolution, so re-combining several already-blurred rows with a second
copy of the same kernel *compounds* the blur rather than just re-weighting
it (two convolutions of width σ produce an effective width of σ√2, not σ) —
a quiet, easy-to-miss resolution loss. Two ways to avoid it: combine each
row's *pre-blur* per-pixel spectrum (reaching one step earlier into
rendering, before the single PSF convolution is applied) rather than the
already-rendered rows; or use a deliberately narrower weighting kernel whose
quadrature sum with the existing blur reproduces the true PSF width. The
former is more invasive but safer — the same "prefer the real, unresampled
thing" instinct that made native beat rectified everywhere else in this
study.

**PSF commonality across bands.** Both alternatives above are simpler if
the two bands in a pair share one PSF kernel rather than needing two
separately-calibrated ones. §1 already records ground test reporting the
same ~1.5 px FWHM for all four FPAs — a single shared number, not four
independently-measured ones that happen to agree — which is stronger
evidence than "the telescope should dominate" would be on its own, since it
also rules out the along-slit PSF being diffraction-limited (a
diffraction-limited system would scale with wavelength, roughly 3× wider
for FPA3 at 2.3 µm than FPA0 at 0.76 µm). Consistent with the shared
telescope/slit architecture in §11b: the along-slit PSF is set mainly by
the common front end, not by each arm's own downstream optics. The residual
caveat is unit conversion, not physics — "same PSF" is exact in angular
terms and only approximately so in pixels, since converting to a pixel
FWHM needs each band's own plate scale. Given all four FPAs share the
1024-row format and `s_max` values within ~0.6% of each other, this is a
small, already-bounded correction, not a live risk.

### 11d. Phasing

1. ~~Validate FPA0 individually first~~ **Done (§11e, 2026-07-27) — FPA0
   passes.** No FPA2-style anomaly found; one genuine but non-blocking
   methodological gap surfaced (round-trip self-consistency, affects all
   four FPAs equally, not an FPA0-specific issue). See §11e for the full
   battery and numbers.
2. ~~Resolve the shared-reference-frame question~~ **Done (§11b,
   2026-07-27)** — confirmed by the ground-test calibration methodology
   (co-aligned lasers, shared scanning-mirror point) rather than assumed.
   Only remaining piece: confirm the EM27/SUN refinement step's observation
   setup preserves the same property (§11b's residual question) — worth a
   quick check, not blocking.
3. ~~Pick a co-registration mechanism~~ **Nearest-native-row: built and
   validated (§11f, 2026-07-27)** — `geocarb_gert.cross_band.
   nearest_row_pairing()`. PSF-area-weighted combination (§11c) remains
   available as a later refinement if nearest-row's discretization error
   turns out to matter in practice; it hasn't been built.
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
   native-grid (paired via §11c's mechanism) and, if built for comparison,
   the rectified pipeline. Does adding O2-A actually reduce
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

### 11e. FPA0 individual validation results (2026-07-27) — passes

The phase-1 battery from §11d item 1, run directly against the real
coefficients (polynomial checks) and the existing FPA0 uniform-scene sweep
result (`results/gd_band_stress_test_fpa0_uniform.pkl`, already produced by
§9v's original run — no new render/retrieve needed for this part).

**Keystone-null row: confirmed at row 600.0** (`rows_crossed` minimised
there, value 0.0054) — matches §9f's earlier finding (599.7) and the
report's stated 600.

**Smile-slope-null row: ~550, distinct from the keystone-null row.**
Checked at the same five columns §9i used for FPA2 (4, 256, 512, 768,
1019): minimum |∂ν/∂row| lands at row 548–556 depending on column, a
~8-row spread across the full spectral range — wider than FPA2's <1-row
spread, but still a single well-defined, well-separated minimum, ~50 rows
from the keystone-null row. Confirms §9i's keystone/smile-slope
orthogonality finding isn't FPA2-specific.

**Smile amplitude: ~42–55 px across the band**, band-centre ~48 px,
growing toward the long-wavelength edge (short edge 42.2 px → centre
47.8 px → long edge 55.5 px) — same qualitative growth-toward-long-
wavelength-edge shape already reported in §9f, broadly consistent with
(not pixel-identical to) that section's figures — the small numerical
difference is most likely a methodology detail (exact wavelength/`s`-range
endpoints chosen), not a new finding.

**Uniform-scene rectify→retrieve bias check: FPA0 matches the established
cross-band pattern exactly, no FPA2-style anomaly.**

| pipeline (order) | converged in-family | bias (p_surface / h2o) | chi2 median |
|---|---|---|---|
| native (2) | 1024/1024 | ±0.015 hPa std / ±35 ppm std | 0.00055 |
| undistorted (0) | 1024/1024 | ±0.13 hPa std / ±104 ppm std | 0.078 |
| rectified (2) | 1010/1024 | ±0.35 hPa std / ±486 ppm std | 3.02 |

Native and undistorted both converge cleanly at every one of 1024 rows,
with small biases and native's chi2 ~140× lower than undistorted's — the
same dispersion-order artifact already explained in §9x, not a new effect.
Rectified's 14 non-converging rows split cleanly into two unremarkable
categories: 11 off-detector rows sitting exactly at the two slit extremes
(rows 0–9 and row 1023 — the same edge-clipping behaviour already seen for
FPA2, §9l), and 3 scattered mid-slit stalls (rows 334/750/818, no spatial
clustering). Total failure rate ~1.4%, well below FPA2's known ~11–14% on
the harder dense-sweep test (§9m) — no sign of an FPA2-style localized
calibration-coverage gap (§9d): no bias spike or failure cluster anywhere
in particular, just the expected edge effect plus ordinary optimizer noise.

**One genuine gap surfaced, but it's methodological and not FPA0-specific:
round-trip self-consistency could not be independently reproduced.**
Computing `xy_to_wavelength_slit` → `wavelength_slit_to_xy` round-trip
residual on a generic pixel grid gives residuals of order 0.1–2 px,
scattered non-monotonically across the detector (e.g. FPA0 at dead centre,
pixel (512,512): 0.96 px) — one to two orders of magnitude larger than the
report's cited "few hundredths of a pixel," and **not** concentrated at
the edges the way a simple extrapolation story would predict. Checked
across all four FPAs for context, since a discrepancy isolated to FPA0
would be a real concern: it isn't — FPA1 (0.05 px), FPA2 (0.83 px), and
FPA3 (0.37 px) show the same order-of-magnitude mismatch at their own
centre pixels, in an order that doesn't even track the report's own
FPA-ranking (FPA2 is supposed to be the outlier, worst by far; here it
isn't uniquely so). The most likely explanation: the report's figure was
evaluated at or very near the actual 55 sparse calibration control points
(or the ~1M-point EM27/SUN reference, §9g), neither of which this codebase
has access to — a 4th-degree, 15-coefficient polynomial fit to only 55
points per FPA has real room to wobble *between* those constraints even
while matching them closely. This is a limit on what's independently
verifiable from the coefficients alone, affecting every FPA equally, and
it does **not** touch §11b's cross-band conclusion — that rested on how
the calibration lattice was physically *acquired* (co-aligned lasers,
shared scanning-mirror point), not on any single FPA's own forward/inverse
polynomial-pair self-consistency, which is what this check measures.

**Conclusion: FPA0 is validated for the multi-band plan to proceed.**
Nothing here resembles FPA2's real, independently-documented calibration
gap (§9d) — the round-trip discrepancy is a shared measurement-methodology
limit across all four bands, not a defect discovered in FPA0.

### 11f. Nearest-native-row co-registration: built and validated (2026-07-27)

`geocarb_gert.cross_band` (new module): `real_s_of_row(fpa, rows)` (each
row's real slit angle, the same convention `x_km_of_row` already uses
elsewhere) and `nearest_row_pairing(fpa_a, fpa_b, rows_a)` — for each row of
one band, the real row of the other band closest in true `s`, no
interpolation of either spectrum, rows outside the other band's covered
range dropped rather than extrapolated. Keyed on real `s` throughout, never
normalized `eta` (§11b, and directly re-quantified for this exact pair when
asked, 2026-07-27, below).

**Smoke-tested on the plan's proposed pair, FPA0+FPA2:**

| check | result |
|---|---|
| coverage | 1007/1024 rows kept (17 dropped, outside FPA2's range — same magnitude as the ~11–17-row edge coverage loss already seen for both bands individually, §9l/§11e) |
| mismatch bound | mean 0.262 rows, **max 0.500 rows** — bounded, as it must be for nearest-neighbour matching against a ~1024-point target grid |
| position dependence | no systematic edge growth — 0.31/0.04/0.43 rows at the two edges and centre respectively; jagged, not monotonic, consistent with a discretization "beat" between two almost-but-not-quite-matched row grids rather than any real physical trend |
| monotonicity | `rows_a` vs. matched `rows_b`: correlation 1.0000 |
| reciprocity | FPA2→FPA0 gives the same magnitude (mean 0.262, max 0.502 rows) and similar coverage (1012/1024) — symmetric, no directional bug |

**This also gives the eta-vs-real-`s` question from earlier in the
conversation a concrete before/after number, not just the general argument.**
Using normalized `eta` instead for this same FPA0+FPA2 pair would have
introduced a *systematic*, edge-growing mismatch up to **1.58 rows** at the
slit edge (computed directly, previous turn) — roughly **3× worse than
nearest-row-in-real-`s`'s worst case (0.50 rows), and unlike nearest-row's
error, not bounded**: it keeps growing for any pair with a larger `s_max`
gap (up to 3.35 rows for the worst pair, FPA1+FPA2). Real `s` isn't just
the more principled choice — it measurably dominates the alternative here.

**Status:** phases 1–3 of §11d are done (FPA0 validated, §11e; reference
frame resolved, §11b; co-registration mechanism built and validated, this
section).

### 11g. First joint retrieval result: FPA0+FPA2, no aerosol, order=0, uniform scene (2026-07-27)

**Deliberately skipped ahead of §11d item 4 (aerosol).** Aerosol's own
nonlinearity would compound with anything found here and costs much more
compute — decided to see what row-pairing plus a shared gas/pressure state
does *on its own* first, uniform scene (Sec. 9's null-test convention),
before adding it.

**Built:** `scripts/gd_joint_band_test.py` (new). Renders both bands' native
detector images independently (same real projection-operator + ILS
mechanism as every single-band result in this study), pairs rows via
§11f's `nearest_row_pairing`, then retrieves both bands' native spectra
*jointly* against one shared state vector — a single `p_scale` and
`h2o_scale` common to both bands, `co2_scale` sensitive only through FPA2.
(Confirmed by reading `gert`'s own multi-window Jacobian assembly before
writing this: a gas absent from a window's molecule list gets a correctly
zeroed column block there automatically — no special-casing needed.
`gert.osse.py`/`gert_demo.ipynb` is the only prior working multi-window
precedent anywhere in either codebase; nothing in `geocarb_simulator` had
ever exercised a real 2+-window joint retrieval before this.) Dispersion
order 0 for both bands, to keep the new state vector's plumbing as simple
as possible to validate first.

**Full 1024-row uniform-scene sweep: 1007/1007 paired rows converged
(100%).**

| quantity | joint (FPA0+FPA2) | single-band FPA2 alone | single-band FPA0 alone |
|---|---|---|---|
| co2 bias [ppm], order=0 | mean −0.18, **std 0.39** | mean −0.32, std **1.71** | — |
| p_surface bias [hPa], order=0 | mean +0.063, std 0.149 | — | mean +0.002, std **0.130** |

(order=0 "undistorted" used for both single-band comparisons — the
order-matched baseline, not native's order=2, which benefits from the
dispersion-order chi2 artifact in §9x and isn't a fair comparison here.)

**CO2 improves substantially; surface pressure itself doesn't.** CO2's bias
scatter drops **~4.4×** (std 1.71 → 0.39 ppm) when the fit shares a common
pressure/H2O state with O2-A instead of retrieving CO2_strong alone — the
expected degeneracy-breaking mechanism (a real pressure constraint loosens
the `co2_scale`/`p_scale` tie inside the CO2 band's own fit), showing up
for the first time in this study on real per-pixel synthetic data rather
than as a stated hypothesis. Surface pressure itself does *not* improve the
same way: joint's std (0.149 hPa) is worse than FPA0 alone (0.130 hPa), and
the mean bias moves further from zero (+0.063 vs +0.002 hPa). Plausible
reading: FPA0 alone already constrains pressure very tightly, so folding
CO2_strong's own weaker, noisier pressure sensitivity into the *same*
shared parameter pulls the combined estimate slightly off O2-A's clean
value, even as CO2 benefits from borrowing it. A real asymmetric win, not a
symmetric one — worth remembering when deciding how a production algorithm
should weight the two bands' pressure information (an O2-A-dominated
pressure prior, rather than one equally-shared parameter, might get CO2's
improvement without pressure's small cost) rather than assuming joint
retrieval helps everything it touches equally.

**Caveats before reading too much into this:** one band pair, uniform
scene, no noise, no aerosol, dispersion order 0. The improvement could
shrink, grow, or change character with dispersion floated per band, with
real along-slit composition variation, with noise added, or with aerosol's
own nonlinear coupling to photon path length (still deliberately deferred).
A first, genuinely encouraging data point, not a final result.

**Status:** §11d item 5 (build the joint retrieval, uniform scene first) is
done for the no-aerosol case. Natural next steps, in rough order of
cheapness: (a) repeat with dispersion order=2 per band, to check whether
CO2's improvement survives once each band also gets its own
wavelength-calibration nuisance term; (b) repeat on the realistic scene now
that the uniform null test looks clean; (c) the aerosol extension (§11d
item 4/6) remains deliberately deferred.

### 11h. Full joint battery built: uniform/barcode/realistic x noise/no-noise x native/rectified/undistorted (2026-07-27)

**The single-band full battery's (§9v/§9w) exact counterpart for the joint
case.** `gd_joint_band_test.py` extended (still no aerosol, same rationale
as §11g) to cover all three scenes and both noise settings, and gained the
two pipelines §11g didn't yet have:

- **native / undistorted** — unchanged mechanism from §11g, nearest-row
  paired (§11f).
- **rectified** — new. Each band rectified independently (`gd_render.
  rectify`, already-existing single-band machinery) onto **one shared
  cross-band `s_grid`** (real degrees, intersection-clipped to both bands'
  covered range) — this is the *original* co-registration mechanism
  §11b/§11c proposed and then set aside in favor of nearest-row, built here
  specifically so the decision could be checked empirically rather than
  left as an argument. Rows align by construction (same `s_grid` index for
  both bands) — no pairing call needed for this pipeline.

**Smoke-tested all four scene/noise combinations (uniform, barcode,
realistic, uniform+noise) across all three pipelines** before handing off —
every combination renders and retrieves without error. One result already
worth reporting from the smoke test itself, at just 4 rows: **rectified's
CO2 bias sits around −8 ppm, versus native's/undistorted's well under
1 ppm** — the same order-of-magnitude gap already established for the
single-band case (§9l/§9m), now confirmed to reproduce when rectification
is the *cross-band* co-registration mechanism too. Consistent with §11c's
prediction; not yet run at full row-count/scene coverage to know if it
holds everywhere the way the single-band finding does.

**Realistic-scene rendering is the slow part**: ~110s per band for the
along-slit lookup table (both bands together ~220s), versus ~10s total for
uniform/barcode. Full native+undistorted (~1007 nearest-row pairs each)
plus rectified (~1024 shared-grid points) is roughly 3000 joint retrievals
per scene/noise combination — the one full run done so far (§11g, native
only, order=0, uniform, 1007 rows) took 585s at ~16-worker parallelism, so
the full 3-pipeline battery is real compute, not something to run
interactively case by case.

**Built for SLURM submission, not run at full scale yet:**
`scripts/gd_joint_band_test.slurm` (one task per scene/noise combination,
all three pipelines within it, env-var driven — `FPA_A`/`FPA_B`/`UNIFORM`/
`BARCODE`/`BARCODE_BARS`/`NOISE`/`SNR`/`NOISE_SEED`/`PIPELINES`/`ROW_STEP`,
mirroring `gd_band_stress_test.slurm`'s conventions) and
`scripts/gd_joint_band_test_submit_all.sh` (6 `sbatch` calls — 3 scenes x 2
noise settings — for the default FPA0+FPA2 pair, mirroring
`gd_band_stress_test_submit_all.sh`). Output:
`results/gd_joint_fpa<A>_fpa<B>[_uniform|_barcode][_noise].pkl`, one file
per combination, no collisions regardless of submission order.

**Status:** infrastructure complete and smoke-tested; queued for the user
to submit via `bash scripts/gd_joint_band_test_submit_all.sh`. Once results
land: build the same bias-vs-slit-position summary figures §9v/§9w used for
the single-band case, and check whether §11g's CO2-improves/pressure-
doesn't asymmetry (and this section's early rectified-bias signal) hold up
across all six combinations, not just the one uniform/order=0 case tested
so far.

### 11i. Full battery results, and a real barcode-specific failure mode found and diagnosed (2026-07-27)

**All 6 combinations completed** (`scripts/gd_joint_band_test_submit_all.sh`,
~3000 joint retrievals each, ~35-55 min per combination at 32 cores).
`scripts/gd_joint_band_plot.py` built for the summary figures (bias/chi2 vs.
along-slit position + failure-location panel, all three pipelines at once,
the joint counterpart of §9v/§9w's figures) and
`scripts/gd_joint_band_plot_residual_spectra.py` for representative
spectral residuals **by band** (best/worst/10-spanning native chi2, same
method as the single-band script) — output pkls only saved scalar bias/
chi2, not residuals, so this script re-renders each case and re-runs just
the ~12x3 representative retrievals fresh rather than the full ~3000;
noted directly in its own docstring as a gap found after the battery had
already completed, not by design.

**Realistic/uniform (both noise settings) reproduce §11g's pattern at full
scale**: native/undistorted track each other closely with small bias; a
real, shared feature (the H2O/pressure degeneracy, same one every single-
band figure shows) survives in both; rectified sits apart at a large,
roughly position-dependent offset (co2 mean -8 ppm, p_surface excursions up
to +16/-20 hPa in the realistic case) -- confirming §11c's prediction that
shared-grid rectification for cross-band co-registration inherits (and
here, compounds with real cross-band mismatch) the same interpolation bias
already found for single-band rectification (§9l/§9m).

**Barcode is a real, new joint-specific failure mode -- found, diagnosed,
not a bug.** Native's convergence drops to 844/1007 (barcode) / 843/1007
(barcode+noise) -- far below uniform/realistic's ~100% -- and of those
"converged" rows, a further 117 are chi2-outliers (§9v's mask), leaving
only 727/1007 (72%) actually trustworthy. Unlike the single-band barcode
finding (§9j: a sharp, narrowly-localized spike at a handful of rows near
bar boundaries), this failure is spread broadly across almost the entire
slit, not confined to a few rows.

Diagnosed directly from the residual-spectra plots: the "worst" row's
residual (chi2=6.43, no noise) is a huge spike concentrated in a handful of
channels near one band's edge (~1.5, against an otherwise near-zero
residual everywhere else) -- not a broadly bad fit, a narrow one. Since
`chi2_reduced` sums squared residuals over *every* channel, a handful of
channels with a huge residual dominate the whole statistic even though 99%+
of the spectrum fits essentially perfectly -- this is §9j's own mechanism
(keystone-heterogeneity produces a sharp spike right at a bar boundary),
just with much wider reach here. The reach is wider specifically *because*
it's joint: each band has its own real keystone/smile geometry, so bar-edge
discontinuities land at different rows in FPA0 than in FPA2 -- a joint fit
inherits *both* bands' own bar-transition footprints, not just one,
roughly doubling the exposed-row count relative to either band alone.

**With noise added, this mostly disappears** -- barcode+noise's worst chi2
drops to 1.15 and looks like ordinary noise-dominated residual throughout,
no isolated catastrophic spike. A deterministic spike that's dramatic
against an otherwise-perfect noiseless background is largely swamped once
real photon noise (~O(0.5-1) amplitude here) is already present everywhere.

**Bearing on the next band pair (FPA0+FPA1 vs. FPA0+FPA3, asked about
directly 2026-07-28):** computed the smile-slope-null row for FPA1/FPA3
(only ever done for FPA2 then FPA0 before now) specifically to check
parity before choosing:

| FPA | keystone-null row | smile-slope-null row | separation |
|---|---|---|---|
| 0 | 600 | ~550 | ~50 rows |
| 1 | 570 | ~565-572 | **~2-7 rows** |
| 2 | 25 | 519 | ~494 rows |
| 3 | 266 | ~507-511 | ~243 rows |

FPA1 is the one band where these two critical points nearly coincide --
every other band has them well separated, giving those bands a broader
region where at least one of the two mechanisms is favorable. FPA1 likely
has less of that safe margin. Combined with the barcode mechanism above
(each band contributes its own exposed-row footprint to a joint fit),
**FPA0+FPA1's barcode case is expected to show an equal or higher affected-
row fraction than FPA0+FPA2's ~28%, not less** -- a prediction to check once
that battery runs, not a reason to hold off on it.

**Status:** summary and residual-spectra figures complete for all 6
FPA0+FPA2 combinations. Barcode's failure mode is understood and explained
by an already-known single-band mechanism (§9j) operating jointly across
two bands' independent geometries -- not a defect in the joint-retrieval
machinery itself, and already correctly handled by the existing chi2-
outlier filtering in the summary statistics. FPA0+FPA3 (O2-A + CH4/CO,
`scripts/gd_joint_o2a_ch4co_submit_all.sh`) queued next; FPA0+FPA1 flagged
above as plausibly the more barcode-sensitive pair given FPA1's null-row
coincidence, worth running rather than skipping specifically because of
that prediction.

### 11j. Generalized to N>=2 bands (2026-07-28)

Everything in §11f-§11i above was built for exactly 2 fixed bands
(`fpa_a`/`fpa_b`). Asked directly what generalizing to more bands (e.g. a
full 4-FPA joint retrieval) would require; answer was that `gert`'s own
forward-model/state-vector machinery already handles an arbitrary number
of `Instrument` windows with no changes needed (`ForwardModel.run()`
concatenates however many windows it's given; `StateVector.gas_scaling()`
already zeroes a gas's Jacobian column per-window automatically) -- the
2-band assumption lived entirely in this study's own stress-test scripts,
in five specific places, all now generalized:

1. **Row co-registration.** `geocarb_gert.cross_band.nearest_row_pairing`
   (pairwise, unchanged) is joined by a new `nearest_row_pairing_multi
   (fpas, rows_ref=None)`: pairs every band in `fpas[1:]` to `fpas[0]`
   independently, then restricts to the along-slit rows that fall within
   *every* other band's covered range (an N-way intersection, not just one
   pairwise range). Each individual pairing stays independently bounded
   (<=0.5-row mismatch, §11f), so error doesn't compound as bands are
   added -- only the valid coverage can shrink.
2. **Rectified shared grid.** `_shared_s_grid(fpas, n=1024)` (both in
   `gd_joint_band_test.py` and reimplemented locally in
   `gd_joint_band_plot.py`, matching that file's existing
   don't-cross-import-plotting-helpers convention) now intersects all of
   `fpas`' covered ranges, not just two. Coverage shrinks (never grows) as
   more bands are added -- worth checking before committing to a specific
   4-band combination, since each FPA has a slightly different `s_max`.
3. **Band-indexed setup/state.** `_band_setup`/`_native_row`/
   `_undistorted_row`/`_rectified_row` were already single-band functions,
   unchanged; `_joint_retrieve` and `_joint_retrieve_with_residual` now
   take **lists** of bands/nu/y (order matching `fpas`) and build the
   `Instrument`'s windows, `prior_albedo`, and `y_true`/`sigma` concatenation
   in a loop instead of two hardcoded `_a`/`_b` slots. Per-band state names
   (`albedo_{b}`, `disp_a{k}_{b}`) already used the window's *position* in
   `instrument.windows`, so no change was needed there -- `b` now just
   ranges over more values. The per-gas truth-source rule generalizes the
   original 2-band rule (prefer `band_b`, default `band_a`) to "prefer the
   *last* band in `fpas` order whose molecule list contains the gas,
   default the first (reference) band."
4. **Task/key format.** Result-dict keys changed from
   `(pipeline, order, ka, kb)` to `(pipeline, order, rows)`, where `rows` is
   a tuple of one row/grid-index per band in `fpas` order -- a single
   representation that works identically for 2 bands or N, rather than a
   fixed-arity tuple. This is a breaking change to the on-disk pkl format
   (old 2-tuple `ka, kb` keys), but only the already-superseded FPA0+FPA2
   battery predates it, and those files were already being cleaned up as
   part of this same session's earlier request.
5. **CLI/SLURM/filenames.** `--fpa-a`/`--fpa-b` became a single `--fpas`
   flag taking a comma-separated list (>=2, unique, each 0-3); `gd_joint_
   band_test.slurm`'s `FPA_A`/`FPA_B` env vars became one `FPAS` env var
   (same comma-separated form); `gd_joint_band_test_submit_all.sh`'s
   positional-arg support (added externally between sessions, must not be
   reverted) now takes one positional arg holding the whole `FPAS` string
   rather than two separate `$1`/`$2` slots. Output filenames use a new
   shared helper `geocarb_gert.cross_band.fpas_tag(fpas)` (`[0, 2] ->
   "fpa0_fpa2"`, `[0, 1, 2, 3] -> "fpa0_fpa1_fpa2_fpa3"`) so the test
   script's output names and both plotting scripts' expected input names
   can never drift apart independently -- both plot scripts (`gd_joint_
   band_plot.py`, `gd_joint_band_plot_residual_spectra.py`) gained the same
   `--fpas` flag and now build filenames via the same helper.
   `gd_joint_band_plot_residual_spectra.py`'s per-band figure layout
   (previously a fixed 2-column native-`nu_a`/`nu_b` grid) is now
   `n_bands` columns wide, one per band in `fpas` order.

**Smoke-tested** (`--uniform --row-step 100-200`, small subsets): the
2-band FPA0+FPA2 case reproduces byte-for-byte the same state-vector names
and co2/p_surface bias as before the rewrite (regression check); a genuine
3-band FPA0+FPA1+FPA2 case runs cleanly end-to-end through all three
pipelines (native/rectified/undistorted converge; N-way pairing reports
each additional band's own mismatch-vs-reference separately), and both
plotting scripts render correctly against the 3-band result (`gd_joint_
band_plot.py`'s 2x2 summary figure; `gd_joint_band_plot_residual_spectra.
py`'s N=3-column residual grid, 11 representative rows x 3 bands = 33
axes, matching expectation).

**Status:** the pipeline now supports any >=2-band combination (e.g. a
future full FPA0+FPA1+FPA2+FPA3 run) without further script changes --
only a `--fpas 0,1,2,3`-style argument. Not yet run at full battery scale
for any >2-band combination; the FPA0+FPA1 and FPA0+FPA3 2-band batteries
already queued (§11i) are unaffected by this rewrite (still 2-band, just
now going through the generalized code path, confirmed equivalent above).

### 11k. Undistorted's excess scatter root-caused and fixed: float dispersion there too (2026-07-28)

**Symptom.** FPA0+FPA3 uniform/no-noise summary plots showed undistorted's
chi2 (~0.05) and gas-bias scatter far worse than native's (~1e-4-1e-8),
which was confusing given undistorted is supposed to be the *cleanest*
pipeline (zero geometric distortion, real per-row wavelength grid, same
gases retrieved almost to the true state). `_conv=True` for every row --
this was never a convergence failure.

**Root cause, traced to source.** Evaluated the forward model directly at
the TRUE prior state (no optimizer involved) and found the mismatch
between the "truth" measurement and a fresh `ForwardModel.run()` call
already present there, at essentially the SAME magnitude for native and
undistorted alike (std 0.0731/0.0731 FPA0, 0.0107/0.0107 FPA3, matched to
4 significant figures for the same row) -- ruling out a bug in
`_undistorted_row` specifically, since native's completely different
rendering path (`gd_render.image()`, real 2D pixel projection + spatial
PSF) shows the identical floor. Traced into `gert`:

- `geocarb_gert/gd_render.py`'s `_diagonal_ils_convolve` -- shared by BOTH
  native's rendering and undistorted's row construction to build the
  "truth" -- always centers the ILS kernel on the *exact* channel
  wavenumber.
- `gert/instrument.py`'s `ILS.convolve(..., exact_center: bool = False)`
  defaults to snapping the kernel center to the nearest hi-res grid point
  instead ("the legacy behaviour that reproduces the original
  ForwardFunction pipeline," per its own docstring).
- `gert/forward_model.py:797-801` is the switch between the two:
  ```python
  wn_centers = None
  if dispersion is not None and i in dispersion:
      wn_centers = win.dispersion_centers(dispersion[i])
  R_wn = win.convolve(I, wn_centers=wn_centers, width_scale=s_ils)
  ```
  `SpectralWindow.convolve()` only requests `exact_center=True` when
  `wn_centers` is given. Merely *having* dispersion in the state vector
  (order>=1) is enough to take this branch on every iteration --
  regardless of whether the coefficients are literally zero -- flipping
  the retrieval's own forward-model evaluation onto the same exact-center
  convolution the truth-generation path always uses. Order=0 (undistorted,
  no dispersion at all) can never take this branch, so it's permanently
  stuck on grid-snapped convolution, a small but real, ~row-independent
  chi2 floor against the exact-centered truth -- invisible everywhere else
  in this study because real gas signal or measurement noise normally
  swamps it, exposed here only because FPA0+FPA3's uniform/no-noise case
  removes both.

**Empirically confirmed** via `scripts/gd_joint_undistorted_dispersion_diag.py`
(a standalone diagnostic, not part of the regular battery -- runs
native/undistorted-order-0/undistorted-order-2 side by side for a ~26-row
sample across 3 scenes):

| scene | native (order=2) chi2 | undistorted order=0 chi2 | undistorted order=2 chi2 |
|---|---|---|---|
| uniform, no noise | median 9.9e-05 | median 0.050 | median **8.2e-11**, ch4/co/h2o std **0.000** |
| uniform, noise | median 0.997 | median 1.055 | median 0.996 (noise-dominated, floor irrelevant) |
| realistic, no noise | median 0.00409 | median 0.0545 | median **0.00337** |

Floating dispersion for undistorted doesn't just shrink the gap -- it
collapses it to at or below native's own level in every no-noise case,
and makes no difference once real measurement noise dominates (as
expected, since the floor is tiny relative to noise).

**Fix applied**, contained to the joint-retrieval scripts:
`gd_joint_band_test.py`'s undistorted task now uses order=2 (was 0) --
same 6 dispersion parameters as native/rectified, expected to converge to
~0 since there's no real calibration error for them to explain; floated
purely to force the matching exact-center convolution path. Both plotting
scripts' `PIPELINE_ORDER` dicts and the residual-spectra script's
undistorted retrieval call updated to match. Docstrings in all three
updated to explain this is a numerical-self-consistency workaround, not a
physical correction -- so a future reader doesn't mistake undistorted's
now-nonzero dispersion coefficients for it modeling a real effect.

**Scope note -- single-band script NOT changed.** `gd_band_stress_test.py`
has the identical `pipeline_orders = {"native": [2], "rectified": [2],
"undistorted": [0]}` convention, with its own comment explaining the
original reasoning: floating dispersion for undistorted "would give those
parameters nothing real to fit, risking spurious degeneracy with
h2o_scale/p_scale that would contaminate exactly the 'fundamental limit
vs. distortion artifact' question this baseline exists to answer." The
diagnostic evidence above directly contradicts that worry for the joint
case (dispersion collapsed the scatter, it didn't create spurious
degeneracy) -- but single-band `gd_band_stress_test.py`'s undistorted
results underpin a large fraction of this document's already-recorded §9
analysis (the H2O/p_scale degeneracy findings explicitly rely on comparing
against this order=0 baseline). Changing it would be a much bigger, more
consequential decision than the joint-only fix above -- left untouched
pending an explicit, separate decision on whether/how to revisit those
existing §9 conclusions.

**Status:** fix applied and smoke-tested for the joint scripts (FPA0+FPA3
uniform, native+undistorted only, chi2 confirmed collapsing to ~1e-10,
matching the diagnostic). FPA0+FPA3's existing battery results (all now
superseded by this convention change) queued for a full rerun next,
`--pipelines native,undistorted` (rectified is unaffected by this fix and
not being rerun).

### 11l. Residual capture added to the joint battery; full-scale confirmation and one prediction revised (2026-07-29/08-10)

**Residual capture closed.** `gd_joint_band_test.py`'s `_joint_retrieve` now
saves `_residuals`/`_nus` (one array per band) alongside the state vector,
matching `gd_band_stress_test.py`'s convention (residuals saved per row
since 2026-07-25) -- this was the exact gap §11's own memory-rule note
described, and had persisted in the joint script specifically until now.
`gd_joint_band_plot_residual_spectra.py` was simplified to read this
directly instead of re-rendering + re-running ~36 retrievals per case (its
own prior workaround for the missing data): **~1s per band pair now**,
down from ~15-20 minutes each. Verified across all 18 pkls after a full
rerun (all 3 band pairs, full 3-pipeline battery): 100% of converged rows
carry `_residuals`/`_nus`, undistorted's dispersion terms confirmed present
everywhere -- no gaps.

**§11k's fix reconfirmed at full battery scale**, not just the earlier
spot-checks: every uniform/no-noise case across all 3 band pairs now shows
undistorted's bias std at machine precision (`0.000` in every printed
summary stat, chi2 ~1e-10) -- the dispersion-order convolution artifact is
fully collapsed everywhere, not just the FPA0+FPA3 case it was originally
found in.

**New, previously-masked signal now visible for realistic scenes.** With
the numerical floor gone, undistorted's *mean* bias for the realistic
scene is now consistently ~0 across all three pairs (e.g. FPA0+FPA3 CH4
mean -0.002 ppb, FPA0+FPA1 CO2 mean +0.074 ppm), while native retains a
small but real, systematic mean bias (FPA0+FPA3 CH4 mean -0.525 ppb,
FPA0+FPA1 CO2 mean -0.298 ppm) at comparable scatter. This is a clean
confirmation that native's residual bias is a genuine, small effect from
real geometric distortion, not noise or a fitting artifact -- previously
this signal was there but harder to isolate against the pre-fix floor.

**Rectified remains the dominant bias source at every combination**,
unchanged in kind from §9l/§9m/§11i -- CO2 mean bias -8 to -14 ppm, CH4
mean bias -14 to -15 ppb, roughly an order of magnitude past native/
undistorted, now reconfirmed against the complete, dispersion-fixed,
residual-carrying dataset.

**Barcode convergence-failure fractions, computed for the first time for
all three pairs** (native pipeline, no noise, fraction of paired rows that
failed to converge or were chi2-outliers):

| pair | affected fraction |
|---|---|
| FPA0+FPA1 | 21.9% |
| FPA0+FPA2 | 27.8% |
| FPA0+FPA3 | 15.3% |

FPA0+FPA2's number matches §11i's earlier ~28% estimate closely. **FPA0+FPA1's
21.9%, however, contradicts §11i's own prediction** ("FPA0+FPA1's barcode
case is expected to show an equal or higher affected-row fraction than
FPA0+FPA2's ~28%, not less" -- based on FPA1's keystone-null and
smile-slope-null rows nearly coinciding, §9i/§11i's table). Empirically
FPA0+FPA1 is *less* affected, not more or equal. The null-row-coincidence
reasoning that motivated the prediction isn't necessarily wrong, but this
result shows it isn't sufficient on its own to predict barcode-mode
convergence-failure rate across band pairs -- worth revisiting if the
barcode-heterogeneity mechanism gets modeled more precisely later (e.g. as
part of the bias-correction covariate work in the project-summary
handoff's §6.3/§6.4).

**Status:** joint battery (all 3 pairs, all 6 scene/noise combos, full 3
pipelines) complete with residuals; all plots regenerated
(`gd_joint_fpa0_fpa{1,2,3}_all_summary.pdf`,
`gd_joint_fpa0_fpa{1,2,3}_all_residual_spectra.pdf`, plus per-case PNGs).

### 11m. Session housekeeping — local artifact copy, robust-stats table in the summary PDFs, plot-directory cleanup (2026-08-12)

A local, non-scratchpad copy of the published artifact was added at the repo
root (`NATIVE_VS_UNDISTORTED_ARTIFACT.html`) for offline reference alongside
the code that generated it.

**`gd_plot.py`'s combined summary PDFs now carry the robust-statistics
table**, previously console-only output from §9v's `robust_mean_std` across
every scene/noise/pipeline combination. New `_stats_table_lines()` (formats
a monospace scene/noise/quantity/pipeline/n/mean/std/median/MAD-sigma/n_out
table) and `_add_stats_table_pages()` (paginates it into extra
`PdfPages`-appended pages, ~46 rows/page) are called at the end of `main()`.
Verified: the FPA0+FPA2 combined PDF grew from 6 to 8 pages (6 case figures
+ 2 stats-table pages), content cross-checked against the console printout
via `pdftoppm` rendering (no `pypdf` available in this env).

**`plots/` cleanup**: 25 files using the pre-`gd_joint_*` naming convention
were removed (`git rm`) after confirming via `grep -l` across every current
`.py` file that nothing still generates or references them (e.g.
`alongslit_fpa2_uniform.png`, `barcode_chi2_grid_all_fpas.png`,
`gd_barcode_realistic_comparison.png` — full list in the commit).
`gd_along_slit_atm_profiles_fpa2.png` was kept since its generating script
(`scripts/gd_along_slit_atm_profiles.py`) is still live.
**Caution logged for next time**: an overly broad `rm -f
plots/gd_joint_fpa0_fpa2_*_summary.png` cleanup glob during this same
session briefly deleted files that were part of the user's own independent
commits (`637ffd8`, `908fe65`) made mid-session outside the assistant's
visibility — `plots/` is actively written to by the user directly, not only
by generating scripts, so cleanup globs there need to be scoped to exact,
known-generated filenames, never a wildcard broad enough to catch
externally-committed output. Fully restored via `git checkout HEAD --
plots/`; no data lost.

### 11n. Toy walkthrough of the native pipeline mechanism, for the write-up (2026-08-12)

`scripts/gd_toy_native_row_demo.py` — a single-row, code-faithful walkthrough
of exactly how one native detector row is built and retrieved (reusing the
real `xy_to_wavelength_slit`, `gd_render._diagonal_ils_convolve`,
`gaussian_blur_rows`, `ForwardModel.run()`, and `gd_test._joint_retrieve`
directly, not a simplified stand-in), built to support the along-slit
write-up's methodology section:

- **Panel A** — the target row's true per-column wavenumber vs. a reference
  row's, showing how a fixed column samples a different wavenumber (and,
  via keystone, a different true along-slit position) row to row.
- **Panel B** — neighbouring rows' hi-res spectra *relative to the target
  row's* (a difference view). The first attempt overlaid raw spectra at
  ±K, ±2K rows and found them visually indistinguishable even at ±60 rows,
  since the full dynamic range swamps the subtle depth differences that
  matter; switching to `S_show[i] - S_show[target]` immediately revealed
  real, line-shape-correlated structure.
- **Panel C** — a small local-neighbourhood render, pre- vs. post- along-slit
  PSF blur (`gaussian_blur_rows`, 1.5 px FWHM) — the one genuine cross-row
  physical mixing step in native rendering.
- **Panel D** — forward-model-at-prior vs. the measured row, plus pre-fit
  vs. actual post-fit residual (from a real `_joint_retrieve` call on that
  one row).

Output: `plots/gd_toy_native_row_demo_fpa2_row200.png`, generated and
visually verified for FPA2/row 200.

**Known follow-up, not yet applied**: Panel A used row 512 ("slit-centre
reference") as an implicit low-distortion comparison row, but
`geocarb_gert.gd_polynomials.rows_crossed(fpa=2, row=512) ≈ 5.24` is
actually the *largest* keystone value in the band, not the smallest — the
true FPA2 keystone-null row is row 25 (`rows_crossed(2,25) ≈ 0.005`). Row
512 should be swapped for row 25 before this figure is used in the write-up.

### 11o. Along-slit autocorrelation / effective native spatial resolution — a flawed test caught and corrected, then answered with a new scene (2026-08-12)

**Motivation.** The user raised a specific methodological concern about
§11's whole native-pipeline approach: retrieving each detector row
independently, when keystone+PSF genuinely mix multiple rows' true content
into each row's rendered spectrum, could make native retrievals *look* more
spatially resolved than the instrument actually delivers — i.e. adjacent
independent retrievals could be spuriously correlated by the shared
rendering mechanism, understating the real resolution loss.

**First attempt (flawed).** Built `scripts/gd_toy_row_autocorrelation.py`:
lag-0..60 Pearson autocorrelation of native's retrieval bias, computed on
the **uniform** scene, with a degree-6 polynomial detrend to remove the
smooth slit-wide distortion floor first. Found the ACF decaying to a noise
floor by lag 2-3 rows, with FPA2 showing a real lag-1 excess (~0.1-0.2 over
undistorted); reported to the user as "effective along-slit resolution
~2-3 rows (~5-8 km)."

**Why it was wrong** (caught directly by the user: *"this is not the
correct test because it doesn't account for state vector variations along
the slit"*): on a uniform scene, `radiance(eta)` doesn't depend on `eta` at
all — every column sees an identical spectrum regardless of which true
slit position keystone nominally assigns it, so there is *nothing* for
keystone to blend. A uniform-scene ACF can only capture pure
wavelength-registration (smile) correlation; it is structurally incapable
of showing keystone-driven spectral (absorption-depth/continuum) blending,
which needs truth that genuinely varies with position.

**Corrected methodology.** Rewrote the script to default to the (already
existing, §11's own) realistic along-slit scene; added `--scene
{realistic,uniform,barcode,realistic_barcode}`; replaced the polynomial
detrend with a raw/undetrended ACF over a longer lag range (150 rows, vs.
60) so both a short-range (PSF/keystone) regime and a long-range (shared
real-atmosphere) regime are visible and separable by eye — detrending a
scene with real spatial structure risks removing the very short-range
correlation being measured, not just a nuisance trend; and added an
explicit `excess = acf_native − acf_undistorted` curve, isolating what
native's own rendering (keystone+PSF) contributes beyond whatever
correlation the real, shared atmospheric truth already produces in *both*
pipelines.

**Result 1 — realistic scene.** Excess is tiny (|excess| ≤ 0.008, peaking
near lag 7-9 rows) and slightly **negative** — native decorrelates
marginally *faster* than undistorted at short range, not slower. Both raw
ACF curves are dominated (0.99 → ~0 over ~140 rows / ~400 km) by the real
atmosphere's own along-slit spatial correlation, which both pipelines carry
equally since both see the same underlying scene. **No evidence that native
manufactures extra, artifactual spatial correlation** beyond what the real
shared truth already produces — the "artificially inflated resolution"
concern doesn't materialize against a smoothly-varying truth.

**New scene added to answer the follow-up ask** ("do the same analysis for
a scene where the realistic scenes are modulated by the barcode pattern").
`gd_test.py` gained a new mode, `--realistic-barcode` (mutually exclusive
with `--uniform`/`--barcode`; `_band_setup()` gained a matching
`realistic_barcode` branch), combining the realistic scene's genuine
along-slit atmospheric variation with a 32-bar barcode brightness gain
multiplied on top — unlike the existing `--barcode` mode (which fixes the
atmosphere at its *center* value and only varies brightness, i.e.
`xtrue_x_km` forced to zero), this one keeps `xtrue_x_km = x_km_of_row` so
the underlying truth genuinely varies with position, same as the plain
realistic scene. Implementation reuses
`geocarb_gert.focalplane.barcode_scene`'s own η→gain bar/boundary math by
applying it to an all-ones "spectrum" (broadcasting one shared gain across
every hi-res bin) and multiplying that elementwise onto the real per-η
radiance from `als.build_lookup_radiance(..., uniform=False)`, rather than
duplicating the bar-boundary logic. `gd_test.slurm` gained a matching
`REALISTIC_BARCODE=1` env var. Smoke-tested on a 6-row FPA2 subsample
before committing to the full run.

Full run: FPA0-3, single-band, no noise, `realistic_barcode` scene, via
SLURM (`FPAS=<n> REALISTIC_BARCODE=1 sbatch scripts/gd_test.slurm`, jobs
1707625-1707628, ~40 min each on 32 CPUs). Native convergence dropped to
849-893/1024 (83-87%) vs. undistorted's 1023-1024/1024 (>99.9%) — some
native rows near a bar edge fail to converge outright, itself a signal that
real geometric mixing across a sharp brightness discontinuity is happening.

**Result 2 — realistic_barcode scene.** Starkly different from the smooth
case. Native's raw ACF **collapses from 1.0 at lag 0 to ~0.08 by lag 1**
and stays near zero (noisy, roughly ±0.1) out to lag 150, punctuated by
sharp re-correlation spikes at lag ≈ 32, 65, 97, 130 — exact multiples of
the 32-row barcode bar period (1024 rows / 32 bars). Undistorted keeps the
same smooth ~140-row decay it has in the plain realistic scene (insensitive
to the brightness pattern — it evaluates truth directly, no rendering, no
noise here). Excess is strongly **negative**, ~−0.92 to −1.10 by lag 1-10.

**Interpretation.** Keystone+PSF's effect on along-slit correlation is
strongly scene-dependent: negligible against a smooth truth (swamped by the
real atmosphere's own broad correlation), but produces large,
sharply-localized **de**correlation exactly where the truth has real
fine-scale (few-row to few-tens-of-row) structure for keystone+PSF to
genuinely mix — precisely the regime a "spatial resolution" claim is
actually about. This is the opposite of "artificially inflated resolution":
near real fine structure, native retrievals become *more* independent
(noisier row-to-row) than the underlying truth, not falsely smoothed
together. This directly answers and resolves the user's concern from
earlier in this thread.

**Status**: three figures
(`plots/gd_toy_row_autocorrelation_{uniform,realistic,realistic_barcode}.png`),
`gd_test.py`/`gd_test.slurm`/`gd_toy_row_autocorrelation.py` updated and
staged (not committed). New `results/gd_joint_fpa{0,1,2,3}_realistic_barcode.pkl`
(no-noise, single-band each) are gitignored, not committed — regenerate via
the SLURM command above. The old, un-suffixed `plots/gd_toy_row_autocorrelation.png`
from the flawed uniform-only version was removed (`git rm --cached` + delete)
in favor of the three scene-suffixed files. **Not yet done**: rerun any of
the three scenes with noise on; sweep the barcode bar count to see how the
native decorrelation length scales with bar width; apply §11n's row-25 fix
to the toy demo.

### 11p. A sharper thought experiment answered directly: does a real, single-footprint point source survive native retrieval near the high-keystone end of the slit? (2026-08-12)

The user pushed §11o's question further with a precise thought experiment:
near the end of the slit, a native row is built end-to-end from ~10
contiguous spatial regions' spectral contributions (modulo PSF). If a
single one of those footprints carries a strong, localized anomaly (a real
point-source plume, or a strong albedo gradient), the row's spectrum is
effectively ~90% one atmosphere and ~10% a totally different one — a
sharper, more directly diagnostic question than §11o's aggregate
autocorrelation statistics, and not the same thing as "correlation between
neighboring samples."

**This is exactly what `gd_render.image()` does, confirmed by code, not
just by the earlier docstring citation**: `rows_crossed(fpa, row)` (§11's
own keystone amplitude metric) IS the number of contiguous footprints a
row's own column range splices together, since keystone is precisely the
shift in true ground position as column/wavelength varies within a fixed
row. For FPA2 this grows from ~0.005 rows (the null row, 25) to **10.4
rows at row 1023** — confirmed by direct computation, not estimated. The
user's "10 contiguous regions" is not a hypothetical; it's very close to
the real number at the high-row end of this band.

**Directly testable with data that already exists.** `along_slit_scene.py`
already has real, ~9 km-wide Gaussian CO2 point sources built in
(`HOTSPOTS_CO2`) specifically for this purpose (see its own docstring —
"stress-test whether keystone/smile row-crossing preserves or smears a
genuinely localized source"), and one of the two happens to sit almost
exactly on FPA2's highest-keystone rows. Pulled straight from the existing
`results/gd_joint_fpa2.pkl` (plain realistic scene, no noise, no new
retrieval run needed) via new `scripts/gd_toy_hotspot_dilution.py`:

| location | keystone (rows crossed) | true peak enhancement | native captures | undistorted captures |
|---|---|---|---|---|
| row ~112 (near slit start), x0=−1100 km | 1.0 | +5.4 ppm | 90% | 91% |
| row ~910 (near slit end), x0=+1050 km | 9.3 | +4.0 ppm | **44%** | 99% |

Undistorted (no splicing, one true position per row) recovers the point
source almost completely at *both* locations. Native tracks it almost as
well at low keystone (90% vs. undistorted's 91% — both dominated by
ordinary retrieval scatter, not splicing) but **loses more than half the
true peak amplitude at high keystone** — and the recovered bump in
`plots/gd_toy_hotspot_dilution_fpa2.png` is visibly wider than the true
one, spread over roughly the same ~9-row span keystone predicts.

**This is a materially different, and more directly relevant, finding than
§11o's autocorrelation work.** §11o measured how correlated retrieval
*bias* is between neighboring rows (answering "how independent are two
adjacent samples"); this measures how much of a real, localized *signal*
a single native retrieval fails to see at all, because of the same
splicing mechanism, quantified in the physical units (ppm peak amplitude)
that actually matter for a source-detection or source-attribution claim.
Together with §11o's realistic_barcode result (native decorrelating
sharply near real fine-scale structure, not smoothing it away), this gives
a coherent, two-part answer: native doesn't inflate apparent resolution by
falsely smoothing independent ground truths together — but near real
fine-scale structure (a point source, an albedo edge) at high keystone, it
can substantially **under-recover** the true signal's amplitude, which is
its own, different, and arguably more operationally important limitation
to flag for the write-up.

**Status**: `scripts/gd_toy_hotspot_dilution.py` and
`plots/gd_toy_hotspot_dilution_fpa2.png` added and staged. Uses the
existing `results/gd_joint_fpa2.pkl`; no new SLURM run needed. **Not yet
done**: repeat for the other three FPAs' own hot spots (CH4/CO on FPA1/
FPA3) and for a sweep of keystone amplitude in between the two cases shown
(to map out capture-fraction vs. rows-crossed as a continuous curve rather
than two points); check whether the effect is source-width-dependent (the
current hot spots are ~9 km, close to one footprint — a narrower or wider
source would presumably dilute less or more).

---

## 12. Plan: calibration-mismatch (imperfect keystone/smile knowledge) experiment (2026-07-27)

**Motivation.** Every experiment so far (§9v/§9w, the full 24-case sweep)
gives the retrieval *exact* knowledge of the real GD polynomials — native's
`nu_row`, rectified's rectification target grid, and undistorted's grid are
all derived from the same `xy_to_wavelength_slit`/`wavelength_slit_to_xy`
calls that `gd_render.image()` used to render the "truth" detector image in
the first place. Real calibration knowledge is never exact: the ground-test
polynomial fit (`gcmap_em27.csv`, §9's provenance) has its own residual fit
error, and the real instrument may drift from that fit on-orbit (thermal
changes shifting the optics). This section designs an experiment that
introduces a genuine mismatch between what the detector *actually* does
(render side) and what the L1B/retrieval pipeline *believes* it does
(retrieval side) — a calibration-knowledge axis this study hasn't touched
at all yet.

### 12a. Design principle — two coefficient sets, not one

Mirrors the pattern `_retrieve()` already uses for the atmosphere (prior is
deliberately the fixed slit-centre state, not the row's true state, per its
own docstring): keep `gd_render.image()`/`rectify()` on the **real**
coefficients (`gcmap_em27.csv` as loaded today, unperturbed) to define
truth, and give only the retrieval-side position bookkeeping in
`_worker()` — native's `nu_row`, rectified's target `wn_grid`/`s_grid`,
undistorted's `nu_row`/`eta_row` — an **assumed** (perturbed) coefficient
set. Nothing about how a detector pixel's true spectrum is rendered
changes; only the pipeline's *belief* about which (wavelength, slit
position) that pixel corresponds to changes. This isolates calibration
error from every mechanism already characterized (rectification
interpolation §9l/§9m, row-crossing §9w, prior-atmosphere mismatch) since
none of those require the two coefficient sets to differ at all.

### 12b. Two perturbation mechanisms, matching the two named causes

- **Training-data noise** — a single random, smooth, low-order perturbation
  to the (wavelength, slit) polynomial maps, drawn once per experiment with
  a fixed seed. Deliberately *not* per-pixel white noise: the real error
  source is a finite-sample polynomial fit, whose residual error is smooth
  across (x, y), not independent pixel-to-pixel. Implemented as an additive
  perturbation to a handful of the polynomial's own low-order terms
  (`const`, `x`, `y`, `xx`, `yy` — see `_TERMS` in `gd_polynomials.py`),
  scaled to a target RMS in physical units (cm⁻¹ for wavelength, km for
  slit position) rather than raw coefficient magnitude, so the knob means
  the same thing regardless of which terms happen to carry it.
- **On-orbit thermal drift** — a deterministic (not random), smooth
  systematic shift, representing optical-bench thermal expansion moving
  the whole dispersion/keystone relation. Modeled the same way (perturb
  the same low-order terms) but with a fixed sign/shape rather than a
  random draw, and swept over a magnitude parameter standing in for
  assumed ΔT, rather than reseeded.

Both perturbation types share one mechanism (`perturbed_coeffs()` in
§12d below) — "noise" and "drift" differ only in whether the perturbation
vector is drawn randomly (seeded) or set deterministically, not in how
it's applied.

### 12c. Critical asymmetry to exploit — wavelength error is partly self-correcting, slit-position error is not

Native/rectified already float `disp_a0_0`/`disp_a1_0`/`disp_a2_0` in the
state vector when `order=2` (`_retrieve()`'s `include_dispersion=(order>0)`)
— a nuisance nudge to the assumed *wavelength* calibration that a real
retrieval fits per-scene. A **wavelength-only** mismatch experiment
therefore measures how much of an assumed smile/dispersion error the
existing pipeline can absorb before it shows up as gas bias.

There is **no equivalent nuisance parameter for slit position** anywhere
in `StateVector.gas_scaling()` — nothing in the state vector lets a fit
compensate for "this pixel's assumed along-slit position is wrong." A
**keystone/slit-only** mismatch experiment therefore measures a raw,
structurally uncorrectable bias: whatever error this injects is exactly
what a real pipeline would carry too, since there's nothing today (in
either this simulator or, presumably, an equivalent real L2 algorithm's
own state vector) built to self-calibrate it out.

Plan: run wavelength-only, slit-only, and both-combined mismatch as three
separate cases (not just one "calibration error" case) specifically to
keep these two failure modes distinguishable in the results.

### 12d. Implementation sketch

New function `perturbed_coeffs(fpa, wn_bias_cm1=0.0, wn_noise_rms_cm1=0.0,
slit_bias_km=0.0, slit_noise_rms_km=0.0, seed=None)` returns a copy of
`gd_polynomials._coeffs()`'s `A{fpa}`/`B{fpa}` term dicts with the `const`
(and optionally `x`/`y`) terms nudged by an amount calibrated to hit the
requested RMS/bias in physical units over the real pixel grid — a
low-order, smooth perturbation by construction, satisfying §12b's "not
per-pixel noise" requirement without needing a richer basis. A parallel
`xy_to_wavelength_slit_assumed(fpa, x, y, mismatch)` evaluates the same
`_poly2d` call `xy_to_wavelength_slit` does, but against the perturbed
coefficients — a drop-in replacement wherever `_worker()` currently calls
the real function for retrieval-side bookkeeping.

`gd_band_stress_test.py` gains a `--mismatch-mode {none,wavelength,slit,
both}`, `--mismatch-wn-bias-cm1`, `--mismatch-wn-noise-cm1`,
`--mismatch-slit-bias-km`, `--mismatch-slit-noise-km`, `--mismatch-seed`
CLI surface; `_worker()`'s native/rectified/undistorted branches call
`xy_to_wavelength_slit_assumed(...)` instead of `xy_to_wavelength_slit(...)`
when any mismatch is requested. `gd_render.image()`'s own truth-rendering
call is untouched.

### 12e. Run matrix

Start with the **realistic** scene (most physically relevant), all 4 FPAs,
no measurement noise (isolate the mismatch signal cleanly before stacking
it with SNR noise): `{none, wavelength-only, slit-only, both} x {small,
large magnitude}` = 8 mismatch cases x 4 FPAs = 32 runs, reusing the
existing bias/chi2/robust-stats (§9v) and grid/residual-spectra plotting
scripts unchanged (they already key off whatever's in each result pkl).
Extend to noisy/uniform/barcode scenes only once the mismatch-only signal
is characterized, mirroring how §9v's own scope grew.

### 12f. Open: where do the magnitude numbers come from

Ideally the "noise" magnitude is the real ground-test polynomial fit's own
residual scatter (if `keystone_report.pdf` documents a fit RMS/covariance)
rather than a guess. Absent that, parameterize both magnitudes as a
fraction of each FPA's own real smile/keystone amplitude — a "small"
mismatch comparable to the report's stated fit-quality claims, a "large"
one large enough to be clearly visible in the bias/chi2 statistics — and
say so explicitly in the run's metadata so results are never mistaken for
a real, documented calibration-error estimate.

### 12g. Built and smoke-tested (2026-07-27) -- a finding that revises §12c

Implemented: `geocarb_gert.gd_polynomials.perturbed_coeffs()` and
`xy_to_wavelength_slit_assumed()` (bias + seeded-random low-order
perturbation of the real A{fpa}/B{fpa} term dicts, physical units in,
verified against target RMS/bias -- see module for details); `gd_band_
stress_test.py` gained the `--mismatch-mode {none,wavelength,slit,both}`
CLI surface described in §12d, wired into native's and undistorted's
`nu_row` construction in `_worker()`. `rectified` is dropped from the run
whenever a mismatch is active (needs an assumed-*inverse* mapping,
`wavelength_slit_to_xy`'s counterpart, not built yet -- `gd_render.
rectify()` still only knows the real one).

A 4-row FPA1 smoke test (`--row-step 300`, `wavelength` vs `slit` vs
`none`) turned up something the design in §12a-c didn't anticipate:
**`slit`-mode mismatch has exactly zero effect on native/undistorted's
retrieved bias** -- byte-identical results to the `none` baseline at every
row, not just small. `wavelength`-mode mismatch, by contrast, does move
the bias (row 0: −3.955 -> −3.483 ppm CO2 at wn_bias=0.05 cm-1 + wn_noise_
rms=0.02 cm-1), confirming that channel works as designed.

**Why**: tracing `_worker()`/`_retrieve()`, slit/keystone position never
actually enters the retrieval math for native or undistorted. Native reads
its row directly from the real rendered image (`g["A"][k,:]`) and only
ever hands the retrieval a wavenumber grid (`nu_row`, from the wavelength
polynomial) -- there is no per-pixel slit-position input anywhere in
`ForwardModel`/`StateVector.gas_scaling()`'s single-atmosphere-per-row
design. Undistorted's simulated measurement is built from the row's REAL
`x_km_of_row` (correctly so -- that's the true position the along-slit
scene should be evaluated at); an *assumed* slit position has nowhere to
plug in without conflating "what the detector saw" with "what the
retrieval believes," which would defeat the render/retrieval separation
Sec. 12a is built on.

The one place slit-position calibration error *does* have a real
mechanism to bias results is **rectified**: `gd_render.rectify()`'s
inverse mapping (`wavelength_slit_to_xy`) decides which raw pixel's data
lands at a given target `(s, wn)` grid point -- get that mapping wrong and
the rectified spectrum is built from the wrong raw pixels, a genuine,
already-partially-built mechanism (§9l's whole rectification-
interpolation-bias finding lives in this same function). This reframes
§12c's asymmetry: it isn't just that keystone error is *harder to
correct* than wavelength error within this simulator -- with the current
sketch, keystone error has **no way to enter the bias at all** for
native/undistorted, and the only pipeline structurally sensitive to it
(rectified) is exactly the one this sketch had to drop for lack of an
assumed-inverse mapping.

**Revised next step**: build `wavelength_slit_to_xy_assumed()` (same
`perturbed_coeffs()` dict, evaluated through the C{fpa}/D{fpa} inverse
polynomials) and a parallel `rectify_assumed()` (or a `mismatch` kwarg on
`gd_render.rectify()` itself) so `slit`-mode mismatch has a real pipeline
to show up in. Until that exists, `--mismatch-mode slit`/`both` for
native/undistorted should be understood as exercising the wavelength term
only in practice (`both` currently degrades to `wavelength` for those two
pipelines) -- not wrong, just not yet testing what its name promises.

**Status:** `wavelength`-mode mismatch is real and usable today for
native/undistorted (not yet run at full 1024-row/4-FPA scale -- only the
4-row smoke test above). `slit`-mode mismatch needs the assumed-inverse-
mapping/rectify extension above before it measures anything.
