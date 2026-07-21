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
| keystone | slit **image** grows **20 px** blue→red = **1.95 %** (`keystone_px=20`); symmetric about slit centre → each end grows 10 px |
| N/S PSF | **1.5 px** FWHM (`spatial_psf_fwhm_px=1.5`) ≈ 4.4 km ground blur |
| smile | "pronounced", parabolic opening toward long-wave — **per-band px TBD** |

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

1. **Per-band smile amplitude** (px) and per-band keystone (default `keystone_px=20`
   for all four) — need the real numbers per band.
2. **η sampling**: dense nonlinear sweep (e.g. 32–64 η) + linear-vs-nonlinear validation
   at ~5 η; revisit after the Phase-1 comparison.
3. **Dispersion order to carry into Phases 2–4** (from the Phase-1/order sub-study).

---

## 8. First concrete step

Build the GERT-grid smile truth generator + the retrieval driver **with dispersion
retrieval (order-2, loose prior)**, then for **Phase 1** run the **5-η
linear-vs-nonlinear** comparison at two configs (smile-only, smile+keystone) **across
dispersion order 1/2/3** — expecting order-2 to zero the uniform-scene bias, and
confirming where the linear estimate diverges near the slit ends.
