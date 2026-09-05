# Realistic-prior experiment class

**Goal.** Move the joint-block retrieval away from the "local-truth nuisance
idealization" (every prior set to the exact truth at its own bin center, used
throughout the closed regression, anchor-density sweeps, and barcode tests to
find code bugs) toward priors that are *plausible but wrong*, the way a real
L2 prior is, so retrieval performance can be measured honestly. Truth carries
real fine-scale structure (dense along-slit rendering); priors know only
large-scale/climatological structure and known static topography, never the
localized hot spots. This doc covers the new mechanisms, the forward-only
sanity check built to validate them before any retrieval is run, and its
results.

---

## 1. What's new

### 1a. Structural (background/topography-aware) priors

`geocarb_gert/along_slit_scene.py` gained `xco2_ppm_prior`, `xch4_ppb_prior`,
`xco_ppb_prior`, `p_surface_hpa_prior`, and a `STATE_FIELDS_PRIOR` table
mirroring `STATE_FIELDS`. Each truth field decomposes as `background(x_km) +
plume(x_km) + hotspots(x_km)` (trace gases) or `background(x_km) +
mountain(x_km)` (p_surface); the structural prior keeps only:

- trace gases: the smooth regional `background` sinusoid, dropping both the
  broad `plume` and the narrow `hotspots` — a real prior wouldn't know about
  either kind of localized, unmodeled enhancement.
- p_surface: the `mountain` term exactly (a Gaussian depression, amp=-250 hPa
  at x0=-250 km, width=140 km — the topographic pressure signature, knowable
  via a hypsometric adjustment) plus the **midline** of the synoptic
  sinusoid, not the sinusoid itself (day-to-day weather a static prior
  wouldn't have).
- `h2o_surface_vmr` is unchanged (a single smooth climatological gradient,
  no hot-spot component to begin with).

### 1b. Prior-resolution knob

`geocarb_gert/joint_state.py`'s `state_spec_from_scene()` gained
`prior_anchor_density: float | None`, orthogonal to which field table is
used. `None` (default) reproduces today's exact mechanism unchanged — every
bin's prior is its own field function evaluated at that bin's own position.
Any float instead samples the field at `round(density * n_bins)` anchor
points spread across the window and linearly interpolates onto the actual
bin positions: `1.0` lands on exactly the bin count and is special-cased to
reuse the bin positions themselves (bit-identical to the default); `<1.0` is
a genuinely coarser prior (interpolating between sparse *true-field* samples
smooths away hot spots even though the field itself is exact truth); `>1.0`
oversamples.

### 1c. Dense truth rendering

`gd_joint_block_whole_slit_sweep.py` and `gd_cache_band_image.py` both gained
`--n-lookup-samples` (default 400 = 7 km along-slit spacing, ~3 samples
across a 10 km hot-spot FWHM) so the truth image's own resolution is
independent of, and can be pushed well past, the retrieval's bin/anchor
resolution — a one-time forward-rendering cost, not paid per solve.

### 1d. New CLI flags and output location

`gd_joint_block_whole_slit_sweep.py`: `--realistic-prior` (switches to
`STATE_FIELDS_PRIOR`, defaults `--n-lookup-samples` to 5600 and `--out-root`
to `results/realistic_prior`), `--prior-anchor-density`, `--out-root`.
Mutually exclusive with `--uniform`/`--barcode`/`--realistic-barcode` (those
flatten the *truth* scene; `--realistic-prior` only changes what the prior
knows). This is a new, separate experiment class — every existing result and
default is unaffected (`prior_anchor_density=None`, `fields=None` reproduce
prior behavior exactly).

---

## 2. Forward-only sanity check (`gd_realistic_prior_forward_check.py`)

Before spending a full retrieval sweep on the new mechanisms, this script
answers a narrower question directly: **how wrong is the prior, by itself,
before any solve gets a chance to correct it, and how does that change with
prior resolution/construction?** It renders the dense-truth FPA image once,
then for each named prior config evaluates the forward model *at the prior
itself* — `forward(spec.x0())`, i.e. `free=()`, no `gauss_newton_state` call
— window by window, stitches the results into a full 1024×1024 image, and
diffs against truth.

```
PYTHONPATH=. python3 scripts/gd_realistic_prior_forward_check.py
PYTHONPATH=. python3 scripts/gd_realistic_prior_forward_check.py \
    --prior-anchor-density 0.25,1,4 --no-structural --n-lookup-samples 400
PYTHONPATH=. python3 scripts/gd_realistic_prior_forward_check.py --plot-only
```

`--plot-only` regenerates every plot from the already-saved `.npy` arrays
without re-rendering — useful for iterating on the figures.

Six configs, all against the same FPA2 truth (`--n-lookup-samples 5600`,
0.5 km spacing, ~20 samples/hot-spot-FWHM):

| config | fields | prior_anchor_density | native_res | median %resid | max %resid |
|---|---|---|---|---|---|
| `exact` | STATE_FIELDS | `None` | False | 0.003% | 0.073% |
| `exact_ad1` | STATE_FIELDS | `1.0` | False | 0.003% | 0.073% |
| `oversample4` | STATE_FIELDS | `4.0` | False | 0.003% | 0.073% |
| `coarse0.25` | STATE_FIELDS | `0.25` | False | 0.005% | 0.462% |
| `structural` | STATE_FIELDS_PRIOR | `None` | False | 0.203% | 0.344% |
| `highres` | STATE_FIELDS | `None` | **True** | 0.002% | 0.074% |

(%resid = per-row residual RMS / mean\|truth\| that row.)

`highres` is a fundamentally different knob from `prior_anchor_density`,
worth spelling out since it's easy to conflate the two: `prior_anchor_density
> 1` ("oversampling") only changes how *accurately* the existing `G`
coarse-bin prior values are computed from `fields` — it can only approach the
`exact` (`prior_anchor_density=None`) result from below, never exceed it,
because `exact` already gives every one of those `G` bins its perfect value
(confirmed both analytically and by `oversample4` landing exactly on
`exact`'s own numbers above). `highres` instead builds the **state itself**
directly on the native-row anchor grid — one exact-truth value per detector
row, not per coarse bin — so there is no bin→row interpolation step at all
(`state_interp` becomes a no-op, same as the coarse solve's own "scene
positions are the state positions" case).

### Outputs

- `results/realistic_prior/forward_check/truth_fpa2_nls5600.npy` — the dense
  truth image.
- `results/realistic_prior/forward_check/prior_<label>_fpa2.npy` /
  `resid_<label>_fpa2.npy` — each config's forward-predicted image and its
  residual against truth, full 1024×1024.
- `plots/realistic_prior/forward_check_residual_by_row_fpa2.png` — per-row
  %residual, all configs on one log-scale line plot.
- `plots/realistic_prior/forward_check_2d_<label>_fpa2.png` — per-config
  full-FPA residual, two panels (raw W/m²/sr/µm and % of local truth
  continuum), both `SymLogNorm` + `RdBu_r`, matching
  `gd_joint_block_residual_plot.py`'s `full_detector_figure()` convention.
- `plots/realistic_prior/forward_check_2d_comparison_fpa2.png` — all six
  configs' %residual on **one shared color scale**, for direct amplitude
  comparison.

### Findings

- `exact`, `exact_ad1`, and `oversample4` are indistinguishable (median and
  max both identical to 3 significant figures) — confirms `prior_anchor_density
  =1.0` really does reproduce the default mechanism bit-for-bit, and that the
  windows here are already narrow enough that 4x anchor oversampling adds
  nothing beyond what exact-per-bin already provides.
- `exact`'s own nonzero floor (median 0.003%, max 0.073%) is the pure
  bin-center→native-row linear-interpolation error characterized earlier in
  this project — present even with a perfect prior, since a discrete set of
  bin priors is still being interpolated onto every detector row.
- `coarse0.25` tracks `exact` almost everywhere (same background level in
  the row plot, same faint keystone-shaped bands in the 2D image) but spikes
  roughly an order of magnitude higher exactly at the two CO2/CO hot-spot
  locations (~row 100-120, ~row 890-920) — sparse anchors smear a feature
  narrower than their own spacing, visible as saturated bands in the 2D
  comparison grid.
- `structural` is uniformly worse — not just at hot spots — sitting at
  roughly 0.2-0.3% almost everywhere in the 2D image (mostly one sign, since
  the missing plume/synoptic terms are smooth, slowly-varying offsets rather
  than symmetric noise), consistent with it being blind to real structure
  broadly, not only at the narrow point sources.
- `highres` (median 0.002%, max 0.074%) is barely distinguishable from
  `exact` — visually identical in the 2D comparison grid, and its median is
  only marginally lower. **Superseded by §3 below**: this is NOT because
  `exact`/`highres` sit on some resolution floor — `highres` is just density
  1.0 on the same continuous convergence curve §3 maps out, and density 1.0
  hasn't gained much yet relative to `exact` because `G`-bin interpolation
  error was already small at that density, not because further resolution
  stops helping.

---

## 3. RT-resolution convergence sweep (`--rt-sweep`)

**Question**: starting from a coarse-like state (`g_ratio=1`, ~2.7 km/bin)
and pushing anchor resolution toward the truth's own resolution, prior=truth
exactly throughout, is there an inflection point where finer sampling stops
helping — presumably set by pixel/PSF resolution?

**Mechanism**: `render_prior_image(..., native_res=True, rt_density=d)`
builds the state *directly* on the anchor grid at spacing `KM_PER_ROW / d`
(no separate `G`-bin layer at all — see the function's own docstring for why
this differs from `prior_anchor_density`), prior=truth exactly at every
anchor, `free=()`. `--rt-sweep 0.25,0.5,2,4,8,16` (density 1.0 already
covered by `--highres`) renders one such image per density, all diffed
against the same `n_lookup_samples=5600` truth.

**Finding — corrects the `highres`≈`exact` reading above**: `residRMS`
does **not** plateau anywhere in the tested range (spacing 11 km down to
0.17 km) — `forward_check_rt_sweep_fpa2.png` is a straight line on log-log
axes throughout: global max goes 0.175% → 0.073% → ... → 0.004% as spacing
drops 11 km → 2.7 km → ... → 0.17 km, an ~18x drop for a ~16x finer spacing,
i.e. first-order convergence with no sign of leveling off. This makes sense
once traced to the actual mechanism (§4 below, from a follow-up question):
`nearest_bin_scene` is a **hard nearest-anchor pixel assignment, never
interpolated** — every detector pixel snaps to its single nearest anchor's
precomputed spectrum, by construction a first-order-in-spacing error
(`~(1/4)|f'|h`, per that function's own docstring). The row-mixing PSF
(~4.4 km) doesn't remove this: it blends across ROWS, but keystone spreads a
single row's own pixels across a range of true η column-to-column, and nothing
smooths that out except finer anchor spacing along the slit itself. So there
is no reason for this curve to level off near the PSF scale, and empirically
it doesn't.

**So what answers the original question?** Not a kink in this curve — a
comparison against the instrument's own noise floor, since resolution finer
than that can't be distinguished from noise in a real fit. FPA2's SNR = 300
(`gd_test.DEFAULT_SNR_BY_FPA[2]`) puts that floor at `1/300 ≈ 0.33%` of
continuum, plotted as a reference line. It sits **above every curve at every
density tested**, including the coarsest (density=0.25, spacing ~11 km,
global max ~0.175%, already ~2x below the noise floor). Practical
consequence: representation error from RT/anchor resolution alone, given a
perfect prior, is already negligible relative to instrument noise at even
the coarsest resolution tested here — it is not the resolution axis that
will limit a real retrieval on this scene, it's prior quality (`structural`,
§2, sits closer to the noise floor: median 0.203% vs. the 0.33% line).

---

## 4. Pixel-snap mechanism (background for §3)

`geocarb_gert/focalplane.py`'s `nearest_bin_scene`: "Each pixel's η is
assigned to its single nearest bin center and returns that bin's own hi-res
spectrum unchanged — hard assignment, no interpolation of spectra." Every
one of the ~1024×(window rows) individual detector pixels gets its own
continuous η from the real keystone/smile geometry
(`gd_render.predict_neighborhood`: "every pixel is still evaluated at its
own true (eta, nu)"), and is snapped to whichever anchor is nearest — never
a blend of two anchors' spectra. Only the *state* feeding each anchor's own
RT run can be interpolated (`spec.interp_to`); the resulting spectra
themselves never are (the docstring notes this was deliberately chosen after
spectrum-blending was found to be the dominant bias source in an earlier
version, `rectify()`). This is the mechanism §3's convergence curve is
tracing the discretization error of, and it is a genuinely separate error
source from state/representation error (how well the state's value at each
anchor tracks the true field) — increasing `subsample_density` (planned
naming, see project notes) only ever reduces the latter, and both are
subject to §3's first-order-in-spacing scaling with no inherent floor from
the PSF.

---

## 5. Status

Forward-only validation is done; the mechanisms behave as designed. No
retrieval sweep has been run yet under `--realistic-prior`. Still open: the
solve-resolution sweep (hold RT evaluation fixed near a practical density,
sweep `G` via `g_ratio`, to find how coarse the SOLVE itself can go — see
project discussion) and the `subsample_bin`/`subsample_density` renaming for
the retrieval-side (not just diagnostic-side) code.
