# Multi-band joint-block retrieval: plan (draft, 2026-09-20)

Goal: a joint-block retrieval that fits several GeoCarb bands (FPAs) in ONE
state vector, on a code baseline good enough that all single-band experiments
can then be rerun on it (user's plan, 2026-09-20). Draft for discussion --
nothing here is implemented yet except the eta groundwork (Sec.33 of
`PROJECT_STATUS.md`).

## 1. Why (what single band cannot do)

Single-band retrievals show correlated errors that only extra bands can break:
p/aerosol in O2-A (o5y: p error correlates 0.95 with the true plume amplitude;
no-aerosol o5n removes it), p/h2o (~-0.69, survives every solver in FPA2), and
low-albedo CO2 (~0.84 ppm). A multi-band fit should constrain p from O2-A and
CO2/h2o/T/aerosol from the other bands at the same scene locations.

## 2. What already exists

| Piece | Status |
|---|---|
| Shared coordinate: per-band slit-image `eta` (`gd_polynomials.eta_of_s`), scene/state fields functions of `eta * SLIT_HALF_KM` | done (Sec.33); checked by `scripts/check_eta_consistency.py` |
| Cross-band row pairing keyed on `eta` (`cross_band.nearest_row_pairing[_multi]`) | done; residual <= half a row |
| gert multi-window `Instrument` (`build_geocarb_instrument` builds all 4 bands); Jacobian helpers take a `window` index | exists in gert; NOT used by the joint block (it builds a 1-window instrument per band in `band_basics`) |
| Per-row multi-band retrieval (`gd_per_row_retrieve.py --fpas`) | exists, separate pipeline (per-row, not joint-block/StateSpec) |
| Combined pixel-density bin placement (`combined_information_weighted_bin_centers`, `pixel_density_bin_centers`) | exists in `joint_state.py`; accepts any eta samples |
| `gauss_newton_state(forward, y, spec, Sy_inv_diag, jacobian_fn=...)` | generic in `forward`/`y`: a stacked multi-band forward drops in without changing the solver |

## 3. What is single-band today (the work)

Everything keyed on ONE `fpa`: `build_forward_state(fpa, rows_win, ...)`,
`render_at_anchors(fpa, ...)`, `jacobians.linearize(fpa, ...)`,
`state_spec_from_scene(..., band_label=...)` (one `albedo` row, one band label),
`geocarb_noise_model(FPA)`, `band_basics(fpa)` (1-window instrument),
`build_window_tiles(fpa, ...)` (row tiling of one band), driver `--fpa`, output
naming, sbatch CONF cases.

## 4. Design

**Keystone (decided).** The GD mapping polynomials are treated as known exactly, for the
truth AND the retrieval (no perturbed/assumed polynomials in these baselines).
No experiment is to be run with keystone switched off or set to zero -- real
keystone is always on, since it is what makes the multi-band eta matching hard.

**4.1 Composite forward, not a rewrite (recommended).** Keep the per-band
`render_at_anchors` / `build_forward_state` / `linearize` as they are and build:

    forward_joint(x)  = concat_b forward_b(view_b(x))
    J_joint(x)        = vstack_b  J_b  embedded into the joint columns
    Sy_inv_joint      = concat_b Sy_inv_b

where `view_b(x)` is a `StateSpec` VIEW giving band b its own copy of the shared
rows (gases, p, T, h2o, aerosol) plus its OWN albedo row. The solver, priors,
sigmas, dx/sigma convergence and clamping are unchanged. Cost is the sum over
bands; a later optimisation can share atmosphere-profile setup across bands or run
gert's multi-window `ForwardModel.run` once per anchor.

**4.2 State vector.** Shared rows on the shared eta bin grid; per-band albedo rows
`albedo_<band>` (user: albedo is per-bin/spatially resolved, never a shared
scalar), each with its own positions, sigma and correlation length. Row-name
mapping (`albedo_O2_A` -> that band's `albedo` row) lives in the view. Shared
`ROW_SIGMAS`/`ROW_KINDS` (absolute rows need physical-unit sigmas) carry over.
Bin centres from the concatenated pixel-eta samples of all bands.

**4.3 Windows (decided: tiling must respect BOTH bands' keystone maxima).**
Today's single-band tiling (`build_window_tiles`) sets each window's half-width
from `rows_crossed` at the window's CENTRE only (`r = 2.2 * rows_crossed`, fixed
point). That is not sufficient for two bands: keystone differs by band and by
position (0 at FPA2's keystone-null row 25, ~10 rows at the slit ends), and the
maximum inside a window is not at its centre. Multi-band tiling:
1. Work on the shared `eta` axis. For each band, express the keystone crossing in
   `eta` (span of `eta(row, col)` across the columns, from the real polynomials),
   not in real-angle rows.
2. A window's half-width in `eta` = the largest of the two bands' scaled crossing
   MAXIMA over the whole window (not the centre), so neither band's window is
   narrower than its own keystone requires; iterate to a fixed point as today.
3. Per-band row range = every row whose pixels' `eta` (all columns, plus footprint
   half-row) overlaps the window's `eta` interval, plus the usual PSF pad.
4. Overlap between neighbouring windows defined in `eta` and applied to both bands.
Gate: a test that, for every window, each band's row range contains every pixel
whose keystone trace intersects the window's eta interval, and that the window is
never narrower than either band's max crossing inside it.

**4.4 Aerosol (decided).** Spectral properties (extinction/`qext` scaling, single-
scattering albedo, phase function) come from GERT's aerosol model at each band's
wavelength -- not re-implemented here. The state carries one shared set of
aerosol rows per location: the same atmospheric HEIGHT (`height_aerosol`) and
WIDTH (`thickness_aerosol`) in every band, with amplitude defined at the reference
wavelength and scaled per band by GERT. Open sub-item: `amplitude * thickness`
still collapses to one column optical depth before GERT (Sec.27), so the
amplitude/thickness degeneracy is unchanged by adding bands -- `thickness`
is shared and FIXED (decided).
CONFIRM how `_build_aerosol_kwargs`/`aerosol_scalars_for` currently scale tau per
band before relying on it (the collapse happens per call, so per-band scaling must
be applied consistently for both bands). XRTM is required whenever height is free
(Sec.30).

**4.5 Noise.** Per-band `geocarb_noise_model(fpa)` sigma, block-diagonal
`Sy_inv`. No cross-band noise correlation assumed.

## 5. Phases and gates

| # | Phase | Gate (must pass before continuing) |
|---|---|---|
| 0 | Freeze decisions (Sec.7) | user sign-off |
| 1 | Data model: band list, `StateSpec` views, per-band albedo rows, stacked `Sy_inv` -- **DONE 2026-09-20**: `geocarb_gert/multiband.py`, gate = `scripts/check_multiband_state.py` (all pass) | unit tests: view round-trips, column embedding |
| 2 | Geometry: shared-eta tiling from BOTH bands' keystone maxima (4.3); per-band row ranges -- **DONE 2026-09-20**: `geocarb_gert/multiband_geometry.py`, gate = `scripts/check_multiband_geometry.py` (all pass; FPA0+FPA2 -> 33 windows) | each band's row range covers every pixel whose keystone trace touches the window; window never narrower than either band's max crossing |
| 3 | Composite forward + Jacobian -- **(a) DONE 2026-09-20**: `gd_multiband_window.py` with ONE band (FPA2 rows 25-37, `--anchor-mode nominal`) reproduces the single-band driver BIT-FOR-BIT (max |dx| = 0, same rms_resid 2.926e-4, 5 iterations); **(b) DONE 2026-09-20**: on the real FPA0+FPA2 window (tile 12, `cover` anchors) the stacked analytic Jacobian matches finite differences to <= 2.2e-4 (albedo columns ~1e-11); each band's data has exactly zero sensitivity to the other band's albedo | (a) N=1 reproduces the existing single-band result exactly (use the `_etaslit` c4 window 25-37 test: 5 iterations, rms_resid 2.9e-4); (b) stacked analytic J vs finite difference |
| 4 | Driver: `--fpas 0,2`, dir tag `_fpa0-2`, sbatch CONF, record bands in pickle -- **first 2-band window DONE 2026-09-20** (`gd_multiband_window.py --fpas 0,2 --tile 12`): converged in 4 iterations, 5051 s at 4 workers, rms_resid FPA0 5.8e-3 / FPA2 3.6e-4 (each at its single-band level), p error 0.028 hPa (prior 0.80), T 0.013 K (2.0), h2o 2.7e-6 (1.1e-3), CO2 0.135 ppm (0.31), albedo O2_A 0.0215 (0.0427), CO2_strong 0.0104 (0.0183). Driver integration (sweep/array/dir tags/sbatch CONF) still to do | 2-band single-window run converges (perfect model, no aerosol) |
| 5 | **DONE 2026-09-21**: widest window (tile 0, 49 rows/band, 16 workers): converged in 5 iterations + a cheap check, 8561 s (2h23m), cgroup memory peak 120.5 GB and FLAT after ~28 min (no per-iteration growth, unlike the single-band XRTM probe); n_free 722; rms_resid FPA0 3.0e-3 / FPA2 5.9e-4. Two-point cost fit (23 rows: 5.6 core-h; 49 rows: 38 core-h) => core-h ~ width^2.5; whole 33-window sweep ~580 core-h (~26 node-hours at 2 tasks/node) | Cost/memory probe on ONE window (cgroup `memory.peak`, s/iteration) | sizing table for tiers before any sweep (ask before submitting) |
| 6 | Pilot, paired arms (no-aerosol + aerosol) on the 10-window subset, FPA0+FPA2 vs the single-band baselines | p/aerosol error in the plume windows drops vs single-band o5y |
| 7 | 4-band | same gates |
| 8 | Rerun ALL single-band experiments on the new baseline (they become `--fpas N`) | results reproduce the intent of Sec.31/32 under fixed code |

Phase 3(a) is the most important gate: the single-band reruns in phase 8
depend on the new code reducing exactly to the current single-band behaviour.

## 6. Cost and memory (rough -- to be measured in phase 5)

Measured single-band: O2-A xrtm width 13, 4 workers ~786 s/iter; width 47, 16
workers ~1150 s/iter, 85.9 GB rising +8.6 GB/iter (cgroup); no-aerosol
single_scatter FPA2 width 13 ~13 min per window total. Expectation for N bands:
RT cost and Jacobian memory grow roughly with the number of bands (sum of
per-band cost), so a 4-band xrtm window is plausibly 3-4x a single-band one;
memory growth per iteration likely scales the same way and would need a `max_iter`
cap or the per-iteration duplication hunted down (Sec.17-19 history). Not
measured. Keep concurrency inside the atmos cap (6 nodes; 300G tasks are 1/node).

## 7. Decisions (user, 2026-09-20)

1. Band pair: **FPA0 + FPA2** (O2-A + CO2_strong).
2. Tiling: must respect **both** bands' keystone maxima (Sec.4.3).
3. Albedo: **independent** per band (own rows, positions, sigma, correlation length).
4. Aerosol: spectral properties from **GERT**; **shared height and width** across
   bands (Sec.4.4).
5. Keystone: real polynomials, known exactly; **no zero-keystone experiments**.

6. `thickness_aerosol`: shared across bands and **fixed** (not retrieved).
7. Albedo prior sigma/correlation length: **reuse each band's single-band values**.
8. Testing: **no aerosols** until they are needed (all of phases 1-5 are no-aerosol
   tests; the aerosol arm comes with the phase-6 pilot).

## 8. Risks

- Memory/time blow-up with bands (see Sec.6) -- probe first.
- Pool timeouts on wide multi-band windows (`_POOL_RESULT_TIMEOUT_S` now 3600 s).
- Silent stale-cache reuse: any change to rendering must bump
  `TRUTH_CACHE_VERSION` (currently 4).
- Directory-name collisions between band sets: tag with the band list.
- Silent failures: the driver exits 0 after recording an error; scan part files
  for an `error` key, not just Slurm state.

## 9. Planned: sbatch load-balancing table (added 2026-09-24, user request)

Motivation: `submit_multiband_sweep.sbatch` requests a fixed 16 CPU / 180G / 5h per task (sized for the
aerosol arm on FPA0+FPA2), which limits packing to ~2 tasks per atmos node. The FPA0+FPA1 / FPA0+FPA3
no-aerosol smoke tests (tile 12, n_free 382) ran 7-8 min at ~4.4 cores average CPU with no OOM, so the
fixed request is very likely over-sized for many configurations. `sacct`/`sstat` MaxRSS is unreliable
here (double-counts pages shared across the anchor-pool workers), so peaks must be measured directly.

Plan:
1. Add an optional per-task peak-memory sampler (summed PSS over the process tree from `/proc`, plus wall
   time and mean CPU) that writes a small JSON next to each result.
2. Measure a grid of runs, one representative tile per cell (mid-slit and a wide/edge tile):
   number of bands (1 / 2 / 4) and which bands; state-vector size (n_free); aerosol free or not
   (single_scatter vs xrtm); number of tiles / tile width (rows and anchors per tile); anchor density;
   g_ratio (1 / 0.5 / 0.25); `--anchor-workers` (CPUs) and `GEOCARB_ANCHOR_POOL_MAXTASKS`.
3. Fit peak memory and wall time against those drivers (expected: anchors x bands x n_free for the
   Jacobian, times workers for the pool) and publish a table of recommended `--cpus-per-task`, `--mem`,
   `--time`, and tasks-per-node, with a safety margin (previous 60G under-request caused OOM-kills).
4. Have the sbatch wrappers (or a helper that prints the `sbatch` line) pick the request from the table
   and the 6-distinct-atmos-node cap, replacing the hand-set per-sweep values.

## 10. Planned: one common tiling for all multi-band sweeps (added 2026-09-25, user request)

Today the tiling is standardized only within a band pairing (single-band vs multi-band of the same pair, via
an exported `--geometry-config`): FPA0+FPA2 = 33 tiles, FPA0+FPA1 = 60, FPA0+FPA3 = 47 (tile width follows
the largest keystone of the bands in the pair). Cross-pairing plots therefore do not share tile edges or
bin positions. Goal: a single geometry config (tile eta ranges + explicit `bin_centers`) used by every
multi-band sweep, so all results share spatial locations. Open choices: which tiling (e.g. all-four-band
max keystone vs the finest pair's), and the rerun cost (noiseless + noise seeds 1-3 for every band set).

## 11. Analytic surface-pressure Jacobian under XRTM with aerosol (assessed and IMPLEMENTED 2026-09-26; see PROJECT_STATUS Sec.40)

Today `jacobians.p_surface_dI_dparam` uses an RT finite difference (2 extra XRTM calls per anchor per iteration) whenever an
aerosol row is present; every other row (amplitude, height via `height_aerosol_dI_dparam_xrtm`, gases, T, H2O, albedo) is
analytic. So p_surface is the ONLY finite-difference column in the 4-band aerosol solve. Measured cost of one 4-band aerosol tile
(tile 12, 744 unknowns, smoke Mie): 4.3 h on 16 CPUs, ~200 GB peak; the same tile without aerosol: 16 min, ~31 GB.

**Physics.** p_l = sigma_l * p_sfc (pure sigma levels), so every layer quantity moves with p_sfc:
1. gas optical depth (pressure, temperature, H2O paths) -- ALREADY analytic: `sum_l K_mol_lay * dtau_l/dp_sfc` (T and H2O
   paths by algebra-only central differences of `atmosphere_from_params`);
2. Rayleigh optical depth: `dtau_ray_l/dp_sfc = tau_ray_l/p_sfc` exactly. Its effect on I needs, per layer, dI/dtau_ray_l =
   K_tau_lay + K_ssa_lay * d(omega_l)/dtau_ray_l + K_coef_ray_lay. The first two are available from existing per-layer arrays
   (omega_l = tau_sca/tau_tot -> d omega/d tau_ray = (1 - omega)/tau_tot); K_coef_ray_lay (the layer phase-function-moment
   response to Rayleigh fraction) is NOT: gert's XRTM coefficient seeds exist only for aerosol (`combined_layer_coef_aer_deriv`,
   slots 2n+n_surf+l);
3. aerosol layer profile: the Gaussian layer fractions `aer_frac_l` (gert `forward_model.py`, the layer-weight function near line 67)
   are functions of the layer pressures, hence of p_sfc: `sum_l K_aer_lay[l] * tau_aer_col * d(aer_frac_l)/dp_sfc`, where the
   fraction derivative is algebra only (analogous to height_aerosol_dI_dparam_xrtm). `K_aer_lay` already includes its own
   coefficient term (`K_coef_aer_lay`, gert commit 1f095b9). height_aerosol is its own state row, so it does not move with p_sfc.

**Work items.**
- gert (private repo): a Rayleigh phase-moment derivative `combined_layer_coef_ray_deriv` (mirror of the aerosol one) and XRTM
  seeds for it. Cheapest form: ONE extra derivative slot seeded with the directional derivative along p_sfc
  (`d chi_l/d tau_ray_l * tau_ray_l/p_sfc` summed as a direction), instead of n_layers extra slots; expose it as a new ForwardResult
  field. Confirm XRTM's per-slot cost so the extra slot is cheaper than the 2 RT calls it replaces.
- geocarb_gert/jacobians.py: `_p_surface_dI_dparam_xrtm_aerosol` composing the three parts; dispatch when `solver == "xrtm"` and an
  aerosol row is present; keep the FD as an option (`GEOCARB_PSURF_FD=1`) for regression.
- Check first whether the existing no-aerosol XRTM path (`_p_surface_dI_dparam_analytic` under XRTM) already omits the Rayleigh
  coefficient term; if so it has a small residual that this work would also fix.
- Validation: cosine and relative error against the current FD over anchors spanning p_sfc 750-1000 hPa (the mountain), thin and
  thick aerosol, all four bands, two-stream and a higher-stream XRTM run; target < 1e-3 relative. Then re-converge tile 12 and
  compare the retrieved state and iteration count.
- Profile before committing: time the RT calls per GN iteration on one tile to measure the FD share (by call count it is 2 of 3 RT
  calls per anchor iteration, i.e. a possible 2-3x; not yet measured).

**Cheaper alternatives / partial wins.** (a) one-sided instead of central FD: 1 extra RT call, O(h) error; (b) a fully general
directional-derivative seed in gert (one slot carrying all per-layer tau/omega/coef perturbations) -- exact but a larger gert change;
(c) more frequent anchor-pool respawns to cut memory (does not reduce time); (d) run a tile subset.
**Risk.** Changes live in gert; the Rayleigh phase-moment term is the only new physics; a mistake there gives a biased Jacobian that
still converges (see the earlier 6-10% p_surface error found with aerosol), so the FD comparison is the gate.

**Status 2026-09-26: implemented as planned** (gert `psurf-analytic-jacobian`, merged; `_p_surface_dI_dparam_xrtm_aerosol`). One directional derivative slot (not n_layers) carries the Rayleigh phase-moment term. Relative L2 error vs the FD 5e-6..1.2e-3; per-anchor Jacobian evaluation ~2.4x faster. `GEOCARB_PSURF_FD=1` keeps the FD available.
