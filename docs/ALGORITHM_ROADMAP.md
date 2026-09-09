# GeoCarb Joint-Block Retrieval — Status Summary and Path to a Production Algorithm

**As of 2026-09-04.** This is a forward-looking companion to
`docs/PROJECT_STATUS.md` (the chronological lab notebook — every number,
plot, and dead end behind the claims here lives there, indexed by
section). This document instead asks: given everything that
investigation has settled, what does a *permanent* retrieval algorithm
look like, and what stands between here and there?

It also effectively supersedes `docs/JOINT_BIN_RETRIEVAL_ATBD.html`, an
earlier Algorithm Theoretical Basis Document scoped to a single-species
(CO2-only), nearest-bin/interpolated estimator — well before the joint
multi-parameter state vector, the footprint-integrated forward model, or
any of the error-mechanism work below existed. That file is left in
place, unedited, rather than rewritten in place, the same way this
project has archived superseded documents before.

---

## 1. What this project has actually established

Three eras of work, each closing a real question rather than just
producing a result:

**Era 1 — geometric distortion mechanisms (archived: `KEYSTONE_SMILE_
BIAS_PLAN.md`, `PROJECT_SUMMARY_AND_NEXT_STEPS.md`).** Identified and
confirmed against real ground-test data which keystone/smile/
rectification mechanisms actually drive retrieval bias, and at what
magnitude. Concluded that native-grid (never rectify-then-retrieve)
processing is required, motivating Era 2.

**Era 2 — joint multi-row/multi-parameter retrieval design (`PROJECT_
STATUS.md` Sec.1-12).** Built the `StateSpec`/joint-block machinery: a
shared along-slit grid, per-row free/frozen control, analytic Jacobians,
sub-bin anomaly support, and (Sec.12) resolved a real bug hunt down to a
genuine mechanism -- a representability gap between the retrieval's own
bin grid and the truth image's resolution, not a persistent degeneracy.
Established that with the gap closed, finer bins behave exactly as
intuition expects (strictly better), and that prior quality stops
mattering once the fit is this well data-constrained.

**Era 3 — retiring spectral interpolation, and a systematic imperfect-
prior investigation (`PROJECT_STATUS.md` Sec.13-16, this week).**
Replaced two-sample spectral blending (confirmed to be the actual
dominant source of the representability-gap artifacts Era 2 chased) with
real per-pixel footprint integration, everywhere -- now the default on
`main`, verified across all 4 FPA bands with each band's own physically
correct target gas. Fixed a genuine Gauss-Newton convergence bug (LM
damping). Then, with the rendering pipeline trustworthy, ran a
systematic diagnostic chain under a deliberately IMPERFECT prior (the
condition a real retrieval is actually in):

- **Frozen-row contamination: ruled out completely.** Free-row results
  are bit-identical whether frozen rows are exact or structurally wrong
  -- the system is close to exactly-determined at `g_ratio=1`, so
  whatever a frozen row gets wrong is fully absorbed by the free rows
  without leaving a trace.
- **Prior-pull (via the Rodgers averaging kernel) explains most of the
  remaining error, unevenly by row.** Nearly all of albedo's error,
  partial for CO2, and frequent-but-harmless for p_surface (whose
  structural prior stays close to truth even when data-unconstrained).
- **A near-zero-albedo (water) region degrades CO2 for a genuinely
  different reason than it degrades p_surface** -- information
  starvation pulling an already-EXACT CO2 prior away from truth, versus
  correcting an unusually bad p_surface prior under worse conditioning
  -- and that damage reaches several rows beyond the region's own
  boundary via keystone smearing (a detector row's spectral samples
  span ~20km along the slit, not a point).
- **With all four currently-modeled state rows free at once** (CO2,
  p_surface, H2O, albedo -- the first genuinely "we don't know
  anything" configuration run this session), real cross-talk appears:
  strongest between CO2 and H2O (corr=0.60, the two gases FPA2 actually
  sees), the expected CO2/p_surface degeneracy (corr=0.49), and albedo
  staying the most decoupled row. That cross-talk correlates with
  keystone strength (higher-G windows show worse H2O and CO2 error),
  consistent with keystone smearing degrading the retrieval's ability to
  SEPARATE correlated absorbers specifically, not just adding noise
  uniformly.
- **A within-window position effect was found that doesn't fit the
  naive story**: CO2 error is worse in window INTERIORS and in FLAT-
  truth regions -- the opposite of p_surface/H2O/albedo, which are worse
  at window edges (albedo dramatically, 14x) and show no gradient
  dependence. Working hypothesis (queued as future work, Sec.16 of
  `PROJECT_STATUS.md`): the spatial-correlation prior has more same-
  window neighbors to smooth against in interiors/flat regions, which
  blurs CO2's genuinely localized structure (hotspots) more than it
  affects the smoother p_surface/albedo fields.

**Net effect of Era 3**: the error budget for an imperfect-prior joint
retrieval is no longer a mystery. Every major error source found so far
is a NAMED, characterized mechanism (prior-pull in low-averaging-kernel
directions; a genuine low-SNR/low-information floor near sharp albedo
contrast, smeared by keystone; real physical cross-talk between
co-sensed gases and the pressure/gas degeneracy; a spatial-prior-driven
position effect within windows) -- not an open-ended "something is
wrong somewhere."

---

## 2. What's still open before this is a production algorithm

Roughly in the order they'd need resolving:

1. **Correlation-length sensitivity study** (`PROJECT_STATUS.md`
   Sec.16, queued not run). Needed to confirm or refute the smoothing-
   prior hypothesis above, and to know how much cross-talk is a free
   design knob (looser prior = more cross-talk resolution but more
   blur) versus fixed by the physics.
2. **A resolved bias-correction strategy for prior-pull.** Sec.14.6
   proposed two candidates (a looser/shorter spatial prior; an
   averaging-kernel-weighted post-hoc correction) -- neither has been
   built or tested yet.
3. **RESOLVED: temperature (`t_offset_k`) is now a retrievable state row**
   (`PROJECT_STATUS.md` Sec.10, 2026-09-07). A uniform additive shift to
   the standard-atmosphere temperature profile, ±3K synoptic-sinusoid
   truth (same period as `p_surface_hpa`'s own synoptic term, distinct
   phase), flat 0K structural prior. Its analytic Jacobian
   (`geocarb_gert.jacobians.t_offset_dI_dparam`) needed genuinely zero
   finite differences, exactly as predicted -- just `dtau_mol_dT_lay_
   hires` x `K_mol_lay_hires` with no chain rule. `kind="absolute"` is
   now wired end-to-end for the first time (the second open item this
   entry flagged) -- found and fixed a real gap along the way:
   `jacobians.linearize` hardcoded the `kind="scale"` chain-rule factor
   and explicitly raised on any other kind, so `kind="absolute"` was
   advertised (module docstring) but not actually reachable through the
   analytic path. Fixed generically (any future absolute-valued row
   benefits, not just this one). Verified four ways: the `x0()`/prior
   round-trip, an analytic-vs-FD cross-check (correct once the FD step
   was rescaled from the other rows' O(1)-multiplier convention to a
   physically appropriate Kelvin-sized step -- rel L2 ~2e-5, matching
   `p_surface_hpa`'s own documented gert-internal precision floor), a
   smoke test (converged to within 1e-4 K of the true value in a fully
   representable config), and the standing forward-model-agreement check.
   **Accepted side effect** (user's explicit call): since `PRIOR_FIELD_
   SETS["exact"]`/`["structural"]` point at `STATE_FIELDS`/`STATE_FIELDS_
   PRIOR` directly, every driver script that builds its own prior-fields
   registry by comprehending over those dicts now picks up `t_offset_k`
   automatically (frozen at its real ±3K truth) if rerun from now on --
   intended, matching the file's own "no quantity is privileged" design,
   not a bug. Not yet run: any of the prior-pull/keystone-floor/defocus
   diagnostics this session already put every other row through.
4. **Multi-band bin placement** (`PROJECT_STATUS.md` Sec.15). Each
   band's window/bin tiling is chosen independently today (`--fpa`
   refuses more than one band outright) -- 58/66/63/78 natural windows
   for FPA2/1/0/3 at identical settings, driven by each band's own
   keystone curve. A real multi-band retrieval needs bins on ONE shared
   spatial grid across bands (a sounding is one physical location with
   one true state), balancing per-band spectral-point density against
   per-band keystone strength -- not designed yet.
5. **Per-band free-parameter sets.** Confirmed this session: FPA0 (O2_A)
   has zero CO2 sensitivity and should retrieve `p_surface_hpa`; FPA3
   (CH4_CO) should retrieve `ch4_ppb`/`co_ppb`; only FPA1/FPA2 (CO2_weak/
   strong) see CO2 at all. A production multi-band config needs this
   mapping made explicit and enforced (e.g. validated in the sweep
   script's argument parsing), not left to whoever configures a run to
   get right by hand -- the FPA0/FPA3-with-`--free co2_ppm` mistake
   this session made once already.
6. **Partially resolved: the frozen-row representability gap under
   genuine (non-representable) truth is real, and albedo's own sub-bin
   texture is A major driver -- but not the only one**
   (archived `PROJECT_STATUS.md` Sec.16.2, follow-up in the current
   `PROJECT_STATUS.md` Sec.1). Freezing ch4/co/h2o/albedo at their exact
   value AT EACH BIN'S OWN POSITION against genuine Mode-1 dense (500m)
   truth produced co2/p_surface errors 100-250x worse than the
   equivalent representable Mode-2 case. Confirmed this wasn't a
   constant-vs-varying bug (both truth and the frozen assumption used
   the real spatially-varying albedo field). A new `--surface-positions
   {shared,anchor}` flag let albedo be frozen on its own fine ANCHOR
   grid instead of sharing the atmosphere rows' coarser `bin_centers`
   grid -- this cut rms error ~9-11x (co2 47.3->4.92 ppm, p_surface
   26.1->2.34 hPa), confirming albedo's texture as a major driver, but
   a substantial gap remains (~25x worse than the Mode-2 ceiling in rms,
   ~40-160x at the max). Remaining candidates, not yet tested: albedo
   texture finer than even this retrieval's own `anchor_density`; the
   still-bin_centers-frozen ch4/co/h2o rows; or another mechanism
   entirely. This should inform whether the current `anchor_density`/
   `g_ratio` production defaults are adequate once truth is allowed to
   be genuinely non-representable, not just for free rows but for
   frozen ones too -- still open, now with a much narrower remaining
   gap to explain.

---

## 3. Recommended shape of a "permanent" single-band algorithm

Based on everything in Sec.1, here is what a settled (not "one more
diagnostic run") FPA2 retrieval should look like, and why:

- **Forward model**: `render_at_anchors` + `footprint_average_scene`
  unconditionally. Spectral blending (`build_lookup_radiance`/
  `gd_render.image`) is deprecated, not a configuration choice.
- **Free parameters**: `co2_ppm`, `p_surface_hpa`, `h2o_surface_vmr`,
  `albedo` together (Experiment A's configuration) -- this is the
  realistic "we don't know anything" state, and cross-talk between
  these four is now characterized rather than a surprise. `ch4_ppb`/
  `co_ppb` frozen (FPA2 has no sensitivity to either).
- **Bin/anchor resolution**: `g_ratio=0.5` as the new working default
  (`PROJECT_STATUS.md` Sec.6, 2026-09-05/06) -- finer `g_ratio` (0.5,
  0.25) gives real, consistent accuracy improvement over `g_ratio=1` at
  every window tested (`p_surface` especially, up to ~23x median-error
  reduction end to end), confirming Sec.12's original "finer is
  strictly better" finding still holds under genuine non-representable
  truth. `g_ratio=0.25` is more accurate still but costs ~6-9x
  `g_ratio=1`'s wall-clock per window (cost scales ~G^1.24) -- the
  single widest window alone took ~5.3 hours, making a full 58-window
  production sweep at 0.25 impractical without further work (see Sec.6
  for the full per-window table). `g_ratio=0.5` costs only ~2-3.5x and
  captures most of the benefit, the better production trade-off for
  now. `anchor_density>=4` unchanged.
- **Prior**: structural (imperfect) by construction -- exact-prior
  results are a ceiling/diagnostic tool (Sec.12.7's own conclusion),
  never the production configuration, since a real retrieval never
  starts from truth.
- **Convergence**: LM-damped Gauss-Newton (Sec.13.3) with the early-
  exit check -- already the default.
- **Known, accepted error floor**: prior-pull in low-averaging-kernel
  directions (worst for albedo), plus a bounded, keystone-smeared
  degradation near strong along-slit albedo contrast (the "water
  region" mechanism). Both are now real, quantified, *expected*
  behaviors of this algorithm rather than bugs to keep chasing -- unless
  item 2 above (a bias-correction strategy) is built and shown to
  reduce them further.
- **Verification convention going forward** (established this session,
  should become standard for any future change to this pipeline): a
  forward-render smoke test AND a full-retrieval smoke test on every
  FPA band actually affected, each using that band's own physically
  correct target gas -- not just "doesn't crash" on the band the change
  was developed against.

---

## 4. Prioritized next steps

1. **RESOLVED: the residual ~25x at rows 1013-1023 is extreme keystone
   smearing, the same mechanism as the archived Sec.14.4 water-region
   finding -- not a representability gap at all** (`PROJECT_STATUS.md`
   Sec.1-5). Chain: giving albedo its true value at every anchor closed
   ~90% of the original 100-250x gap (Sec.1); neither finer albedo
   resolution (Sec.3) nor freezing ch4/co/h2o at anchor resolution too
   (Sec.4) moved the remaining residual at all; checking the OTHER slit
   edge (Sec.5) found `rows_crossed` (keystone-smearing strength) is
   ~0.06-0.27 near the left edge (FPA2's own keystone-null, near row
   25) but ~10.2-10.4 near the right edge -- a ~40x difference,
   directly explaining why rows 1013-1023 specifically (maximally far
   from the null point) behaves so much worse than a typical window,
   independent of any frozen-row assumption. Practical implication:
   windows near the slit edge opposite a band's own keystone-null point
   should be EXPECTED to show elevated error under any configuration --
   an intrinsic instrument-geometry floor for that location, not a
   fixable retrieval-configuration problem. **Independently confirmed,
   more strongly than expected** (`PROJECT_STATUS.md` Sec.7,
   2026-09-06): reversing the truth spatially while leaving geometry
   fixed left rows 1013-1023 comparably bad -- actually WORSE (co2 rms
   7.6 -> 32.1 ppm) -- ruling out "the original truth happened to be
   easy there" entirely. The same test showed the OPPOSITE signature at
   a low-keystone window (rows 189-197: error dropped 3x under
   reversal), confirming that away from the keystone floor, error IS
   genuinely truth-content-sensitive -- cleanly separating the two
   effects rather than leaving them conflated. **Follow-up** (`PROJECT_
   STATUS.md` Sec.8, 2026-09-06): reran the reversed-truth control at
   `g_ratio in {0.5, 0.25}` -- finer resolution shrinks the floor
   substantially and monotonically at the extreme-keystone window
   (co2 rms 32.1 -> 4.7 -> 2.4 ppm end to end), enough that at
   `g_ratio=0.25` its error is actually competitive with, or better
   than, the low-keystone window's own `g_ratio=1` numbers. So the
   keystone effect is a floor FOR A GIVEN resolution, not an absolute
   resolution-independent one -- `g_ratio=0.5`/`0.25` (Sec.6's
   production-default recommendation) is a real, substantial mitigation
   for the worst-affected rows specifically, not just a broad average
   improvement. Not yet checked: where each OTHER FPA band's own
   keystone-null sits (clocking offsets differ per band per the
   project's Era-1 findings), which would move this "worst edge" to a
   different location for FPA0/1/3; and the same reversed-truth/finer-
   g_ratio check at the widest window (rows 968-1012) as a second,
   independent high-keystone data point.
2. Correlation-length sensitivity study (Sec.2 item 1) -- cheapest next
   diagnostic, and gates whether item 3 (a bias-correction strategy) is
   even the right lever to pull.
3. Decide and build a bias-correction strategy for prior-pull (Sec.2
   item 2), informed by 2.
4. Temperature state parameter (Sec.2 item 3) -- implementation plan
   already exists; mainly needs the two open design decisions resolved
   (truth field shape, `kind="absolute"` verification) before building.
5. Multi-band bin placement + per-band free-parameter enforcement
   (Sec.2 items 4-5) -- the largest remaining piece of new design work,
   and the one that actually turns this into a GeoCarb-wide (not
   FPA2-only) production algorithm.
6. **Defocus (wide-PSF) experiments: the retrieval largely self-corrects
   for an unmodeled defocus, EXCEPT at the extreme-keystone floor**
   (`PROJECT_STATUS.md` Sec.9, 2026-09-06, user: simulating the
   telescope's focal-adjustment mechanism, spatial-only). Decoupled the
   TRUTH's along-slit PSF from what the RETRIEVAL's own forward model
   assumes (new `--retrieval-psf-fwhm-px` flag, `spatial_psf_fwhm_px`
   threaded through `render_at_anchors`/`build_forward_state`). At
   low/moderate-keystone windows, a completely wrong retrieval PSF
   assumption (mismatched vs. matched) makes little to no difference --
   at one cell the two were identical to 5 significant figures -- the
   free parameters (especially free albedo) absorb the mismatch almost
   perfectly. At the extreme-keystone far edge (rows 1013-1023, the
   same location Sec.5/7/8 already flagged), the mismatch compounds the
   existing keystone floor instead (co2 rms +14-31% worse mismatched
   vs. matched, growing with defocus severity). Practical implication:
   the focal-adjustment mechanism's calibration matters most exactly
   where retrieval is already most fragile, not uniformly across the
   slit. **Important correction** (`PROJECT_STATUS.md` Sec.9a,
   2026-09-06/07): the "even MATCHED defocus gets worse with wider
   FWHM" half of this finding turned out to be a tiling artifact, not a
   real property of defocus. Root cause: free parameters live only on a
   window's own local `bin_centers`, but the PSF blur needs a padded
   render region scaling with FWHM (`default_pad_for_psf`) --
   `_row_interp1d` CLAMPS the free state into that padding rather than
   extrapolating, so a narrow window (Sec.9's 9-11-row fast subset)
   sees pad/width reach ~200% at FWHM=8px. Rerunning the SAME matched-
   defocus config at a much wider window (41 rows, pad/width ~54%)
   removed the effect entirely -- error went from a sharp edge-U-shape
   to a flat, near-constant residual across every bin, and aggregate
   error IMPROVED versus nominal PSF rather than worsening. The
   MISMATCHED-vs-matched finding (real degradation at the extreme-
   keystone edge) is unaffected -- that comparison holds window
   width/pad fixed between the two configs being compared. Practical
   takeaway: any future sweep axis that changes `pad` (PSF FWHM
   included) must use production-representative window widths, or
   narrow fast-subset windows will systematically overstate the effect.
   Not yet run: the same matrix at the widest window (968-1012); finer
   g_ratio combined with defocus; a middle-ground FWHM to locate where
   matched/mismatched divergence begins; the wide-window
   matched-vs-mismatched comparison (does the keystone-edge mismatch
   penalty also shrink at wide windows, or is it genuinely
   location-driven).
7. **Aerosol (`tau_aerosol`, `height_aerosol`) added as retrievable
   state -- NOT YET converging in a real joint retrieval, root cause
   partially found** (`PROJECT_STATUS.md` Sec.11, 2026-09-08) --
   infrastructure prerequisite for coupling FPA0 with FPA2/FPA3 (user's
   own stated next goal: "get a column average that is responsive to
   aerosols and surface pressure errors"). `tau_aerosol`'s analytic
   Jacobian is fully validated in isolation (reuses the existing
   `SURFACE_ROW_JACOBIAN` machinery unchanged). `height_aerosol` needed
   a genuine RT-level finite difference instead and surfaced a real,
   structural finding: `gert.ForwardModel.run`'s own aerosol-layer-
   placement logic uses a hard layer-boundary threshold, not a smooth
   function of height, so `height_aerosol`'s own sensitivity is
   genuinely non-smooth -- accepted as a documented limitation (user's
   call), and this same threshold was later found to also corrupt
   `p_surface_hpa`'s own Jacobian once aerosol is present (below).

   **Four real bugs found and fixed, none specific to this project's
   own retrieval logic** (a smoke test kept failing to converge --
   `resid_hires_rms` ~0.10-0.15 instead of the ~1e-5 every other row's
   own smoke test reached -- until these were found): (1)
   `gd_jacobian_validate.py`'s own duplicated FD-reference forward model
   silently omitted aerosol physics entirely; (2) `P_aerosol` (the
   scattering phase function) had never been computed anywhere in this
   whole codebase, making the aerosol single-scatter term identically
   zero regardless of height; (3) `p_surface_dI_dparam`'s existing
   (pre-aerosol) analytic Jacobian is missing a real cross-term through
   the SAME hard pressure-grid/aerosol-mask threshold -- fixed via a
   conditional RT-level-FD fallback, zero behavior change when aerosol
   is absent; (4) **the actual truth-rendering path** (`gd_test.py::
   _band_setup`'s own THIRD independent inline spectrum-builder, used
   by `--prior-fields exact` and every standard run) **never threaded
   aerosol through at all** -- a genuine truth-vs-model physics
   mismatch, not a Jacobian issue, and very likely why `tau_aerosol`
   was retrieving unphysical negative values.

   Fixing all four improved things measurably (`resid_hires_rms` 0.147
   -> 0.101 from bug 4 alone) but **did not fully resolve convergence**
   -- at least one more issue remains, not yet found. **The general
   pattern this surfaced, worth fixing at the root**: this codebase has
   at least three independent, hand-duplicated "build a spectrum from
   state parameters" functions (`_make_state_spectrum`, `spectrum_jac`,
   `_band_setup`'s own inline one), each needing the same aerosol
   threading applied separately with no single choke point guaranteeing
   it by construction -- a real refactor target. Also found: `_band_
   setup_cached`'s own truth-image cache key has no dependence on the
   field functions' own content, so a stale (pre-fix) cached truth was
   silently served as a "hit" with no error -- worked around by moving
   the 2 stale entries aside, not fixed at the root. **Explicitly NOT
   resolved**: getting aerosol to converge cleanly in a real joint
   retrieval is its own follow-up task, not something to build the
   FPA0+FPA2/FPA3 coupling work on top of yet.
