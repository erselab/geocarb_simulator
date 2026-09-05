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
3. **Temperature as a state parameter.** Not yet a retrievable quantity
   at all (fixed via the standard atmosphere). A concrete implementation
   plan exists (this session, 2026-09-04): GERT already provides the
   exact per-layer `dtau_mol_dT_lay_hires` term `p_surface_hpa`'s own
   Jacobian already uses, so a uniform `T_offset_K` row's analytic
   Jacobian needs no finite differences at all -- simpler than every
   existing row but albedo. Two things need deciding before building
   it: the truth field's spatial structure (a synoptic sinusoid,
   analogous to `p_surface_hpa`'s, is the natural default), and whether
   `kind="absolute"` (vs. the `kind="scale"` every current row uses)
   actually works end-to-end in `StateSpec`/`gauss_newton_state` --
   flagged, not yet checked.
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
- **Bin/anchor resolution**: `g_ratio=1`, `anchor_density>=4` -- Sec.12
  showed finer is strictly better once truth and bin grid are matched,
  and item 6 above will confirm whether `ad4` specifically is
  sufficient against a genuinely non-representable truth or whether
  `ad16` (or finer) is needed for production.
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

1. **Frozen-row representability RULED OUT entirely; the remaining
   ~25x looks like a slit-edge artifact, not a resolution problem**
   (`PROJECT_STATUS.md` Sec.1-4). Giving albedo its true value at every
   anchor closed ~90% of the original 100-250x gap, but a residual
   remained concentrated at one window (rows 1013-1023, 110 ppm co2
   max). Neither finer albedo resolution (ad16, Sec.3) nor freezing
   ch4/co/h2o at anchor resolution too (Sec.4, using the new general
   `row_positions`/`--frozen-atmosphere-positions` mechanism) moved
   that number at all (110.5 -> 117.5 -> 110.5, all within noise).
   Frozen-row representability, at any tested resolution for any tested
   row, is not the mechanism. New leading hypothesis: rows 1013-1023 is
   the LAST window on the whole 1024-row detector -- a slit-edge
   artifact (PAD truncation, extreme keystone at the boundary), not a
   representability gap at all. Next concrete test: check whether the
   FIRST window (rows 0-8) shows the same pathology -- if both slit
   edges are anomalously bad, that's a cheap, strong confirmation before
   chasing the specific edge mechanism.
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
