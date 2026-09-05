# GeoCarb Keystone/Smile/Dispersion Bias Study — Project Summary and Next Steps

> **Archived 2026-08-19.** Superseded by `docs/PROJECT_STATUS.md`. This
> snapshot's §6 "next steps" were not the path actually taken — the project
> pivoted to joint multi-row retrieval (`JOINT_ROW_INVERSION_PLAN.md`,
> also archived) instead of the covariate-regression bias correction sketched
> here. §1-4's findings are still accurate and are summarized in the new
> document; kept in place, unedited, since it's cited elsewhere by section.

**As of:** 2026-07-28. Written as a handoff summary before a break in active work.
For full chronological detail and every number/plot behind the claims below, see
`KEYSTONE_SMILE_BIAS_PLAN.md` (the running lab notebook this summarizes — search
it by section number, e.g. "§9l", to find the full writeup).

---

## 1. What this project is

GeoCarb's real ground-test calibration report (`keystone_report.pdf`) documents
retrieval anomalies (non-convergence, residual striping) tied to keystone/smile
geometric distortion and the rectify-then-retrieve L1B pipeline. This project
builds a synthetic simulator (`geocarb_gert`, on top of the `gert` forward-model/
retrieval library) that reproduces the *real* GD (geometric distortion)
polynomials, detector geometry, and PSF — then uses it to isolate *which*
mechanisms actually drive retrieval bias, at what magnitude, and whether a joint
multi-band retrieval (the real GeoCarb L2 approach) helps.

The method throughout: build a specific, falsifiable hypothesis tied to a
concrete real-data symptom (§10's evidence table), test it in simulation, confirm
or rule it out, then move to the next candidate mechanism.

---

## 2. What's established (confirmed mechanisms, single-band)

All of the following are simulation-confirmed; real-data confirmation status is
noted per item (see §5 below — this is the biggest open gap).

1. **Row-crossing / pixel-grid aliasing** — the leading single-band bias
   mechanism. A detector row's spectrum is an average over up to ~10 true
   along-slit rows near the slit edges (fewer near center). Matches real
   ground-test non-convergence patterns directly (`keystone_report.pdf` §4.8.2).
   [§9b — confirmed against real data]
2. **Clocking** is a per-band *offset* of the keystone curve (FPA2's zero-point
   is at row 25 of 1024, nearly at the slit edge), not a separate mechanism.
   [§9c — confirmed against real data]
3. **FPA2 has an independent calibration-coverage gap** (insufficient laser
   wavelength coverage during ground test), compounding the above.
   [§9d — confirmed against real data]
4. **PSF×smile coupling creates an irreducible, order-independent CO2 bias
   (up to ~-5.9 ppm) even on a spatially uniform scene** — not yet checked
   against real data. A concrete, falsifiable real-data prediction.
   [§9h — confirmed in simulation only]
5. **Rectification (regrid-then-retrieve) bias dominates over raw geometric
   distortion** — tens-of-ppm, roughly slit-position-independent, doesn't
   vanish at the smile-null row the way PSF×smile bias does, ~10-14% outright
   non-convergence at full row density. This is the single biggest simulated
   bias source found. Not yet checked against real data — a clean,
   falsifiable prediction (real data should show far higher non-convergence
   on rectified/regridded processing than on native-grid processing).
   [§9l/§9m — confirmed in simulation only, biggest open real-data check]
6. **Barcode/scene-transition bias** (keystone heterogeneity) is real but
   narrowly localized — a sharp spike within ~4-10 rows of a genuine scene
   boundary, zero elsewhere. [§9j — confirmed in simulation only]
7. **Dispersion order matters enormously for reported chi2, independent of
   fit quality** — floating a low-order wavelength-dispersion correction
   (order=2) lets a pipeline absorb small residuals that a fixed-calibration
   pipeline (order=0) cannot. Originally read as "native fits much better
   than undistorted" (§9x); **this session found the majority of that gap is
   actually a numerical artifact, not real fit quality — see §4 below.**

---

## 3. What's established (multi-band / joint retrieval)

1. **Cross-band co-registration is solved.** Real slit angle `s` [deg] — never
   normalized `eta = s/s_max(fpa)`, which differs per FPA — is the correct
   shared coordinate across bands, confirmed by the actual ground-test
   calibration methodology (co-aligned lasers hitting one shared point on the
   scanning mirror). `nearest_row_pairing_multi()` (any N>=2 bands) pairs rows
   with <=0.5-row error, bounded and does not compound as more bands are added.
   [§11b/§11f]
2. **Joint retrieval measurably helps.** FPA0(O2-A)+FPA2(strong CO2), no
   aerosol: joint retrieval reduces CO2 retrieval std by ~4.4x vs. single-band,
   by breaking the H2O/surface-pressure degeneracy — the real-world rationale
   for GeoCarb's multi-band L2 approach, now demonstrated in this simulator.
   [§11g]
3. **The joint pipeline generalizes to any >=2 bands** (`--fpas 0,1,2,3`-style),
   smoke-tested at 3 bands; no further code changes needed for a future full
   4-band run. [§11j]
4. **Barcode joint-failure mode diagnosed**: narrow chi2-dominating spikes near
   bar boundaries (same mechanism as single-band §9j) hit a *wider* row
   fraction jointly, since each band has its own independent keystone geometry
   — not a bug, an expected, understood interaction. [§11i]
5. **Rectified pipeline inherits (and compounds) the single-band rectification
   bias** in the joint case too, confirming §9l/§9m's prediction extends to
   cross-band registration. [§11i]

---

## 4. This session's headline finding: the dispersion-order convolution artifact

**Symptom:** the "undistorted" pipeline (zero-distortion baseline, no
dispersion correction by original design) showed chi2 and gas-bias scatter
far worse than "native" — confusing, since undistorted should be the
*cleanest* case.

**Root cause, confirmed to the source line:** `gert/forward_model.py` only
uses exact (sub-grid) ILS-kernel centering when dispersion parameters are
present in the state vector (`wn_centers is not None` triggers
`gert.instrument.ILS.convolve`'s `exact_center=True`); absent dispersion, it
silently falls back to snapping each channel center to the nearest hi-res
grid point (`exact_center=False`, described in `gert`'s own docstring as
legacy behavior). This project's truth-rendering path
(`geocarb_gert.gd_render._diagonal_ils_convolve`) always uses exact centering.
So any retrieval that never floats dispersion (order=0) is structurally stuck
comparing against a mismatched convolution convention — a small, real,
near-row-independent chi2 floor with nothing to do with atmospheric retrieval
quality. Verified directly: evaluating the forward model at the *true* prior
state (no optimizer involved) reproduces the identical-magnitude mismatch for
both native and undistorted alike (matched to 4 significant figures).

**Fix:** float dispersion (order>=1) in every pipeline, including
"undistorted" — not because it now has real calibration error to correct
(it doesn't; coefficients are expected to sit near zero), but purely to
trigger the matching exact-center convolution path. Confirmed empirically via
a dedicated diagnostic (`scripts/gd_joint_undistorted_dispersion_diag.py`):
this collapses undistorted's chi2 to native's level or better, across uniform,
uniform+noise, and realistic scenes.

**Applied to:** `gd_band_stress_test.py` (single-band, all 4 FPAs) and
`gd_joint_band_test.py` (multi-band), plus all five associated plotting
scripts. Full details, evidence tables, and the explicit scope note about
what was and wasn't changed: **§11k**.

**Important open thread this creates** (see §6.1 below): §9x had earlier
attributed *some* of native's dispersion-driven chi2 advantage to a real,
keystone-correlated wavelength-registration error (small, row-dependent,
near-zero at the smile-slope-null row). That reasoning isn't necessarily
wrong — it may be a smaller, real effect riding on top of the larger
numerical artifact this session found. **Disentangling the two is a concrete,
well-motivated first analysis once the current reruns land.**

---

## 5. In-flight right now (check status after the break)

All jobs below apply the §4 fix and supersede all prior results for the same
FPA/band-pair (old results and their old-format plots were already deleted).

| Battery | Jobs | Scope |
|---|---|---|
| Single-band, all 4 FPAs | `1604089, 1604090, 1604097-1604100` (24 array tasks) | `gd_band_stress_test.py`, full 3-scene x 2-noise x 4-FPA matrix |
| FPA0+FPA3 joint | `1603981-1603986` | native+undistorted only (rectified unaffected by fix, not rerun) |
| FPA0+FPA2 joint | `1604101-1604106` | full 3 pipelines |
| FPA0+FPA1 joint | `1604107-1604112` | full 3 pipelines |

Check with `module load slurm/24.05.0 && squeue -u $USER` (or add
`/gpfs/fs1/sfw3/rhel9-x86_64/slurm/24.05.0/bin` to `PATH` directly — `squeue`/
`sbatch` are not reliably on `PATH` by default in fresh shells on this cluster,
unlike `git`, which needs `module load git/2.50.1` for the same reason).

**Once these land**, regenerate plots:
```
python scripts/gd_band_stress_test_plot.py            # per FPA, run 4x
python scripts/gd_band_stress_test_plot_grid.py
python scripts/gd_band_stress_test_plot_residual_spectra.py
python scripts/gd_joint_band_plot.py --fpas 0,1   # and --fpas 0,2 / 0,3
python scripts/gd_joint_band_plot_residual_spectra.py --fpas 0,1   # (slow, re-renders)
```

---

## 6. Concrete next steps

### 6.1 Immediate — disentangle real vs. numerical dispersion signal (cheap, high-value)

Once the single-band FPA1 rerun lands, redo §9x's row-0/512/1023 disp_a0
comparison table, now that undistorted also floats dispersion. If undistorted's
`disp_a0` vs. row now tracks *the same small, row-dependent pattern* native's
already does (near-zero at slit center, growing toward edges, tracking §9w's
row-crossing) — that confirms §9x's real-registration-error story as a smaller
effect layered on the numerical artifact this session fixed. If undistorted's
`disp_a0` stays flat/near-zero everywhere while native's still shows the old
row-dependent pattern, the row-dependence was itself an artifact of something
else (worth another look). Either way this is a same-day analysis on data
that will already exist post-rerun — no new runs needed.

### 6.2 Foundational — real-data validation (the biggest gap in the whole project)

**Almost nothing in §2/§3 above has been checked against real data yet** — §10's
evidence table is explicit about this; most rows say "confirmed in simulation,
not yet checked against real data." Before investing heavily in a bias-correction
scheme tuned to these simulated mechanisms, it's worth confirming the mechanisms
themselves show up in real ground-test or on-orbit data:
- Does real data's non-convergence rate on rectified/regridded processing
  actually exceed native-grid processing by the ~10%+ margin simulation predicts
  (§9m)? This is the single highest-value check — rectification bias is the
  largest mechanism found.
- Does real spectral residual (even from non-converged retrievals) show the
  line-locked (not noise-like) structure §9n found for rectification error, vs.
  the smooth PSF×smile-coupling shape (§9h)? These have different implied
  bias-correction strategies (§10's "Still open" checklist has the full list).
- Does keystone-heterogeneity contamination in real barcode-like scenes (coastal
  transitions, field boundaries) show the same narrow (~4-10 row) footprint §9j
  predicts?

If real data doesn't show a predicted signature, that's just as valuable a
result as confirming one — it re-scopes which mechanism(s) actually matter for
correction.

### 6.3 Bias correction via retrieved covariates (the user's specific ask)

Every retrieval in this study already saves its **full state vector**
(`_state_full`: every element's retrieved value, prior, posterior 1-sigma) and
**post-fit residual**, per row — a persistent-memory rule established this
project specifically so this analysis is always possible without a rerun. Two
concrete starting points:

1. **Regress each pipeline's gas bias against its own co-retrieved nuisance
   parameters** (T_offset, albedo, albedo_slope, and — especially informative
   per §9x/§4 above — the dispersion coefficients, which directly track
   keystone/row-crossing amount). If dispersion coefficients are a good proxy
   for local geometric-distortion severity, a simple linear (or low-order
   polynomial) bias-correction term `bias_corrected = bias_raw - f(disp_a0,
   disp_a1, disp_a2, ...)` fit against the *already-existing* §9v/9w and
   §11g-11i battery results could measurably reduce along-slit bias without
   any new physics or new retrieval runs — worth trying first, cheapest path
   to a result.
2. **Use real slit angle `s` itself (or distance from each FPA's own
   keystone-null/smile-slope-null row, already tabulated per FPA in §9i/§11i)
   as an explicit covariate.** Since the per-band null-row structure is fully
   characterized (table in §11i), a position-aware correction — rather than a
   single global bias number — is a natural next step, and directly testable
   against the existing along-slit sweep data.

### 6.4 New nuisance parameters to explore in the retrieval itself

1. **A spatial/spectral surface-albedo basis** beyond the current constant +
   linear-slope albedo model — `gert.forward_model`'s `surface_basis` /
   `SurfaceBasis` machinery already exists (see `forward_model.py`'s
   `K_albedo_hires`/basis-correction code path referenced in §4's fix
   investigation) but hasn't been exercised in this study. Worth checking
   whether a higher-order albedo basis absorbs any of the rectification-
   interpolation residual (§9l/§9m), which §9n found has real spectral
   structure a simple linear albedo slope can't represent.
2. **Finish the calibration-mismatch experiment (§12).** Built and
   smoke-tested for *wavelength*-only mismatch (§12g); *slit-position*
   mismatch needs `wavelength_slit_to_xy_assumed()` and a `rectify_assumed()`
   counterpart — not yet built. This matters directly for bias correction:
   §12c's finding was that wavelength-calibration error is partially
   self-correcting (a dispersion nuisance parameter can absorb it) while
   slit-position/keystone error has **no equivalent nuisance parameter
   anywhere in `StateVector.gas_scaling()` today** — i.e., a structurally
   uncorrectable bias source under the current state-vector design, unless a
   new nuisance parameter is added specifically for it. Finishing this
   experiment would show exactly how much bias a real (imperfect) ground-test
   calibration would leave *after* whatever correction scheme comes out of
   6.3, which no amount of covariate-regression on perfect-calibration
   synthetic data can show on its own.
3. **PSF-area-weighted cross-band co-registration** (§11c) as an alternative
   to nearest-row pairing — flagged as a possible refinement if nearest-row's
   <=0.5-row discretization error turns out to matter in practice; not built,
   not yet shown to be necessary.

### 6.5 Aerosol (deliberately deferred throughout, still open)

Every joint-retrieval result in §3 is explicitly aerosol-free (§11's own
motivation: aerosol-CO2 degeneracy would compound with whatever's found here,
and it costs much more compute). `ForwardModel` already has an aerosol
Jacobian path (`K_aer_lay`) ready to use. §11d items 4/6 lay out the plan:
extend to a shared atmosphere with an aerosol layer, then compare CO2-only vs.
CO2+O2A+aerosol joint retrieval — with a specific, stated working hypothesis
(aerosol shouldn't touch the rectification-interpolation bias, since that's a
spectral-registration artifact independent of photon path length) worth
confirming rather than assuming. Any bias-correction scheme developed
aerosol-free (6.3/6.4) should be re-checked once aerosol is added, since real
data has both effects simultaneously.

---

## 7. Where things live (quick orientation)

- **`KEYSTONE_SMILE_BIAS_PLAN.md`** — the full chronological record; this
  summary's source of truth. Search by "§N" section number.
- **`geocarb_gert/`** — the simulator package (`gd_render.py` truth rendering,
  `gd_polynomials.py` real GD polynomials, `cross_band.py` co-registration,
  `along_slit_scene.py` truth-atmosphere construction).
- **`scripts/gd_band_stress_test*.py`** — single-band battery + 4 plotting
  scripts.
- **`scripts/gd_joint_band_test*.py`** — multi-band battery (any N>=2 FPAs)
  + 2 plotting scripts.
- **`scripts/gd_joint_undistorted_dispersion_diag.py`** — the diagnostic that
  found/confirmed §4's fix; not part of the regular battery, kept as a
  standing record.
- **`results/*.pkl`** — not git-tracked (gitignored); regenerate via the
  battery scripts, never hand-edit.
- **`plots/*.png`/`*.pdf`** — git-tracked; regenerated by the `*_plot*.py`
  scripts reading from `results/`.
