# GeoCarb Simulator — Project Status

**As of 2026-09-05.** This is a fresh start for the project's next
phase: a permanent, production-shaped joint-block retrieval algorithm,
built on top of the footprint-integrated forward model that is now
`main`'s default.

For the full chronological record of how the project got here —
geometric-distortion mechanism-hunting, the joint multi-row/multi-
parameter retrieval design, the representability-gap bug hunt, retiring
spectral-blend rendering, and the systematic imperfect-prior diagnostic
chain (frozen-row contamination, prior-pull via the averaging kernel,
the water-region/keystone-smearing mechanism, cross-talk between free
state rows, and the frozen-row representability gap under genuine
truth) — see `archive/initial_build/docs/PROJECT_STATUS.md`, archived
in place, unedited, on 2026-09-05.

**`docs/ALGORITHM_ROADMAP.md`** is the standing summary of what that
investigation established and what's still open — read that first for
the current state of the algorithm and the prioritized next steps
before picking up new work here.

This file starts fresh at Section 1 for whatever comes next.

---

## 1. Frozen-albedo representability gap: giving albedo its true value at every anchor closes most, not all, of Sec.16.2's blowup (2026-09-05)

Direct follow-up to `archive/initial_build/docs/PROJECT_STATUS.md`
Sec.16.2 (the initial-build-phase record, not edited further): under
genuine Mode-1 dense (500m) truth, freezing `ch4_ppb`/`co_ppb`/
`h2o_surface_vmr`/`albedo` at their exact value AT EACH BIN CENTER gave
co2/p_surface errors 100-250x worse than the equivalent representable
(Mode-2) case. Working explanation there: a frozen row's own piecewise-
linear bin-to-bin reconstruction can't capture real structure below bin
spacing -- most likely albedo's own fine texture -- even when each
bin's own node value is exact.

First confirmed directly (2026-09-05) that this wasn't a trivial bug:
both the truth image and the retrieval's frozen assumption used the
real, spatially-varying `albedo_for_label` field -- neither held albedo
constant. The mismatch is purely resolution, not a constant-vs-varying
error.

**Test**: added a new `--surface-positions {shared,anchor}` flag to
`gd_joint_block_whole_slit_sweep.py` (threads into `state_spec_from_
scene`'s existing `surface_positions` parameter, previously not CLI-
accessible) and reran Sec.16.2's exact config with albedo frozen on its
own ANCHOR grid (the same fine grid the forward model already renders
from) instead of sharing the atmosphere rows' coarser `bin_centers`
grid. `ch4_ppb`/`co_ppb`/`h2o_surface_vmr` stayed on `bin_centers`,
unchanged.

| config | co2 rms | co2 max | p_surface rms | p_surface max |
|---|---|---|---|---|
| Mode-2 (representable), Sec.14.1 | 0.191 ppm | 2.263 ppm | 0.0171 hPa | 0.159 hPa |
| **Mode-1 dense, albedo on ANCHOR grid** | **4.92 ppm** | **110 ppm** | **2.34 hPa** | **26.0 hPa** |
| Mode-1 dense, albedo on BIN grid (Sec.16.2) | 47.3 ppm | 388 ppm | 26.1 hPa | 104 hPa |

Giving albedo its true value at every anchor cut rms error by
**~9-11x** (co2: 47.3->4.92, p_surface: 26.1->2.34) and max error by
**~3.5-4x**. `resid_hires_rms` also improved substantially (0.024-0.27
-> 0.0027-0.035 across the 58 windows) -- a real, large improvement in
fit quality, not just a coincidental shift in where GN's local minimum
landed.

**Conclusion: albedo's own sub-bin texture was A major driver of
Sec.16.2's blowup, confirmed -- but not the ONLY one.** A substantial
gap remains even with albedo exact at every anchor: still ~25x worse
than the fully-representable Mode-2 ceiling in rms terms, ~40-160x
worse at the max. Candidate remaining causes, not yet tested: (a)
albedo may have real texture even FINER than this retrieval's own
`anchor_density=4` grid (truth is sampled every 500m; ad4 anchors are
coarser), i.e. a smaller residual representability gap for albedo
itself; (b) `ch4_ppb`/`co_ppb`/`h2o_surface_vmr` are still frozen on the
coarser `bin_centers` grid -- even though their true fields are smooth
(broad sinusoids/hotspots, not sharp), their own representability gap
hasn't been ruled out; (c) some other, unidentified mechanism.

Plot: `figures/joint_block/co2p-58-g1-ad4-pf-densefrozenexact-
anchoralbedo-DENSETRUTH_analytic.png`.

**A real, separate gap found along the way**: `gd_joint_block_whole_
slit_sweep.py`'s own `_parts` output-directory naming does not include
`--surface-positions` in its suffix, so this run's raw per-window
output silently landed in and overwrote the SAME directory Sec.16.2's
plain run used. Harmless here (Sec.16.2 was already merged into its own
named `.pkl` and archived before this run started), but a real trap for
future work -- any two configs differing only in a flag the naming
scheme doesn't account for will silently clobber each other's raw
output. Not fixed yet; flagged for whoever next adds a flag that changes
retrieval behavior without changing the output-directory suffix.

**Fast-test-subset finding, for future iteration speed** (user: "is
there a subset of the whole slit that would be good for testing... that
will speed up the process"): pulled directly from Sec.16.2's own
per-window data rather than guessed. `--task-id 21 --n-tasks 58` (rows
189-197), `--task-id 26 --n-tasks 58` (rows 238-248), and `--task-id 57
--n-tasks 58` (rows 1013-1023) are all small (width 9-11 rows, vs. up to
45 elsewhere), fast (7-13 min each vs. up to ~3.2 hours for the widest
windows), and show strong signal (70-215+ ppm CO2 error) for this class
of frozen-row/representability experiment -- notably, the effect is NOT
confined to wide/slow windows at all; even the narrowest windows show
30-100x the Mode-2 error ceiling. Useful for fast iteration on this
line of work instead of always running all 58 windows. (`build_window_
tiles`'s own tile order is deterministic given the same `--min-window`/
`--window-scale`/`--overlap`, so these task_ids stay correct for any
future run at this same tiling config.)


## 2. Freeing albedo instead of pinning it: helps over the original blowup, but usually loses to anchor-resolution pinning (2026-09-05)

Direct follow-up to Sec.1, on the same 3-window fast-test subset
(rows 189-197, 238-248, 1013-1023 -- task_ids 21/26/57). Instead of
freezing albedo (at bin-center or anchor resolution), let it be FREE
(structural prior, default shared `bin_centers` grid -- the same
convention every other free-albedo experiment this project has used),
keeping `ch4_ppb`/`co_ppb`/`h2o_surface_vmr` frozen exact on
`bin_centers` and `co2_ppm`/`p_surface_hpa` free, unchanged, against the
same Mode-1 dense truth.

| window | bin-frozen (orig.) co2 max | anchor-frozen co2/p_surface max | **FREE albedo** co2/p_surface max |
|---|---|---|---|
| 189-197 | 129 ppm | **1.86 / 3.08** | 52.0 / 12.5 |
| 238-248 | 216 ppm | **2.30 / 1.32** | 34.6 / 5.2 |
| 1013-1023 | 168 ppm | 110.5 / 22.5 | **21.4 / 11.0** |

**Freeing albedo beats the original bin-center-frozen case in all 3
windows** (2-8x lower max error) -- GN partially compensating via
fitting instead of being locked to a coarse, wrong-between-nodes exact
value.

**But it underperforms anchor-resolution frozen pinning in 2 of the 3
windows**, sometimes by a lot (52 vs. 1.86 ppm at rows 189-197). The
exception is telling: rows 1013-1023 was exactly the window where
anchor-frozen itself struggled most (110 ppm co2, the single worst
anchor-frozen window in Sec.1's whole-slit run, and a major contributor
to that section's residual ~25x gap) -- and there, FREE albedo actually
beats anchor-frozen.

**Interpretation**: pinning albedo exactly, at fine enough resolution,
beats letting GN fit it -- correct information beats an imperfect fit,
whenever the resolution is actually fine enough to BE correct. Freeing
albedo is not a fix for the underlying resolution problem; it is a
different, generally weaker compensation mechanism that only wins in
the specific case where even anchor-resolution pinning has already run
out of room (real albedo structure below even that grid). This sharpens
Sec.1's own open question: the residual ~25x gap after anchor-frozen
pinning is most likely explained by albedo texture finer than
`anchor_density=4` itself (candidate (a) from Sec.1) rather than the
other frozen rows (candidate (b)) -- freeing albedo helping specifically
at the one window where anchor-frozen already struggled most points at
albedo, not at ch4/co/h2o, as the row still driving the residual gap.

Not yet tested: anchor-frozen albedo at a FINER anchor_density (e.g.
ad16 instead of ad4) on this same fast subset, which would directly
confirm or refute that candidate.


## 3. Ruled out: albedo's own sub-anchor texture is NOT what drives the residual gap at rows 1013-1023 (2026-09-05)

Direct test of Sec.2's sharpened hypothesis: rerun the anchor-frozen-
albedo variant (Sec.1) at `anchor_density=16` instead of 4 -- a 4x
finer anchor grid for both atmosphere rendering and albedo's own
frozen-anchor positions -- on the same 3-window fast subset.

| window | ad4 co2/p_surface max | **ad16** co2/p_surface max |
|---|---|---|
| 189-197 | 1.86 / 3.08 | 2.04 / 3.17 (unchanged, within noise) |
| 238-248 | 2.30 / 1.32 | 2.70 / 1.84 (unchanged, within noise) |
| **1013-1023** | **110.5 / 22.5** | **117.5 / 22.5 (unchanged -- slightly WORSE)** |

**Going 4x finer did nothing.** The residual gap at rows 1013-1023 --
the window driving most of Sec.1's whole-slit ~25x residual -- is
completely unmoved by anchor resolution, ruling out "albedo has real
texture below even the ad4 anchor grid" as the explanation. Sec.2's
own reasoning (freeing albedo helped specifically at this window,
suggesting a resolution problem) pointed at the wrong knob: freeing
albedo lets GN compensate via an entirely different mechanism (fitting
against the DATA) than making the frozen value more spatially precise
does, so that result doesn't actually imply an anchor-resolution
explanation after all -- a real lesson in not over-interpreting one
ablation's direction as confirming a specific mechanism.

**This reopens the question**: with albedo's own resolution ruled out
at two different scales (bin_centers -> anchor -> even finer anchor,
no improvement past the first jump), the remaining candidates from
Sec.1 move to the front: (b) `ch4_ppb`/`co_ppb`/`h2o_surface_vmr` --
still frozen on the coarse `bin_centers` grid this whole time, never
tested at anchor resolution -- or (c) something else about this
specific window (rows 1013-1023) entirely, e.g. a local feature (a
gas hotspot, a keystone effect, proximity to the slit edge) rather than
a representability mechanism at all. Worth checking directly: are
ch4/co/h2o representable at `bin_centers` resolution for THIS window
specifically, the same way Sec.12's own methodology scored the
free-row representability gap originally.

**Next concrete test**: rerun with `ch4_ppb`/`co_ppb`/`h2o_surface_vmr`
ALSO frozen on the anchor grid (`--surface-positions` only covers the
surface/albedo row today -- the atmosphere rows have no equivalent
flag yet; would need a small extension, or a temporary monkeypatch of
`state_spec_from_scene`'s `bin_centers` argument itself for a one-off
test) on the same 3-window fast subset, to see whether THAT closes the
gap at rows 1013-1023 instead.


## 4. Ruled out entirely: frozen-row representability is NOT the driver at rows 1013-1023 -- this looks like a slit-edge artifact instead (2026-09-05)

Second direct test, closing the representability-gap line of inquiry.
Extended `state_spec_from_scene` with a new general `row_positions`
parameter (any row, atmosphere included, can now live on a custom
position grid, not just the surface/albedo row) and a matching
`--frozen-atmosphere-positions {shared,anchor}` CLI flag. Froze
`ch4_ppb`/`co_ppb`/`h2o_surface_vmr` on the anchor grid TOO (in addition
to albedo, already anchor-frozen since Sec.1), same 3-window fast
subset, same ad4.

| window | ad4, albedo-only anchor | ad16, albedo-only anchor | **ad4, ALL rows anchor** |
|---|---|---|---|
| 189-197 | 1.86 / 3.08 | 2.04 / 3.17 | 1.86 / 3.10 (unchanged) |
| 238-248 | 2.30 / 1.32 | 2.70 / 1.84 | 2.30 / 1.30 (unchanged) |
| **1013-1023** | **110.5 / 22.5** | 117.5 / 22.5 | **110.5 / 22.5 (unchanged)** |

Freezing `ch4_ppb`/`co_ppb`/`h2o_surface_vmr` at anchor resolution too
made zero difference (110.47 vs. 110.50, identical within noise).
Combined with Sec.3's albedo-resolution result, this rules out frozen-
row representability -- at ANY tested resolution, for ANY tested row --
as the mechanism behind the gap at rows 1013-1023.

**New leading hypothesis: a slit-edge artifact, not a representability
gap at all.** Rows 1013-1023 are the LAST window on the whole 1024-row
detector (`ROW_MAX_IDX`), where `PAD` extension runs out of real rows
to extend into and keystone geometry is at its most extreme. This is a
qualitatively different kind of explanation than everything else in
Sec.1-3 -- not about how finely a frozen row's value is known, but about
this window's own position at the physical boundary of the data.

**Next concrete test**: check whether the SAME pathology appears at the
OTHER slit edge (the first window, rows 0-8) under the plain (bin-
center-frozen) Experiment B config -- if the first and last windows are
both anomalously bad relative to interior windows of similar width,
that's a strong, cheap confirmation of the edge-artifact hypothesis
before investing in tracking down the specific mechanism (PAD
truncation, extreme keystone, or something else at the boundary).


## 5. Resolved: rows 1013-1023's blowup is extreme keystone smearing, not a representability gap -- the same mechanism as the archived Sec.14.4 water-region finding (2026-09-05)

Direct follow-up to Sec.4's "check the other slit edge" test. Pulled
the first window (rows 0-8, task_id=0) directly from the already-
archived full-58-window runs (no new job needed) for both the plain
bin-frozen (`archive/initial_build/.../co2p-58-g1-ad4-pf-
densefrozenexact-DENSETRUTH_analytic.pkl`) and albedo-anchor-frozen
configs, and compared both slit edges:

| window | x_km range | bin-frozen co2/p_surface max | anchor-frozen co2/p_surface max | resid_rms (bin-frozen) |
|---|---|---|---|---|
| (0,8) first edge | [-1400.5,-1378.0] | 28.7 / 63.8 | 12.5 / 20.6 | 0.218 |
| (1013,1023) last edge | [1309.5,1364.4] | 167.7 / 67.3 | 110.5 / 22.5 | 0.078 |
| (189,197) typical interior | -- | -- | 1.9 / 3.1 | 0.020 |

**Partial confirmation, then a clean resolution.** Both edges ARE
elevated relative to a typical interior window -- especially
`p_surface` (63.8/67.3 vs. ~1-3 elsewhere), confirming a real edge
effect exists. But it's asymmetric, not symmetric: CO2's error at the
LAST window (110-168) is 5-9x worse than at the FIRST window (12.5-
28.7). Checked directly whether this was proximity to the known CO2
hotspot at x0=+1050km (width 9km) -- ruled out: rows 1013-1023 sit at
x_km=[1310,1364], 260+ km from the hotspot, far outside a 9km-wide
feature's influence.

**The actual explanation**: `rows_crossed` (the keystone-smearing proxy
-- how many true along-slit rows a single detector row's spectrum
averages over) is ~0.06-0.27 near the LEFT edge (rows 0-19, essentially
FPA2's own keystone-null point, consistent with the project's earlier-
established finding that FPA2's zero-point sits near row 25) but
~10.2-10.4 near the RIGHT edge (rows 1004-1023) -- a ~40x difference in
physical keystone smearing between the two ends of the slit. Rows
1013-1023 sits in a region of genuinely extreme keystone distortion,
maximally far from the null point.

**This is the SAME mechanism as the archived-record's own Sec.14.4
finding** (water-region damage reaching neighboring rows via keystone,
each detector row's spectral samples spanning ~20km along the slit) --
not a new, separate phenomenon. Extreme keystone smearing degrades the
retrieval's ability to separate correlated absorbers and pin down
localized structure regardless of what mechanism (frozen-row resolution,
water/albedo contrast, or here, sheer proximity to the keystone-null-
point's opposite extreme) creates the local difficulty. Frozen-row
representability (Sec.1-4) was a real, genuine, but ultimately SEPARATE
and smaller effect (closing ~90% of a DIFFERENT blowup, Sec.1) from
this keystone mechanism, which explains the specific residual that
survived every representability fix tried.

**Practical implication**: rows 1013-1023 (and by extension, any window
near the slit edge opposite FPA2's keystone-null point) should be
expected to show elevated error under ANY frozen-row or resolution
configuration -- this is closer to an intrinsic instrument-geometry
floor for that specific along-slit location than a fixable retrieval-
configuration problem. Not yet quantified: whether OTHER FPA bands have
their own keystone-null point at a different location (each band's own
clocking offset differs, per the project's Era-1 findings), meaning
this specific "worst edge" would move for FPA0/1/3.


## 6. Finer g_ratio (0.5, 0.25): real improvement everywhere tested, but a keystone-set floor persists at the worst edge (2026-09-05/06)

User direction: "I'd like to try a whole slit sweep with co2, pressure,
and albedo unfrozen and the others set at the anchor truth values. Use
g_ratios of 0.5 and 0.25" -- explicitly superseding an earlier (now-
outdated) instruction to skip `g_ratio=0.5` in sweeps, since that
guidance predated the current footprint-integrated truth mechanism.

**Cost reality check first**: a full 58-window array at `g_ratio=0.25`
turned out to be far more expensive than any prior config this
session -- the single widest window (G=180) took **19061s (~5.3
hours)** on its own. Capped to the fast 3-window subset
(task_ids 21/26/57, rows 189-197/238-248/1013-1023) plus two extra
single-window checks (rows 0-8 and 968-1012) for `g_ratio in
{1, 0.5, 0.25}`, all anchor-frozen `ch4_ppb`/`co_ppb`/
`h2o_surface_vmr`, Mode-1 dense truth, co2/p_surface/albedo free.

**A real naming-collision trap caught along the way**: an earlier,
differently-configured run (ch4/co/h2o frozen on `bin_centers`, not
anchor) had left a `task057of58.pkl` in the SAME `g_ratio=1` output
directory this sweep also writes to. Checked directly (`ch4_ppb`'s own
position count) before trusting the "g_ratio=1" comparison point --
found it stale (11 positions, matching `bin_centers`, not the ~40+-
point anchor grid this sweep uses) -- and reran that one cell properly
rather than report a mismatched comparison.

### Full comparison (|error| mean / median / max / rms; co2 in ppm, p_surface in hPa)

| window | g_ratio | G | t_hires | co2 |err| (mean/med/max/rms) | p_surface |err| (mean/med/max/rms) |
|---|---|---|---|---|---|
| (0,8) near-null | 1 | 9 | 298s | 4.78/4.89/10.83/5.92 | 2.47/2.75/5.73/2.90 |
| | 0.25 | 36 | 1296s | 3.51/**2.28**/12.54/4.86 | **0.20**/**0.15**/**0.54**/0.24 |
| (189,197) | 1 | 9 | 498s | 13.45/8.41/52.03/19.71 | 9.28/9.81/12.50/9.75 |
| | 0.5 | 18 | 904s | 6.43/6.27/15.71/7.89 | 1.58/1.13/5.85/2.26 |
| | 0.25 | 36 | 2752s | 4.46/**3.62**/**12.57**/5.59 | **0.48**/**0.43**/**1.35**/0.59 |
| (238,248) | 1 | 11 | 549s | 17.06/16.41/34.62/18.45 | 1.75/1.07/5.22/2.38 |
| | 0.5 | 22 | 1211s | 5.16/2.77/18.68/7.32 | 0.74/0.26/7.64/1.73 |
| | 0.25 | 44 | 2295s | 3.57/**2.70**/**9.93**/4.49 | **0.28**/0.28/**0.84**/0.36 |
| (1013,1023) far edge | 1 | 11 | 344s | 5.57/4.42/21.42/7.59 | 3.78/2.72/10.98/4.76 |
| | 0.5 | 22 | 643s | 4.58/1.38/**27.70**/8.26 | 1.52/0.77/6.41/2.29 |
| | 0.25 | 44 | 1745s | 2.58/**1.94**/10.56/3.60 | **0.49**/**0.28**/**3.97**/0.81 |
| (968,1012) widest | 1 | 45 | 5678s | 5.60/5.23/31.16/7.38 | 3.55/2.47/17.28/5.00 |
| | 0.25 | 180 | 19061s | 3.26/**1.92**/**67.59**/6.65 | **0.46**/**0.23**/10.37/1.08 |

### What the data shows

**Finer g_ratio helps, consistently, across every window tested** --
median and rms error drop at every step from `g_ratio=1` to `0.5` to
`0.25`, at every window. `p_surface` improves the most dramatically
(e.g. rows 189-197: median 9.81 -> 1.13 -> 0.43 hPa, a ~23x reduction
end to end). This is a real, substantial, and consistent finding, not
noise -- confirms the user's premise that finer resolution genuinely
buys real-world improvement now that the truth-rendering mechanism is
correct (unlike whatever the earlier "skip g_ratio=0.5" guidance was
based on).

**But max error doesn't always follow the same trend, and the far-edge
window's max is a real floor, not a resolution problem.** Rows 1013-
1023's co2 max actually gets WORSE from g_ratio=1 to 0.5 (21.4 -> 27.7)
before improving at 0.25 (10.6) -- and even at the widest window
(rows 968-1012), a single outlier bin still reaches max=67.6 ppm even
at `g_ratio=0.25`'s finest tested resolution, while that same window's
MEDIAN error (1.92 ppm) is unremarkable. This is consistent with
Sec.5's keystone-smearing finding: a handful of bins near extreme-
keystone locations carry an error floor that finer binning alone
doesn't fully remove, riding on top of the broad, genuine improvement
finer resolution gives everywhere else.

**Cost scales roughly as G^1.20** (power-law fit across all 12 (G,
t_hires) pairs collected here, refined again now that the (968,1012)
widest-window `g_ratio=1` cell -- G=45, t_hires=5678s -- has finally
finished; barely moved the exponent from the earlier 11-point ~G^1.24
estimate, itself a refinement of Sec.13's memory-note ~G^1.6 guess from
just two data points). The widest single case (G=180) still ran
somewhat above the fitted trend, suggesting the curve may steepen
further at very large G rather than staying a clean power law
indefinitely.

**Net recommendation**: `g_ratio=0.25` gives the best accuracy of the
three at every window tested, but at real cost -- roughly 6-9x
`g_ratio=1`'s wall-clock time per window, before even accounting for
G's own superlinear-in-width growth (which is why the widest window's
full-array cost is prohibitive, Sec.6's own opening cost-reality-check).
`g_ratio=0.5` is a genuinely reasonable middle ground: most of the
`p_surface` improvement, a real (if smaller) co2 improvement, at
roughly 2-3.5x `g_ratio=1`'s cost rather than 6-9x. Whether a full
58-window `g_ratio=0.25` sweep is worth ~5.3 hours for its single
worst window (and proportionally more for the rest) is a genuine
cost/benefit call, not yet made -- not attempted in this session beyond
the two extra single-window checks.


## 7. Keystone vs. truth structure, isolated: reversing the truth confirms Sec.5's geometry explanation decisively (2026-09-06)

User direction: "In order to isolate the effects of keystone versus the
structure in the truth, we can run parallel experiments with the truth
profile along the slit reversed." A clean control: mirror EVERY truth
field spatially (`fn(x_km) -> fn(-x_km)`) while leaving the instrument
geometry (row<->eta mapping, the `rows_crossed` keystone curve) exactly
as-is. Since `rows_crossed` is a pure geometry property (computed from
`gd_polynomials` alone, no truth field involved), reversing the truth
changes WHICH physical along-slit content sits at a given row without
changing how much keystone smearing happens there. If a row range's
error is really a geometry effect (Sec.5), it should stay comparably
bad under the reversed truth; if it were really about unlucky truth
structure happening to sit there, the error should move to wherever
that structure now sits after mirroring.

Built a reversed Mode-1 dense truth image
(`scratch_work/build_whole_slit_truth_reversed.py`, verified via a
direct sanity check that `reversed_fn(500) == original_fn(-500)`
exactly) and reran Sec.6's exact `g_ratio=1` config (co2/p_surface/
albedo free, ch4/co/h2o frozen exact on the anchor grid -- matching
fields reversed too, so frozen rows stay consistent with the reversed
truth) on the same 3-window fast subset.

| window | keystone strength | original truth co2 \|err\| (mean/med/max/rms) | **reversed** truth co2 \|err\| (mean/med/max/rms) |
|---|---|---|---|
| 189-197 (near-null-ish) | ~1.8-2.3 | 13.45/8.41/52.03/19.71 | **4.14/2.56/15.41/6.11** (much BETTER) |
| 238-248 (moderate) | ~2.3 | 17.06/16.41/34.62/18.45 | 17.91/19.04/38.14/19.89 (about the SAME) |
| **1013-1023 (far edge, extreme)** | **~10.3-10.4** | 5.57/4.42/21.42/7.59 | **19.00/9.83/97.59/32.12 (much WORSE)** |

(p_surface shows the same qualitative pattern: 189-197 improves
9.28->1.95 mean; 1013-1023 worsens 3.78->6.44 mean, `resid_hires_rms`
jumping 0.017->0.168 there too -- the fit itself got harder, not just
the scoring against a different truth.)

**Decisive, and stronger than expected**: rows 1013-1023 doesn't just
stay comparably bad under the reversed truth -- it gets WORSE (co2 max
21.4 -> 97.6 ppm, rms 7.6 -> 32.1). This rules out "the original truth
happened to have easy structure there" as any part of the explanation;
if anything the reversed content is harder to fit at that same extreme-
keystone location, consistent with the mechanism being about the
LOCATION's own geometry, not the specific values that happen to occupy
it either way.

**The near-null window's behavior is the other half of the confirmation**:
rows 189-197 (low keystone strength, where geometry is NOT the
bottleneck) shows a LARGE, real change under reversal (rms 19.71 ->
6.11) -- exactly what you'd expect if error there is genuinely driven
by which truth content sits at that location, since geometry isn't
constraining it much either way. Rows 238-248 (intermediate keystone)
sits in between, changing only modestly -- consistent with a real but
smaller location-specific contribution layered on top of whatever
content-sensitivity remains.

**Conclusion**: this cleanly separates the two effects the user asked
to disentangle. At extreme-keystone locations (rows 1013-1023), error
is a geometry-driven floor, insensitive to (or even worsened
regardless of) what truth content is actually there -- confirming
Sec.5's explanation more strongly than the original single-truth
finding alone could. At low-keystone locations, error is genuinely
truth-content-sensitive, and finer resolution (Sec.6) or better priors
would be expected to help there in a way they cannot fully help at the
keystone floor. This also reframes Sec.6's own g_ratio table: finer
binning was never going to close row 1013-1023's gap on its own (Sec.5
already implied this; this section now shows the floor exists
regardless of what's being fit there, not just under one particular
truth realization).

Not yet run at the time this section was first written: the same
reversal test at `g_ratio=0.5`/`0.25` (does finer binning shrink the
keystone floor's magnitude even if it can't remove it entirely?), and
at the widest window (rows 968-1012) for a second, independent
high-keystone data point beyond 1013-1023 alone -- see Sec.8 for the
former.

## 8. Reversed-truth control at finer g_ratio: the keystone floor shrinks with resolution, but doesn't flatten out like the low-keystone window does (2026-09-06)

Follow-up to Sec.7, requested directly ("Yes please" in response to
"want me to queue that up next?"): reran the exact Sec.7 reversed-truth
config at `g_ratio in {0.5, 0.25}` on the same fast 3-window subset,
using a new GRATIO-env-var-parameterized driver
(`scratch_work/retrieval_co2palbedo_reversed_truth_anchorfrozen_gratio.py`,
generalizing the g_ratio=1-only script Sec.7 used).

### co2 |error| (mean/median/max/rms, ppm) -- reversed truth, by g_ratio

| window | keystone strength | g_ratio=1 | g_ratio=0.5 | g_ratio=0.25 |
|---|---|---|---|---|
| 189-197 (near-null-ish) | ~1.8-2.3 | 4.14/2.56/15.41/6.11 | 7.63/6.32/16.65/9.34 | 5.86/4.78/16.72/7.16 |
| 238-248 (moderate) | ~2.3 | 17.91/19.04/38.14/19.89 | 4.70/2.37/**33.26**/8.64 | 4.67/3.76/21.26/6.39 |
| **1013-1023 (far edge, extreme)** | **~10.3-10.4** | 19.00/9.83/**97.59**/32.12 | 3.36/2.91/15.36/4.73 | **1.95**/**1.71**/**6.85**/**2.41** |

### p_surface |error| (mean/median/max/rms, hPa)

| window | g_ratio=1 | g_ratio=0.5 | g_ratio=0.25 |
|---|---|---|---|
| 189-197 | 1.95/1.92/3.37/2.27 | 1.17/0.65/3.65/1.65 | 0.63/0.60/1.99/0.76 |
| 238-248 | 1.71/1.95/3.25/2.07 | 0.82/0.40/4.20/1.40 | 0.39/0.26/1.60/0.53 |
| **1013-1023** | 6.44/5.13/11.64/7.26 | 3.19/1.43/14.19/4.89 | **0.75**/**0.52**/**3.75**/**0.99** |

### What the data shows

**At the extreme-keystone window (1013-1023), finer g_ratio shrinks the
floor substantially -- co2 rms drops 32.1 -> 4.7 -> 2.4 from g_ratio=1
to 0.5 to 0.25, a ~13x reduction end to end, and the max error (driven
by the worst single bin) drops even more, 97.6 -> 15.4 -> 6.9.** This
is a real, monotonic improvement, unlike the non-reversed truth's
version of this same window in Sec.6, where co2 max actually got WORSE
from g_ratio=1 to 0.5 (21.4 -> 27.7) before improving at 0.25. So under
this harder (reversed) truth, finer binning behaves more predictably --
monotonic improvement at every step -- than it did under the original
truth's own particular structure at that location.

**Surprisingly, the floor doesn't just shrink -- at fine enough
resolution it stops being a floor at all relative to the low-keystone
window.** At `g_ratio=0.25`, 1013-1023's co2 rms (2.41) is actually
LOWER than 189-197's rms at `g_ratio=1` (6.11), i.e. the extreme-
keystone window's error is now competitive with, or better than, the
low-keystone window's own numbers. That reframes the "floor" language
from Sec.5/7 somewhat: the keystone effect sets how much a GIVEN
resolution can achieve at that location, but it does not set an
absolute, resolution-independent worst case -- sufficiently fine
`g_ratio` closes most of the reversed-truth gap at the worst edge,
consistent with Sec.6's original (non-reversed) finding that finer
resolution helps broadly, just confirmed here to also apply under the
harder reversed-truth stress test.

**The other two windows show the more familiar non-monotonic/noisy
behavior already seen in Sec.6's non-reversed table**: 238-248's co2
max actually jumps at `g_ratio=0.5` (38.1 -> 33.3, roughly flat) before
dropping at 0.25 (21.3), and 189-197 gets modestly WORSE at 0.5 before
partially recovering at 0.25 (never returning fully to its `g_ratio=1`
level) -- a reminder that a single fine-binning step can occasionally
make one bin's fit locally worse even as the aggregate (median/rms)
trend across all windows keeps improving with resolution, matching the
same non-monotonic caveat already flagged in Sec.6.

**Conclusion**: the keystone-driven floor identified in Sec.5/7 is real
but not fixed in magnitude -- it is a floor FOR A GIVEN g_ratio, and
finer resolution buys real, substantial relief at exactly the location
predicted (the extreme-keystone far edge), more so than it does
elsewhere. This strengthens the case for `g_ratio=0.5`/`0.25` (per the
updated production-default guidance in Sec.6/roadmap) specifically as a
mitigation for the worst-affected rows, even though it cannot eliminate
the underlying geometry effect Sec.5/7 established.

Not yet run: the same reversal test at the widest window (rows
968-1012) for a second, independent high-keystone data point at finer
g_ratio; determining each other FPA band's own keystone-null location.
