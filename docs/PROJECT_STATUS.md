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
`gd_joint_block_retrieve.py` (threads into `state_spec_from_
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

**Second, independent high-keystone data point (rows 968-1012, the
widest window)**, run 2026-09-06/07: same reversed-truth config,
`g_ratio in {1, 0.5, 0.25}`.

| g_ratio | G | t_hires | co2 |err| (mean/med/max/rms) | p_surface |err| (mean/med/max/rms) |
|---|---|---|---|---|
| 1 | 45 | 12247s | 14.58/6.52/105.94/25.96 | 11.07/8.99/39.36/13.65 |
| 0.5 | 90 | 12939s | 3.79/2.34/26.44/5.63 | 2.67/1.48/17.82/3.92 |
| 0.25 | 180 | 25941s | 3.09/1.94/57.51/6.19 | **0.60**/**0.40**/6.98/**1.08** |

Confirms the same qualitative pattern as rows 1013-1023: co2/p_surface
rms both drop sharply from g_ratio=1 to 0.5 (co2 rms 26.0->5.6, p_surface
13.6->3.9), and p_surface keeps improving cleanly to 0.25 (down to 1.08
hPa). co2's max/rms at 0.25 are NOT a clean further improvement over 0.5
here (rms ticks up slightly 5.63->6.19, max jumps 26.4->57.5) -- a
reminder, at an even larger G (180) than any single-window case tested
in Sec.6/8's fast subset, that a single fine-binning step can still make
one or two bins locally worse even as the broad trend keeps improving
(the same non-monotonic caveat flagged for the other two windows in this
section and in Sec.6). Net: the keystone-floor-shrinks-with-resolution
finding holds up at a second, independent extreme-keystone location, not
just the one window Sec.7/8 first tested it on.

Not yet run: determining each other FPA band's own keystone-null
location.

## 9. Defocus (wide-PSF) experiments: the retrieval largely compensates for an assumed-vs-true PSF mismatch except at the extreme-keystone edge (2026-09-06)

User direction: "I'd like to plan some experiments with wider PSFs to
simulate a defocusing effect with a focal adjustment mechanism that
exists on the telescope. It doesn't affect the spectral resolution, just
the spatial blurring." The instrument's along-slit (N/S) PSF FWHM was
hardcoded at 1.5px (the real ground-test-measured value) at every call
site in the whole-slit pipeline, with truth and retrieval PSF always
coupled purely because nothing ever passed a different value. Added an
explicit `spatial_psf_fwhm_px` parameter threaded end-to-end
(`geocarb_gert.joint_state.render_at_anchors`/`build_forward_state`,
`gd_build_resolution_matched_truth.py`'s truth-render functions, a new
`--retrieval-psf-fwhm-px` CLI flag on the sweep script), decoupling what
PSF the TRUTH is rendered with from what the RETRIEVAL's own forward
model assumes -- `default_pad_for_psf` scales the render padding
proportionally so a wider PSF doesn't truncate its own Gaussian kernel
at window edges. See the approved plan
(`resolution-matched truth images`/defocus plan, superseded content) for
the full design.

**A real naming-collision bug caught along the way**: the sweep script's
own output-directory suffix keys on `--prior-fields`' STRING NAME, not
its content, and had no dependence on `--retrieval-psf-fwhm-px` at all.
The first full 6-config matrix run (3 truth PSF FWHMs x matched/
mismatched retrieval PSF) registered every truth-PSF config under the
SAME literal prior-fields name, so all 6 configs silently wrote into
ONE shared `_parts` directory -- caught by checking the directory
listing (3 files, not 18) before trusting any comparison. Fixed both the
sweep script's suffix (`_retrpsf{value}` whenever non-default) and the
driver's registry naming (embeds the truth PSF tag), verified two
distinct directories result, and reran the clean matrix.

Ran the fast 3-window subset (task_ids 21/26/57, rows 189-197/238-248/
1013-1023) at truth PSF FWHM in {3, 5, 8} px (vs. nominal 1.5px),
each against BOTH a matched retrieval (told the true PSF) and a
mismatched one (still assumes nominal 1.5px -- an uncorrected/
uncalibrated defocus event), `g_ratio=1` fixed, same anchor-frozen
ch4/co/h2o config as Sec.6/7/8.

### co2 |error| (mean/median/max/rms, ppm)

| window | FWHM | G | matched | mismatched |
|---|---|---|---|---|
| 189-197 | 3px | 9 | 1.53/1.31/2.80/1.66 | 1.49/1.24/2.88/1.64 |
| | 5px | 9 | 2.90/3.14/5.34/3.11 | 2.79/2.94/5.35/3.02 |
| | 8px | 9 | 7.42/7.21/11.58/7.61 | 7.24/7.13/11.57/7.48 |
| 238-248 | 3px | 11 | 2.78/1.10/7.98/3.85 | 3.30/1.26/9.43/4.50 |
| | 5px | 11 | 1.55/1.55/3.69/1.81 | 1.62/1.49/3.96/1.92 |
| | **8px** | 11 | 4.75/4.31/7.50/4.92 | **4.75/4.31/7.50/4.92 (identical)** |
| **1013-1023 (far edge)** | 3px | 11 | 5.42/5.33/14.26/6.42 | 7.06/5.87/16.79/8.41 |
| | 5px | 11 | 6.95/6.02/20.50/8.97 | 7.67/6.14/20.89/10.21 |
| | 8px | 11 | 11.04/9.16/28.51/13.78 | 12.19/10.26/36.75/15.91 |

### p_surface |error| (mean/median/max/rms, hPa)

| window | FWHM | matched | mismatched |
|---|---|---|---|
| 189-197 | 3px | 0.55/0.63/1.28/0.71 | 0.52/0.41/1.22/0.65 |
| | 5px | 0.44/0.30/1.04/0.56 | 0.42/0.43/0.77/0.49 |
| | 8px | 0.84/0.77/1.77/0.96 | 0.71/0.60/1.78/0.85 |
| 238-248 | 3px | 0.47/0.14/1.40/0.69 | 0.53/0.22/1.81/0.79 |
| | 5px | 0.20/0.13/0.74/0.28 | 0.22/0.15/0.81/0.31 |
| | 8px | 0.58/0.39/1.46/0.69 | 0.58/0.39/1.46/0.69 (identical) |
| **1013-1023** | 3px | 1.73/0.74/9.03/3.02 | 2.07/1.18/9.18/3.16 |
| | 5px | 2.48/2.04/8.63/3.26 | 2.37/2.03/8.62/3.22 |
| | 8px | 4.08/3.50/11.36/5.09 | 4.19/3.08/11.09/5.09 |

### What the data shows

**Defocus alone (even MATCHED, i.e. the retrieval knows about it) makes
error worse, monotonically with FWHM, at every window** -- e.g. 189-197's
co2 rms goes 1.66 -> 3.11 -> 7.61 ppm from 3px to 5px to 8px even when
the retrieval is told the correct PSF. This is expected: a wider PSF
mixes more along-slit content into every detector row regardless of
whether the retrieval models it, so information content genuinely drops
with defocus even in the best (fully-corrected) case.

**At low/moderate-keystone windows (189-197, 238-248), matched vs.
mismatched makes little to no difference** -- at 238-248/8px they are
IDENTICAL to 5 significant figures (verified directly: distinct output
files, `retrieval_psf_fwhm_px` correctly recorded as 8.0 vs 1.5 in each,
not a stale-file collision). The retrieval's own free parameters
(especially free albedo, which has the most local degrees of freedom)
apparently absorb a completely wrong PSF assumption almost perfectly at
these locations -- an uncorrected/uncalibrated defocus event would be
nearly invisible in the RETRIEVED STATE at these rows, even though the
underlying forward-model mismatch is real.

**At the extreme-keystone far edge (1013-1023), the mismatch matters
substantially and grows with FWHM** -- co2 rms mismatched-vs-matched
gap: 6.42 vs 8.41 (3px, +31%), 8.97 vs 10.21 (5px, +14%), 13.78 vs 15.91
(8px, +15%); the max error gap is starker still (14.3 vs 16.8 at 3px;
28.5 vs 36.8 at 8px). This is the same location Sec.5/7/8 already
identified as keystone-floor-limited -- with less local flexibility to
compensate (the window's own information content is already
compromised by extreme keystone smearing), an uncorrected defocus event
compounds the existing floor rather than being absorbed by the free
parameters the way it is elsewhere.

**Conclusion**: this reframes the practical stakes of the telescope's
focal-adjustment mechanism. If a defocus event went uncorrected
(retrieval never told), the damage would be roughly self-limiting at
most of the slit (the free-parameter fit compensates), but would
compound on top of the existing keystone floor specifically at the
worst-keystone rows -- exactly where Sec.5/7/8 already showed retrieval
performance is most fragile. A calibration/monitoring priority
implication: knowing the true PSF matters most exactly where it's
already hardest to retrieve well, not uniformly across the slit.

Not yet run: the same matrix at the widest window (968-1012, second
independent high-keystone data point); finer g_ratio combined with
defocus (does resolution help the mismatched case the way it helped the
reversed-truth keystone floor in Sec.8?); a middle-ground defocus level
between nominal and 3px, to locate where the matched/mismatched
divergence actually begins.

## 9a. Confirming the edge-clamp confound: a much wider window removes the "even matched gets worse" pattern (2026-09-06/07)

Direct follow-up, requested after Sec.9's own per-bin analysis identified
a confound: `state_spec_from_scene`'s free parameters (co2_ppm/
p_surface_hpa/albedo) live only on a window's own local `bin_centers`,
but `predict_neighborhood`'s PSF blur needs a padded render region
scaling with FWHM (`default_pad_for_psf`: 4/8/14/22 rows at FWHM
1.5/3/5/8px) -- `_row_interp1d`'s `fill_value=(lo, hi)` CLAMPS the free
state to the nearest window-edge value in that padding rather than
extrapolating any real gradient. For Sec.9's narrow fast-subset windows
(9-11 rows), pad/width ratio reaches ~200% at FWHM=8px; a much wider
window should shrink this ratio and, if the hypothesis is right, largely
remove the "even MATCHED defocus gets worse with wider FWHM" pattern.

**Test**: reran ONE much wider window (`--min-window 20` -> 41-row
window, rows 205-245, vs. the original 11-row window at a nearby slit
location -- pad/width ratio drops from ~200% to ~54% at FWHM=8px) at
both nominal (1.5px) and defocused (8px) matched PSF, same anchor-frozen
config otherwise.

(co2_ppm turned out uninformative for this specific window -- its
structural prior happens to agree with the true field to ~6 significant
figures at this slit location, so GN correctly left it at scale factor
1.0 regardless of PSF, giving near-zero error at BOTH configs. p_surface
is the real test here, since it did move.)

### p_surface |error| per-bin profile, rows 205-245 (G=41)

- **nominal (1.5px)**: highly variable, 0.02-9.04 hPa across bins --
  real per-location fit structure, no systematic pattern.
- **defocus (8px, matched)**: nearly FLAT, 2.75-2.96 hPa across every
  single bin -- no edge elevation anywhere, a smooth near-constant
  residual from one end of the window to the other.

### Aggregate p_surface stats

| config | mean | median | max | rms |
|---|---|---|---|---|
| nominal (1.5px) | 3.30 | 2.62 | 9.04 | 4.28 |
| defocus (8px, matched) | 2.87 | 2.88 | 2.96 | **2.87 (BETTER than nominal)** |

### What this shows

**Decisive confirmation of the edge-clamp confound.** At the wide
window, the defocused-matched result is not just less bad than the
narrow-window pattern predicted -- it's actually BETTER than nominal in
aggregate, with a flat, edge-effect-free error profile. This is the
opposite of Sec.9's narrow-window finding (error growing monotonically
and worst at the edges as FWHM increased). The narrow-window "even
matched gets worse with wider PSF" result was therefore dominated by the
pad/width-ratio artifact (free parameters clamped, not extrapolated,
into an increasingly large padded region relative to window width), not
a fundamental information-loss property of defocus itself.

**Reframes Sec.9's headline finding**: defocus's TRUE cost to a
well-modeled (matched) retrieval is much smaller than the narrow-window
numbers suggested -- most of that apparent cost was a tiling artifact of
testing on very narrow windows relative to the PSF's own reach, not
something a real production algorithm (which would use wider windows,
`MIN_WINDOW`>=Sec.3's own production tiling) would actually suffer from
comparably. The MISMATCHED (uncorrected) result's own real degradation
at the extreme-keystone edge (Sec.9's other finding) is unaffected by
this correction -- that comparison held two configs at the SAME window
width and PSF-vs-pad ratio, so the artifact cancels out between them.

**Practical implication for future defocus/PSF experiments**: always use
production-representative window widths (large `min_window`), or the
narrow fast-subset windows this session's other experiments relied on
for speed will systematically overstate any effect that depends on
`pad` -- a new caveat to flag whenever `pad` (or anything that scales
with it, like `spatial_psf_fwhm_px`) is a sweep axis.

Not yet run: the same wide-window matched-vs-mismatched comparison
(to see whether the extreme-keystone mismatch-penalty finding also
shrinks at wide windows, or is genuinely location-driven and holds
regardless of tiling); a location where co2's structural prior more
meaningfully diverges from truth, to get a real co2 data point for this
same wide-window test.

## 10. Temperature added as a retrievable state row (T_offset_K) (2026-09-07)

Direct follow-up to `docs/ALGORITHM_ROADMAP.md` Sec.2 item 3 (open since
2026-09-04): temperature was not retrievable at all before this -- fixed
via the standard atmosphere. Added `t_offset_k`, a uniform additive shift
to the standard-atmosphere temperature profile, following the pattern
every other row already uses:

- **Truth field**: a synoptic sinusoid, ±3K amplitude, same 2800km period
  as `p_surface_hpa`'s own synoptic term but a distinct phase (both
  represent "today's weather," deliberately not perfectly correlated
  along the slit). No localized plume/hot-spot term -- temperature has no
  point-source physical analogue.
- **Structural prior**: flat 0K ("standard atmosphere, no known anomaly"),
  mirroring co2/ch4/co's own background-only prior convention -- unlike
  `p_surface_hpa` there's no static/topographic component worth keeping.
- **Analytic Jacobian** (`geocarb_gert.jacobians.t_offset_dI_dparam`):
  exactly the temperature-path term `p_surface_dI_dparam` already uses
  (`dtau_mol_dT_lay_hires` x `K_mol_lay_hires`), but with `dT_lay/
  d(t_offset_k) = 1` at every layer instead of p_surface's finite-
  differenced `dT_lay/d(p_surface)` -- genuinely zero finite differences,
  the simplest Jacobian in the module after albedo.
- **`kind="absolute"` wired end-to-end for the first time** -- the second
  item the roadmap flagged as "not yet checked." Found and fixed a real
  gap while doing so: `geocarb_gert.jacobians.linearize` hardcoded the
  `kind="scale"` chain-rule factor (`d(value)/dx = prior[k]`) and
  explicitly RAISED on any other `kind`, meaning `kind="absolute"` was
  advertised as available (module docstring) but not actually reachable
  through the analytic path at all. Fixed: `linearize` now branches
  `dval_dx = prior[k]` (`kind="scale"`) or `1.0` (`kind="absolute"`),
  matching `ParamSpec.apply`'s own identical branch exactly.

**A real, material side effect found and accepted (user's explicit
call)**: `PRIOR_FIELD_SETS["exact"]`/`["structural"]` point AT
`STATE_FIELDS`/`STATE_FIELDS_PRIOR` (not copies), and most of this
session's own driver scripts build their prior-fields registry via a dict
comprehension OVER those same dicts' `.items()`. Adding `t_offset_k` there
means it now flows automatically -- FROZEN at its real ±3K truth value --
into every one of Sec.1-9's own driver scripts if rerun from now on, even
ones that never asked for temperature. This is the intended consequence
of the file's own "no quantity is privileged" design (the same thing
presumably happened when p_surface/H2O were first added) -- accepted
rather than special-cased away. Practical implication: Sec.1-9's own
numbers stay valid as a record of what was true when written; a FRESH
rerun of any of those scripts from now on picks up a new, real temperature
confound that wasn't part of the original experiment.

### Verification (all four passed)

1. **`x0()`/prior round-trip for `kind="absolute"`**: confirmed directly
   -- with the structural (flat 0K) prior as the reference, `x0()` returns
   the prior itself (`[0,0,0,0,0]`), not a `1.0` scale-factor vector.
2. **Analytic-vs-FD cross-check** (`gd_jacobian_validate.py --free
   co2_ppm,t_offset_k --h-scan`): co2 validates as before (rel L2 ~1e-8 to
   1e-10, cosine 1.0 exactly). `t_offset_k` at the script's own default
   h-scan range (1e-3 to 1e-6 K) showed the OPPOSITE of the expected
   V-curve -- disagreement rising monotonically as h shrank, never
   bottoming out (worst case 1.7 rel L2 at h=1e-6). Root cause: those
   steps are calibrated for `kind="scale"` rows near O(1), far too small
   an ABSOLUTE Kelvin perturbation for this row's own numerical
   resolution. Rerun at `--h 0.1`: rel L2 1.1e-5 to 2.7e-5, cosine
   exactly 1.0 -- matching `p_surface_hpa`'s own documented gert-internal
   precision floor (2.1e-5, from `dtau_mol_dT_lay_hires` itself not being
   exact) almost exactly. Confirms the Jacobian is correct; the small-h
   scan was testing an under-resolved step, not a real bug.
3. **Smoke test**: one fast-subset window (rows 189-197), fully
   representable config (`--prior-fields exact`, co2/p_surface/t_offset_k
   free, ch4/co/h2o frozen exact at bin_centers -- no representability
   gap by construction). `resid_hires_rms` = 9.7e-6 (excellent fit).
   `t_offset_k` converged to within **9.3e-5 K mean / 2.4e-4 K max** of
   the true ~1.3-1.5K local sinusoid value -- essentially exact, matching
   the precision of the already-working co2/p_surface rows in the same
   solve (co2 mean 0.0085 ppm, p_surface mean 0.0095 hPa).
4. Forward-model agreement (`gd_jacobian_validate.py`'s own standing
   check): analytic-path vs. sweep-path prediction, rel L2 = 0.0 exactly.

**Conclusion**: temperature is now a real, working member of the joint
state -- retrievable, freezable, with a validated analytic Jacobian and a
correctly-wired `kind="absolute"` parameterization (now generally
available to any future absolute-valued row, not just this one). Not yet
run: any real experiment putting `t_offset_k` through the same class of
diagnostics every other row has already faced this session (prior-pull,
keystone-floor sensitivity, defocus interaction).

**Reversed-truth mechanism updated too** (2026-09-08, user: "make sure to
create this for the reversed truth as well" -- the Sec.7/8/9 keystone/
defocus control-experiment image, `scratch_work/whole_slit_truth_
reversed.pkl`, built before `t_offset_k` existed). Verified the generic
reversal wrapper (`_reversed1(fn) = lambda x: fn(-x)`, applied over
`als.STATE_FIELDS.items()`) already handles `t_offset_k` correctly with
zero code changes -- `reversed_t_offset_k(500) == original_t_offset_k
(-500)` exactly, same sanity check every other row in that mechanism
already gets. Rebuilt the reversed-truth image (old one backed up as
`whole_slit_truth_reversed.PRE_TOFFSET_backup.pkl`, not deleted); the new
image differs from the old by up to 0.17 radiance units (~1.2% relative)
-- a real, physically-sensible-magnitude change from the temperature-
driven optical-depth shift now present, not a rendering artifact.
`retrieval_co2palbedo_reversed_truth_anchorfrozen_gratio.py`'s own
`_merged_fields` comprehension already iterates `als.STATE_FIELDS`
generically too, so it now freezes `t_offset_k` at its exact (reversed)
truth value at every anchor automatically, the same as ch4/co/h2o --
no driver-script edit needed, only the truth-image rebuild.

## 11. Aerosol added as retrievable state (tau_aerosol, height_aerosol) (2026-09-08)

Direct infrastructure prerequisite for the user's own stated multi-FPA
coupling goal ("couple FPA0 with FPA2 and FPA3 (separately) to get a
column average that is responsive to aerosols and surface pressure
errors"): without an aerosol state row, FPA0 coupling cannot inform
anything aerosol-related at all. Checked directly: `gert.ForwardModel`
already has full aerosol support (`tau_aerosol`, `height_aerosol`,
`thickness_aerosol`, `ssa_aerosol`/`g_aerosol`/`P_aerosol`/
`qext_aerosol`, analytic `K_tau_aer_hires`/`K_aer_lay_hires`) but this
project's own scene layer (`along_slit_scene.py`) had never threaded any
of it through.

**Correction along the way** (user caught this directly): `tau_aerosol`
is NOT a band-independent physical quantity -- it is "total aerosol
optical depth at the O2-A REFERENCE WAVELENGTH" (`gert.ForwardModel.
run`'s own docstring). Different aerosol species have genuinely
different spectral extinction signatures (the Angstrom-exponent effect),
so the same physical aerosol layer produces a different optical depth in
different bands -- `gert` models this via `qext_aerosol` (per-window
normalised extinction, `tau_aerosol` scaled by this factor per
wavenumber), held fixed per this work but requiring a real aerosol
type's own spectral shape once multi-band coupling is built (not a flat
placeholder).

**Design**, mirroring the `t_offset_k` addition (Sec.10) closely: two new
`surface`-target rows (`tau_aerosol`, `height_aerosol` are passed
directly to `ForwardModel.run`, like `albedo`, never through
`AtmosphericProfile`) in `SURFACE_FIELDS`/`SURFACE_FIELDS_PRIOR` --
spatially-varying truth (0.05 background AOD + a moderate haze event,
mirroring every other row's background+localized-feature convention;
85000 Pa background height + a synoptic drift, phase-offset from
`t_offset_k`'s own sinusoid), flat structural priors, `kind="absolute"`
(reusing the generic machinery already built for `t_offset_k`). Fixed
aerosol microphysics (`AEROSOL_SSA`/`AEROSOL_G`/`AEROSOL_THICKNESS_PA`/
`AEROSOL_QEXT_NORM`, `along_slit_scene.py`): `smoke` (fresh biomass-
burning)'s band-1 (CO2-weak, closest available proxy in gert's 3-band
registry to GeoCarb's own CO2_strong) values from `gert.
aerosol_properties.get_aerosol_scalars`.

### Two real bugs found and fixed along the way (not in this project's own logic, but exposed by exercising it for the first time)

1. **`gd_jacobian_validate.py`'s own hand-duplicated FD-reference forward
   model silently omitted aerosol physics entirely** -- it never threads
   `tau_aerosol`/`height_aerosol` through at all, so the moment either
   row was free, the analytic path (which DOES include it) and this
   "FD reference" disagreed on EVERY row's own forward computation, not
   just the aerosol rows' -- caught directly when `co2_ppm`'s own
   forward-agreement check broke (rel L2 0.135, should be exactly 0).
   Fixed by threading the same aerosol kwargs through this script's own
   duplicated `make_spectrum()`.
2. **`P_aerosol` (the scattering phase function) has never been computed
   anywhere in this whole codebase -- not even in `gert`'s own sanity-
   check scripts.** Every `ForwardModel.run()` call before this work
   omitted it, silently defaulting to `np.zeros(n_wn)`
   (`forward_model.py`'s own fallback), which makes `I_scatter` --
   the ONLY mechanism `height_aerosol` can act through -- identically
   zero regardless of height. Found by tracing why `height_aerosol`'s
   own analytic Jacobian column measured ~1e-19 (floating-point noise,
   not a small-but-real derivative) despite `K_aer_lay_hires` and the
   layer-redistribution finite difference both being individually
   nonzero and correctly computed -- the two combine to EXACTLY cancel
   when `K_aer_lay` is constant across layers (true in the single-
   scatter solver, checked directly) and the Gaussian profile's own
   `aer_frac` is mass-conserving (`sum_l d(aer_frac[l])/d(height) = 0`
   exactly, by construction of the normalization). Fixed by implementing
   the real Henyey-Greenstein phase function (`along_slit_scene.
   aerosol_phase_hg`) using `gert.geometry.Geometry`'s own already-
   computed `scattering_angle`, wired into all three `fm.run()` call
   sites (`_make_state_spectrum`, `spectrum_jac`, and
   `height_aerosol_dI_dparam`'s own internal RT calls).

### `tau_aerosol`'s analytic Jacobian: fully validated

Reuses `surface_dI_dparam` completely unchanged (`K_tau_aer_hires` is
already a direct column derivative, no chain rule) -- just one new
`SURFACE_ROW_JACOBIAN` entry. Verified all four ways (mirroring Sec.10's
own template): `x0()`/prior round-trip; byte-identical when absent
(regression); analytic-vs-FD agreement ~2e-5 relative at an
appropriately-sized step (matching the same gert-internal precision
floor already documented for `p_surface_hpa`'s temperature path); smoke
test.

### `height_aerosol`'s analytic Jacobian: a genuine, structural non-smoothness in `gert`'s own forward model, not a bug to keep chasing

`height_aerosol` acts on radiance ONLY through `tau_abv` (gas+Rayleigh
optical depth ABOVE the aerosol layer, inside `I_scatter`'s own
`exp(-m*tau_abv)` term) -- `K_aer_lay_hires` alone cannot capture this
(see bug 2 above), so `height_aerosol_dI_dparam` is a genuine RT-level
finite difference (2 extra full `ForwardModel.run()` calls per anchor,
unlike every other row's Jacobian, which reuses the SAME `jacobians=True`
call already made for the nominal state) -- an accepted cost tradeoff
for correctness, since `gert` doesn't expose `I_scatter`/`tau_abv`/the
airmass factor as separate `ForwardResult` fields to compose more
cheaply.

After fixing the `P_aerosol` bug, this Jacobian is no longer
mathematically forced to zero -- it shows real, plausible-magnitude,
mostly-correctly-signed sensitivity. But it does not cleanly pass a
classic analytic-vs-FD validation, and an h-scan from 10 to 5000 Pa
shows why: **`ForwardModel.run`'s own `above_mask = atm.p_layers <
height_aerosol` is a hard BOOLEAN threshold**, not a smooth function --
`tau_abv` (and hence radiance) has a genuine step discontinuity every
time `height_aerosol` crosses one of the atmosphere's discrete layer
pressures. Both the analytic path (itself an internal FD at a fixed
~100 Pa step) and any external FD reference are each sampling a
DIFFERENT, discretization-dependent slice of a fundamentally non-smooth
function -- the erratic, non-converging disagreement across the h-scan
(rel L2 ranging 0.43-14+ with no clean monotonic trend either direction)
is the expected signature of that, not evidence either implementation is
wrong. This is a property of `gert`'s own Gaussian-aerosol-profile
implementation (a hard layer-count threshold where a smooth weighting
would be physically better-justified anyway), not this project's code.

**Decision (user's explicit call): keep the row, document the caveat**
rather than dropping it or chasing a clean analytic Jacobian further.
Practical implication: Gauss-Newton retrieval of `height_aerosol` may be
locally unreliable specifically near a layer-boundary crossing (a real
property of the current forward model, not something a better Jacobian
implementation on this project's own side could fix) -- worth keeping in
mind when interpreting any future `height_aerosol` retrieval result that
sits close to one of `atm.p_layers`' own discrete values.

### Smoke test -- caught a real convergence failure, not just a CLI gap

`--free co2_ppm,p_surface_hpa,tau_aerosol,height_aerosol --vary-albedo`
(fully representable `--prior-fields exact` config, fast-subset window).
**Real gap caught first**: `state_spec_from_scene`'s own surface-row gate
is `band_label = ... if ("albedo" in free or vary_albedo) else None` --
since neither `tau_aerosol` nor `height_aerosol` triggers this on their
own, `--vary-albedo` MUST be passed even when albedo itself isn't in
`--free`, or `tau_aerosol`/`height_aerosol` are silently absent from the
state entirely despite being named in `--free`. Not obvious from either
flag's own name; worth a clearer error message as a follow-up (today: no
explicit check catches "surface-target row in --free without
vary_albedo" the way the reverse -- "albedo in --free without
--vary-albedo" -- already has one).

**But the smoke test itself revealed something much bigger once run**:
`resid_hires_rms=0.147` (should be ~1e-5 for a fully-representable
config, matching every OTHER row's own smoke test this session) and
wildly wrong retrieved values (`co2_ppm` off by up to 47 ppm,
`tau_aerosol` retrieved NEGATIVE -- unphysical for an optical depth).
Isolating this (removing `height_aerosol` from `--free` gave a BIT-
IDENTICAL bad result) ruled out `height_aerosol`'s own known non-
smoothness as the cause -- something else was broken.

### Bug 3: `p_surface_hpa`'s existing (pre-aerosol) analytic Jacobian is missing a real cross-term

Re-running `gd_jacobian_validate.py` with `co2_ppm,p_surface_hpa,
tau_aerosol` together (never tested as a trio before -- only pairs)
found `p_surface_hpa`'s own Jacobian now disagrees with FD by ~6-10%
(cosine ~0.9997 -- direction right, magnitude wrong, not FD noise).
Root cause: `p_surface_dI_dparam` was written before aerosol existed.
Changing `p_surface_hpa` rescales the ENTIRE sigma-level pressure grid
(`atm.p_layers`), which shifts which physical layers count as "above"
the aerosol's own FIXED `height_aerosol` threshold
(`ForwardModel.run`'s own `above_mask = atm.p_layers < height_aerosol`)
-- a geometric effect the existing composition (built from
`K_mol_lay_hires`/`dtau_mol_dpscale_lay_hires`, which correctly handle
GAS-amount changes but know nothing about the pressure GRID itself
shifting) has no way to capture. Confirmed `K_mol_lay_hires` already
includes gert's OWN `above_frac`-weighted scattering term (`K_tau_lay`
from `rt_solver.py`, chained in at `forward_model.py`'s own
`K_mol_lay_this[mol] = K_tau_l + ...`) -- so the missing piece is
specifically the GEOMETRY shift, not the gas-amount path.

**Fix**: `p_surface_dI_dparam` now dispatches to the existing fast
analytic composition when no aerosol row is present (`tau_aer is None`
-- zero behavior change for every existing caller), or a genuine RT-
level finite difference (2 extra full `ForwardModel.run()` calls, same
accepted-cost pattern as `height_aerosol_dI_dparam`) whenever aerosol IS
present -- since `gert` doesn't expose `tau_abv`/`I_scatter`/the airmass
factor as separate fields, there is no cheaper way to compose just the
missing piece without risking double-counting the parts that were
already correct.

Re-validating after this fix still shows ~7-13% disagreement -- but this
is very likely the SAME hard-threshold non-smoothness already found and
accepted for `height_aerosol` (Sec.11's own finding above), now showing
up via `p_surface_hpa`'s indirect coupling to the identical `above_mask`
geometry, not a remaining bug in this fix. Re-running the smoke test
after this fix alone changed NOTHING (identical `resid_hires_rms` to 5
significant figures) -- a strong signal that `p_surface`'s own Jacobian
precision was not, in fact, the dominant problem.

### Bug 4: the actual truth-rendering path never threaded aerosol through at all

`gd_per_row_retrieve.py::_band_setup` -- the function that renders the TRUTH image
`--prior-fields exact` (and every standard, non-"dense-truth" run) is
scored against -- has its OWN THIRD independent inline `spectrum(atm_p,
surf_p)` closure, completely separate from both `_make_state_spectrum`
(`gd_joint_block_retrieve.py`, fixed earlier) and `spectrum_jac`
(`jacobians.py`, fixed earlier). This third copy's own `fm.run()` call
only ever passed `albedo`/`albedo_slope` -- no aerosol kwargs at all --
and `_band_setup`'s own `surf_params` construction hardcoded `{"albedo":
...}`, never extracting `tau_aerosol`/`height_aerosol` even when present
in the state. **This is a genuine truth-vs-model physics mismatch, not
a Jacobian precision issue**: the retrieval's forward model included
aerosol; the truth it was being scored against did not. This is very
likely why `tau_aerosol` wanted to go negative -- the model had MORE
optical depth than the (aerosol-free) truth, and GN pushed the aerosol
term toward zero/negative trying to compensate.

**Fix**: `spectrum`'s own `fm.run()` call now threads the same aerosol
kwargs every other fixed call site uses; `surf_params` now also pulls
`tau_aerosol`/`height_aerosol` directly from the RAW (row-name-keyed,
matching `als.SURFACE_FIELDS`'s own convention) `surface_fields`
parameter -- bypassing `als.build_scene_fields`'s own band-label-keyed
reshaping (which exists only for albedo's per-band/uniform/barcode
needs, irrelevant to aerosol). A related bug fixed alongside: `surf_p[
"albedo"]` would `KeyError` once `surf_p` could be non-empty without an
"albedo" key (the aerosol-only case) -- changed to `.get("albedo",
albedo)`.

**A real, separate caching gap found and worked around**: `_band_setup_
cached`'s own cache key has no dependence on the FIELD FUNCTIONS' own
content (only coarse metadata like `scene`/`vary_albedo`/`spatial_psf_
fwhm_px`) -- so a truth image rendered with the OLD, aerosol-blind code
was silently served as a "cache HIT" to a run using the NEW, fixed code,
with no error and no way to tell from the log alone. Caught by checking
file mtimes directly (the cache entry predated today's fix); worked
around by moving the 2 stale entries aside (`results/truth_cache_stale_
backup/`, not deleted) rather than deleting them outright. Not fixed at
the root -- a real, general gap (this project's OWN prior sessions
already flagged the identical class of issue and built a `resolution_
tag` escape hatch for ONE specific caller; this default code path has no
such protection) worth a proper fix later: hash the actual `fields`/
`surface_fields` dict CONTENTS into the key, not just whether they were
given.

### Where this stands after all four fixes

Real, measured improvement at each step (`resid_hires_rms`: 0.147 ->
0.147 (p_surface fix alone, no change) -> **0.101** (truth-rendering
fix) -- `tau_aerosol`'s own mean error roughly halved, 0.050 -> 0.026
ppm-equivalent AOD), but **still far from the ~1e-5-level convergence
every other row's own smoke test achieved this session**. At least one
more issue remains, not yet found -- candidates: another duplicated
spectrum-builder somewhere; a genuine Gauss-Newton conditioning problem
from real co2/p_surface/aerosol cross-talk (three free rows now
share overlapping sensitivity in ways none of this session's other
combinations have tested); or a residual effect of `height_aerosol`'s
own accepted non-smoothness bleeding into the joint solve even when it
is not itself free (the frozen row's own EXACT truth value still sits
near a layer boundary, feeding a discontinuous forward model regardless
of whether GN is solving FOR it).

**Explicitly scoped as a follow-up, not resolved today**: "aerosol
converges correctly in a real joint retrieval" needs more work before
any aerosol-related result from a whole-slit sweep should be trusted.
What IS trustworthy today: `tau_aerosol`'s own analytic Jacobian in
isolation (validated cleanly, Sec.11 above); the fact that truth
rendering and the retrieval's forward model are now at least physically
consistent with each other (bug 4); the general pattern this session
surfaced repeatedly -- **this codebase has multiple independent,
hand-duplicated copies of "build a spectrum from state parameters,"
and any new physics (aerosol today; conceivably anything else in the
future) must be threaded through EVERY one of them, with no single
choke point that guarantees this by construction.** Worth a real
refactor at some point (one shared spectrum-builder, not three), flagged
here rather than attempted today.

## 12. Spectrum-simulation consolidation + `gert` ForwardResult additions (2026-09-09)

Direct follow-through on Sec.11's closing flag. Inventory (grep + Explore
agent) found **9** independent `gert.ForwardModel(...)` + `.run(...)` call
sites, 6 of them near-byte-for-byte duplicates of the same ~15-line
aerosol-kwarg block -- exactly the pattern behind Sec.11's bugs 1, 3, 4.

**New module: `geocarb_gert/spectrum.py`** -- `simulate_spectrum(atm_params,
surface, absco, wide_inst, geo, solar, jacobians=False, aerosol_type=
"smoke")` (forward-only) and `spectrum_and_jacobian(...)` (analytic
Jacobian, relocated from `jacobians.make_spectrum_jac`, which is now a
thin wrapper). `surface` keeps the existing `{"albedo": ..., "tau_aerosol":
..., "height_aerosol": ...}` convention every site already used; albedo
fallback resolution stays the CALLER's job (documented in the module),
since `simulate_spectrum` has no scene-specific fallback to fall back to.
New `geocarb_gert/aerosol_defaults.py` wraps `gert.aerosol_properties.
get_aerosol_scalars` (band index 1, this project's single-window
convention) instead of re-deriving `ssa`/`g`/`qext_norm` per call site;
`aerosol_type="smoke"` reproduces the Sec.11 constants exactly.

**Migrated** (each verified bit-for-bit against its pre-migration output
before being left in place): `jacobians.make_spectrum_jac` (now delegates),
`gd_jacobian_validate.py`'s FD-reference closure, `gd_joint_block_
whole_slit_sweep.py`'s `_make_state_spectrum`, and `gd_per_row_retrieve.py::_band_
setup`'s truth-rendering closure -- the highest-value target, since this
is the one bug 4 found silently omitting aerosol for so long. Re-running
`gd_jacobian_validate.py --free co2_ppm,tau_aerosol,height_aerosol,
p_surface_hpa` end-to-end after the migration: **"forward agreement
(analytic path vs sweep path): rel L2 0.000e+00 OK"** -- no regression.

**Explicitly not migrated**: `gd_joint_block_retrieve.py`'s `spectrum_for`
builds its atmosphere via `StateVector.gas_scaling().apply()`, not
`als.atmosphere_from_params(**atm_params)` -- discovered during
implementation that this doesn't fit `simulate_spectrum`'s `atm_params`
dict contract without a broader interface change, so it was left alone
rather than forced. `gd_per_row_retrieve.py::_joint_retrieve`'s multi-band `fm` (handed
to `GERTRetrieval`, never bare `.run()`), `along_slit_scene._lookup_sample`
(deprecated, archive-only caller), and `scene.py::hires_spectra_for`
(doesn't build its own atmosphere) stay out of scope for the same reasons
identified during planning -- different object shapes, not oversights.

**`gert` changes** (additive, own repo, user-owned): `RTResult` gained
`I_scatter`/`tau_abv`/`m`; `ForwardResult` gained `airmass_hires`/
`tau_abv_hires`/`I_scatter_hires` (always populated, not gated on
`jacobians=True` -- cheap, already computed every call). Verified
`I_hires == I_direct + I_scatter_hires` to the Beer-Lambert identity
(rel L2 3.5e-17). This let `height_aerosol_dI_dparam` become a true
analytic composition (`d(I_scatter)/d(tau_abv) = -m*I_scatter`, `d(tau_
abv)/d(height)` from a cheap algebra-only re-mask of already-computed
per-layer optical depths -- **zero extra RT calls**, down from 2 full RT
calls) -- cross-checked against the old RT-FD version
(`height_aerosol_dI_dparam_fd`, kept for reference): cosine 0.998-0.9996,
magnitude differing 5-11% wherever the h-window straddled a real layer
crossing, consistent with (not worse than) Sec.11's already-documented
non-smoothness caveat, which is UNCHANGED by this -- `above_mask` is
still a hard boolean threshold; only the FD-step-size ambiguity is gone.
`p_surface_dI_dparam`'s aerosol branch was **not** converted to analytic
composition -- doing so correctly needs re-evaluating ABSCO optical
depths at a perturbed pressure (not just re-masking already-computed
per-layer arrays, `height_aerosol`'s simpler case), assessed as a
separate, riskier follow-up rather than attempted under this pass.

**Not done / open**: the full whole-slit smoke test (Sec.11's own
`resid_hires_rms` regression check) was not re-run end-to-end after this
migration -- cost (500-1500+s per run) vs. the strength of the equivalence
evidence already gathered (bit-for-bit `simulate_spectrum`/`spectrum_and_
jacobian` matches, plus the live `gd_jacobian_validate.py` forward-agreement
check at 0.000e+00) made it a reasonable line to stop at for this pass;
running it is the natural next verification step before trusting any new
aerosol retrieval result. Resolving Sec.11's own remaining convergence gap
is still a separate, open follow-up, unaffected by this consolidation.

### Retired code, once the consolidation made it visibly dead

- `along_slit_scene.build_lookup_radiance`/`_lookup_sample`/`_G_LOOKUP`
  (~150 lines) -- the old two-sample spectral-blend truth renderer,
  already superseded before this session by `gd_per_row_retrieve.py::_band_setup`'s
  own per-anchor rendering; confirmed no live caller (only `archive/`
  scripts). Not something today's consolidation newly obsoleted, just a
  dead-code opportunity in the same neighborhood, swept while here.
  Its now-dead-with-it imports (`multiprocessing`, `warnings`,
  `gd_render.available_cpus`) went too.
- `gd_joint_block_retrieve.py`'s top-level `ForwardModel`/
  `SingleScatterSolver` imports -- dead once `_make_state_spectrum`
  migrated to `simulate_spectrum`.
- **Kept, deliberately**: `jacobians.height_aerosol_dI_dparam_fd` (the
  superseded RT-FD version) -- cheap, and a ready-made cross-check if the
  analytic version is ever suspected of drifting.
- **Not retired**: `gd_per_row_retrieve.py`'s top-level `ForwardModel`/
  `SingleScatterSolver` imports stay -- still used by `_joint_retrieve`'s
  own multi-band `fm` (call site #4, out of scope for migration).

## 13. joint_block script family: renamed for clarity, retired the original single-window demo, folded merge/plot in (2026-09-09)

Follow-through on user questions about the `joint_block` script family's
own naming and duplication, prompted directly by Sec.12's renaming of
`gd_test.py` -> `gd_per_row_retrieve.py` (the per-row/native-pixel
multi-band stress test, contrasted against the `joint_block` family's
per-bin/shared-atmosphere strategy).

**`gd_test.py` -> `gd_per_row_retrieve.py`.** Mechanical rename (every
live, non-`archive/`/`scratch_work/` import/reference updated) -- no
behavior change. `archive/`/`scratch_work/` deliberately left referencing
the old name, as historical record.

**`gd_joint_block_retrieve.py` (the ORIGINAL, `StateVector.gas_scaling()`-
based single-window demo) retired.** Its role -- rows 890-935/G=15, only
`co2_scale` free, every other gas held at local truth -- is now a special
case of `gd_joint_block_whole_slit_sweep.py`'s own general mechanism via
two new CLI flags: `--row-min`/`--row-max` (restricts tiling to a
sub-range instead of the whole slit; `build_window_tiles` already
accepted these as function args, just never wired to argparse) and
`--bin-scheme {pixel-density,uniform}` (the `uniform` choice reproduces
the pre-pixel-density placement, kept for the diagnostics comparison
below). Along the way, fixed a real latent bug found while wiring this:
`_solve_window` unpacked `_SWEEP["gamma"]`/`["sigma_abs"]` into local
variables that were never actually used -- `state_spec_from_scene`'s own
`gamma` calls always used its own hardcoded default (3.0), silently
ignoring `--gamma`/`--sigma-abs` for every run ever made with this
script. Fixed by threading `gamma=gamma` through both `state_spec_from_
scene` calls (`--sigma-abs` still has no effect -- `sigmas=` is a
separate, also-dead parameter on that function, not fixed here, out of
scope for this pass).

**Discovered while retiring it: NOT just glue.** The original
inventory undercounted its own dependents -- `gd_joint_block_diagnostics.
py` imports `build_forward`/`gauss_newton_regularized`/`make_spectrum_fn`
from it too (the actual retrieval mechanism, not just `FPA`/`GERT_ROOT`/
`_eta_of`/`band_basics`), because diagnostics.py's own uniform-vs-pixel-
density bin-placement comparison deliberately reuses the ORIGINAL
simplified mechanism to keep the comparison controlled (bin placement
the only thing that varies). Retiring the demo script therefore required
porting diagnostics.py onto the general mechanism too, not just moving
some constants.

**Circular import found and fixed during the move.** `FPA`/`GERT_ROOT`/
`_eta_of`/`band_basics` moved into `gd_joint_block_whole_slit_sweep.py`
(which itself imports `pixel_density_bin_centers` FROM `diagnostics.py`)
while `diagnostics.py` needs to import `FPA`/etc. FROM the sweep script
-- a genuine cycle. Fixed at the root: `pixel_density_bin_centers` moved
to `geocarb_gert/joint_state.py` (its natural home, next to `state_spec_
from_scene`), so neither script depends on the other for it.

**`gd_joint_block_diagnostics.py` ported onto the general mechanism**
(`state_spec_from_scene` + `build_forward_state` + `gauss_newton_state`,
bin-centers-only, `free=("co2_ppm",)`, `prior_fields="exact"`, `prior_
form="tikhonov"` -- the tikhonov branch is documented as bit-identical to
the original regularization). **The headline capture-fraction number is
a known harness limitation, NOT a retrieval-code defect** (root-caused
2026-09-09, below); both schemes run to completion (rows 890-935,
G=15, ~900s each):

| scheme        | peak-enhancement capture | mean per-bin chi2 | min pixel count/bin |
|----------------|--------------------------|--------------------|----------------------|
| uniform        | 3034.1%                  | 0.008927           | 269                  |
| pixel-density  | 2939.3%                  | 0.008375           | 890                  |

Both a GOOD fit to the data (chi2 ~0.008-0.009) through a wildly
unphysical state, not a forward-model mismatch -- and both off by the
same ~30x order of magnitude, which argues for a systematic setup/scale
issue rather than per-scheme GN divergence. `pixel-density` still wins
on the metrics that don't depend on the broken absolute scale (lower
chi2, and a 3.3x higher minimum per-bin pixel count -- exactly the
starved-bin problem it exists to fix), so the qualitative finding this
script demonstrates may still hold, but the headline capture-fraction
number cannot be trusted either way. **Root cause (2026-09-09):** the
anomaly is confined to this diagnostics harness -- the production
retrieval path is sound. Evidence:

1. `gd_jacobian_validate.py` re-run clean -- forward analytic-vs-sweep
   rel L2 = 0.000e+00; analytic-vs-FD Jacobian rel L2 ~1e-9, cosine
   1.000000000 on every column. This directly disproves the earlier
   "`build_forward_state` Jacobian magnitude differs in absolute scale"
   hypothesis -- a 30x scale error in the shared forward/`K` would put
   a floor on that FD comparison, and there is none.
2. The `capture` metric -- `(retrieved_ppm[peak] - retrieved_ppm[edge])
   / true_enhancement` -- is a pure null-space quantity in this
   deliberately degenerate window (15 bins over 46 detector rows at the
   max-keystone end of FPA2, each row blending ~10 eta locations). GN
   moves it 30x while barely touching chi2 (0.008): a null-space
   excursion, not a broken forward model.
3. The metric was calibrated against the retired nearest-bin demo; the
   port runs on production `state_interp="linear"`, whose different
   inter-bin coupling in `K` steers GN down a different null-space
   direction. Metric/forward-convention mismatch, not a code defect --
   linear interp is the validated production choice.
4. Both schemes off by the *same* ~30x -> systematic (points 2-3), not
   stochastic GN divergence. Whole-slit sweeps against representable
   (Mode-2) truth reached single-digit-ppm CO2 errors through the same
   shared solver/forward -- a real 30x scale bug would have made those
   thousands of ppm.

**Non-blocking follow-up** (diagnostics harness only): the
capture-fraction number is not recoverable from a 15-bin/46-row window
regardless of code -- it is an under-determined inverse there by
construction. To make it meaningful, the script should use a
production-like bin density (G ~ width/3) and `prior_form="exponential"`,
or be retired in favour of the sweep's own scoring. Its residual /
chi2 / bin-placement panels don't depend on `retrieved_ppm`'s absolute
scale and are unaffected. Plot saved at
`plots/joint_block/gd_joint_block_diagnostics_fpa2_row890-935_G15.png`.

**`gd_joint_block_whole_slit_merge.py`/`_plot.py` folded into `gd_
joint_block_whole_slit_sweep.py` as `merge_parts`/`plot_sweep` functions**,
reachable via `--merge <parts_dir>`/`--plot <pkl>` (each gets its own
dedicated argument parser, checked before the main sweep parser, so
there's no flag-namespace collision with `--gamma`/`--fpa`/etc.). The
two standalone scripts are gone; every reference to them (docs, sbatch
scripts) updated to the new invocation.

**Then renamed**: `gd_joint_block_whole_slit_sweep.py` -> `gd_joint_
block_retrieve.py` (the name the original demo vacated). Every live
importer across the codebase (`gd_jacobian_validate.py`, `gd_build_
resolution_matched_truth.py`, `check_mission_config.py`, `gd_joint_
block_diagnostics.py`, the two `submit_smoke_*.sbatch` scripts) updated
-- since the import STATEMENTS (`from gd_joint_block_retrieve import
FPA, ...`) were already written against the target name, most needed no
change beyond the sweep-specific symbol imports (`ROW_KINDS`,
`build_window_tiles`, `state_spec_from_scene`, etc.) that previously came
from the old `..._whole_slit_sweep` module name.

**Verified**: every touched module imports cleanly and every CLI parses
(`--help` on each); `check_mission_config.py` (unrelated to any of this
directly, but imports the renamed module as `sweep`) still passes every
check end-to-end. `gd_jacobian_validate.py` re-run and clean (forward
rel L2 = 0.000e+00; Jacobian FD rel L2 ~1e-9) -- see the diagnostics
root-cause above.

**Critical files**: `scripts/gd_joint_block_retrieve.py` (the renamed,
consolidated script -- was `gd_joint_block_whole_slit_sweep.py`),
`scripts/gd_joint_block_diagnostics.py` (ported; capture-fraction
number is a known harness limitation, not a code defect),
`geocarb_gert/joint_state.py` (`pixel_density_bin_centers` moved here,
`gamma` wiring fix). `scripts/gd_joint_block_retrieve.py` (the OLD demo),
`gd_joint_block_whole_slit_merge.py`, `gd_joint_block_whole_slit_plot.py`
are deleted.

## 14. Aerosol still does not converge -- Sec.12's consolidation did not fix it, and the regression is broader than Sec.11 knew (2026-09-09)

Re-ran Sec.11's own single-window aerosol smoke test (rows 189-197, the
low-keystone fast subset Sec.10's `t_offset_k` smoke reached ~1e-5 on),
fully representable (`--prior-fields exact`), now that Sec.12 folded the
three hand-duplicated spectrum-builders into `geocarb_gert.spectrum` --
the exact root-cause refactor Sec.11's closing flag called for. It did
**not** close the gap. Config `--free co2_ppm,p_surface_hpa,tau_aerosol,
height_aerosol --vary-albedo`: `resid_hires_rms = 0.166` (Sec.11 left it
at 0.101; slightly worse now), vs the ~1e-5 every representable non-
aerosol solve reaches. `--no-truth-cache` gives a bit-identical result,
so the stale-truth-cache confound (Sec.11's own worked-around hazard) is
ruled out -- the cached truth was fine.

**Bisection (all rows 189-197, fully representable, `--jacobian
analytic` unless noted):**

| free rows | `resid_hires_rms` | co2 error |
|---|---|---|
| `co2_ppm` | 0.557 | **+110 ppm** |
| `co2_ppm, p_surface_hpa` | 0.532 | +50 ppm, p_surf +60 hPa |
| `+ tau_aerosol` (free) | 0.166 | +3 ppm |
| `+ height_aerosol` (free) | 0.166 | +3 ppm (height: zero AVK sensitivity) |
| `co2_ppm` + aerosol FROZEN at exact truth (`--vary-albedo`) | 0.229 | garbage (+48/-32/+85 ppm) |
| same, `--jacobian fd` | 0.229 | **bit-identical to analytic** |
| `co2_ppm`, **aerosol removed from `SURFACE_FIELDS`** | **2.0e-5** | **exactly 0** |
| `co2_ppm, p_surface_hpa`, aerosol removed | **1.7e-5** | ~0.003 ppm / 0.005 hPa |

**What this establishes:**

1. **The pre-aerosol machinery is perfect.** Delete the two aerosol
   entries from `along_slit_scene.SURFACE_FIELDS` and co2 / co2+p_surface
   snap to ~1e-5 with zero state error. Tiling, `--row-min/--row-max`
   single-window path, pixel-density bin placement, anchor grid,
   footprint integration, the analytic Jacobians for co2/p_surface --
   all sound.

2. **The regression is broader than "aerosol won't converge in a joint
   retrieval" (Sec.11's framing).** Sec.11 added `tau_aerosol`/`height_
   aerosol` to `SURFACE_FIELDS`, and `gd_per_row_retrieve.py::_band_
   setup` now *always* renders the truth image with background aerosol
   (AOD 0.05 everywhere, + a haze feature elsewhere) -- `_aer_fields =
   surface_fields if surface_fields is not None else als.SURFACE_FIELDS`,
   and the non-resolution-matched truth path passes `surface_fields=
   None`. But the retrieval forward only carries aerosol when a surface
   row exists (needs `--vary-albedo` or an aerosol/`albedo` row in
   `--free`). So **every representable retrieval that does not explicitly
   free an aerosol row is now scored against a truth containing 0.05 AOD
   its own forward model omits** -- co2 absorbs the missing optical depth
   (+110 ppm). The `_band_setup` comment claiming `surface_fields=None`
   "reproduces prior behavior exactly" is wrong: `None` falls back to
   `als.SURFACE_FIELDS`, which now contains the aerosol entries.

3. **Even with aerosol present and frozen at exact truth, the forward
   model does not reproduce the truth** (row 5: `resid 0.229`, co2
   garbage). Freeing `tau_aerosol` lets GN move it to 0.07/0.03/0.09
   (true value 0.05) to partially compensate -- the aerosol contribution
   has a different *shape*, not just amplitude, between the truth-render
   path and the retrieval-forward path.

4. **Not the Jacobian.** `--jacobian fd` gives a bit-identical bad result
   to `--jacobian analytic` (row 6). GN is converging correctly to a
   wrong minimum; the forward model itself is inconsistent.

5. **Not instrument/grid/plumbing.** `_band_setup` and `band_basics`
   build byte-identical `wide_win`/`wide_inst` (same `real_wavenumber_
   range`, `hires_spacing=0.01`, `channels_per_fwhm=3`, ILS fwhm), so
   `n_wn` and the `_build_aerosol_kwargs` flat arrays match. Both
   `render_at_anchors` (truth) and `build_forward_state` (retrieval)
   slice per-anchor *scalars* into the same `simulate_spectrum` ->
   `_build_aerosol_kwargs` -> `fm.run` path.

**Leading remaining hypothesis** (not yet confirmed): `height_aerosol`
(Pa) landing in different model layers between the two paths through the
hard aerosol-layer-boundary threshold Sec.11 bug 3 documented -- the
retrieval interpolates a G=3 frozen `height_aerosol` row linearly onto
anchors while the truth evaluates the continuous field, and if that
threshold sits between the window's true height range and the
interpolated one, the aerosol layer placement (and thus `tau_abv`, and
thus the whole `I_direct`/`I_scatter` split) jumps discontinuously. That
`height_aerosol` shows exactly zero AVK sensitivity in every run
(rows 4) is consistent: its forward response is a step, so its local
gradient is genuinely zero almost everywhere.

**Next step**: a direct single-spectrum A/B -- `_band_setup`'s own
`spectrum` closure vs `_make_state_spectrum`'s, called at one identical
aerosol state (same `atm_params`, same `tau_aerosol=0.05`, same
`height_aerosol`) -- and bisect where `I_hires` diverges. This is the
"aerosol converges in a real joint retrieval" follow-up Sec.11
explicitly scoped as its own task; it needs interactive iteration on the
RT internals, not another sweep.

**Mitigation shipped (2026-09-09, same day)** -- aerosol is now OPT-IN,
so every other config is genuinely aerosol-free again while the
convergence bug is chased:

- `gd_per_row_retrieve.py::_band_setup`/`_band_setup_cached` take a
  `with_aerosol` flag (default `False`); the `SURFACE_FIELDS` aerosol
  pull is gated on it. `with_aerosol` is folded into the truth-cache key.
- `gd_joint_block_retrieve.py` gains `--aerosol` (implied when
  `tau_aerosol`/`height_aerosol` is in `--free`). Off: both the truth
  render and the retrieval forward are aerosol-free, and the aerosol rows
  are stripped from the retrieval-side surface registry. On: `band_label`
  is forced so the frozen aerosol rows reach the forward model, matching
  the truth. Blocked with `--resolution-matched-*` (that truth builder
  has no aerosol fields).
- `TRUTH_CACHE_VERSION` 2 -> 3: every v2 entry was rendered WITH
  background aerosol and must be invalidated; also clears the
  Sec.11-bug-4 + Sec.12 stale-render hazard (both touched the render path
  without a bump).

Verified on rows 189-197, `--prior-fields exact`, `--jacobian analytic`:

| config | `resid_hires_rms` | co2 error |
|---|---|---|
| `--free co2_ppm` (default, no aerosol) | **2.0e-5** | **0** |
| `--free co2_ppm,p_surface_hpa` (default) | **1.7e-5** | ~0.003 ppm / 0.005 hPa |
| `--free co2_ppm --aerosol --vary-albedo` | 0.229 | garbage (bug, now flag-gated) |
| full aerosol smoke `--aerosol` | 0.166 | +3 ppm (bug, unchanged) |

The still-open question the A/B bisection must answer: with the root
cause understood, should background aerosol eventually become a *frozen
nuisance row always present* (like `p_surface`/`t_offset_k`), or stay
opt-in? Deferred until the forward-model inconsistency is fixed.

**Repro**: `scripts/submit_smoke_fpa2_aerosol.sbatch` (checked in, now
passes `--aerosol`). Bisection runs used `--row-min 189 --row-max 197
--prior-fields exact
--hires-only` with the `--free` sets above.

## 15. Aerosol non-convergence RESOLVED: it was a frozen-albedo representability confound, not an aerosol bug (2026-09-10)

User asked two things this session: (a) confirm the Sec.14 `--aerosol`
opt-in mitigation reproduces the pre-aerosol baseline when off, and (b)
rerun the imperfect-prior "free" experiments with the truth matched to
the retrieval's resolution. Chasing (b) surfaced the actual cause of
Sec.14's "aerosol still does not converge."

### (a) Aerosol-off reproduces the pre-aerosol baseline exactly

`submit_smoke_aerosol_onoff_impprior.sbatch` (job 1885853), rows
189-197, `--prior-fields exact`, HEAD `8c0c91d`:

| config | resid_hires_rms | co2 error | Sec.14 reference |
|---|---|---|---|
| `--free co2_ppm` (aerosol off) | 2.00e-5 | exactly 0 | 2.0e-5 / 0 ✓ |
| `--free co2_ppm,p_surface_hpa` (off) | 1.70e-5 | rms 0.004 ppm / 0.005 hPa | 1.7e-5 ✓ |
| `--aerosol` on (repro) | 0.166 | +0.36 ppm, tau 0.05->0.066 | 0.166 ✓ (still bad) |

Clean. The `--aerosol` opt-in is a correct mitigation.

### (b) The Sec.14 aerosol bisection was confounded by --vary-albedo

Every "aerosol broken" cell in Sec.14 ran `--vary-albedo` (forced -- an
aerosol run needs a surface row) with albedo FROZEN on the window's G=3
bin grid, scored against a dense truth whose albedo carries full
fine-scale texture. Every "clean" (~1e-5) cell ran CONSTANT albedo.
So the two groups differ in the albedo grid, not just in aerosol.

Diagnostic `submit_aerosol_confound_diag.sbatch` (job 1885933),
rows 189-197, `--prior-fields exact`:

| config | aerosol | resid_hires_rms |
|---|---|---|
| D1 `--vary-albedo`, frozen 3-bin albedo | none | 0.265 |
| D2 D1 + `--surface-positions anchor` | none | 0.309 (no better) |
| D3 aerosol frozen-exact + anchor albedo | yes | 0.267 |
| D4 aerosol frozen-exact, shared albedo (= Sec.14 D4) | yes | 0.229 |
| D5 free `tau_aerosol` + anchor albedo | yes | 0.078 |

`--vary-albedo` with a frozen albedo row gives resid ~0.27 with NO
aerosol at all. Adding aerosol changes nothing.

**Field probe** (window 189-197, 26.6 km span): `als._albedo_fine_field`
is `ALBEDO_COV=0.10`, `ALBEDO_CORR_KM=0.5` -- a 10%, 0.5 km-correlated
texture. Albedo swings 0.355->0.574 across the window; the 3 bin centers
all land near 0.47-0.48; `|dense - 3-knot interp|` rms = 0.047 (~10% of
the value). No retrieval grid short of ~0.5 km resolves it.

### The mechanism: coarse forward-model anchor grid cannot integrate a sub-grid albedo field, and neither pinning nor an oracle helps below its own spacing

Direct probes (`_make_state_spectrum` / `build_forward_state` /
`render_at_anchors` at the exact prior, aerosol correctly stripped):

| anchor density | plain frozen 3-knot albedo | + `--sub-bin-anomaly truth` |
|---|---|---|
| ad1 (~3 km) | resid(x0) 0.359 | 0.384 (WORSE -- aliasing) |
| ad16 (~0.19 km) | 0.359 | **0.0041** |

- `interp_to` with the anomaly oracle reconstructs the true albedo at
  rms EXACTLY 0.0 at every grid -- the mechanism is not buggy. The
  analytic Jacobian correctly treats the anomaly as an additive
  constant in state space.
- But the oracle only places correct values AT the anchor points; the
  footprint quadrature BETWEEN them is still piecewise-constant at the
  anchor spacing. When albedo structure (0.5 km) is finer than the
  anchor spacing (3 km at ad1), the oracle cannot help and slightly
  hurts (sampling a 0.5 km field at 3 km aliases). At ad16 (0.19 km <
  0.5 km) it works: resid collapses to the grid-match floor.
- Plain frozen 3-knot albedo has a ~0.36 residual floor at ANY anchor
  density -- refining the forward-model grid alone does nothing.

### Two real bugs in gd_joint_block_retrieve.py

1. **`--resolution-matched-*` renders a varying-albedo truth but did not
   force `args.vary_albedo`.** `vary_albedo=True` is hardcoded at the
   rm-truth `_band_setup_cached` call, but `_SWEEP["vary_albedo"] =
   args.vary_albedo` (the CLI flag, unforced). Without an explicit
   `--vary-albedo`, `band_label` stays `None` -> the retrieval carries NO
   albedo row and uses one constant scalar for the whole window ->
   ~0.37 residual vs the varying truth, co2 driven to ~400 ppm. The
   flag's own `--help` claims "Implies --vary-albedo"; it only implied it
   for the truth build. **Every A1/A2/C1 resolution-matched run this
   session was invalid for this reason** until caught (only C2, which
   passed `--vary-albedo` explicitly, was valid). **Fixed 2026-09-10**:
   `resolution_matched_active and not args.vary_albedo` now forces it on
   with a NOTE.
2. **`--resolution-matched-g-ratio-bins` at coarse G is not actually
   representable for a single window.** The whole-slit bin-center union
   (`whole_slit_bin_centers`, 341 points at g_ratio=3) interleaves
   OTHER windows' bin centers into this window's eta span (keystone --
   row-disjoint windows are not eta-disjoint), and a G=3 window state
   cannot reproduce those interleaved knots. That mode was only ever
   validated at g_ratio=1 (archived Sec.11). `--resolution-matched-
   anchor-density` has the same issue but far milder (anchors much
   denser than bins). Not fixed -- documented here; use anchor-density
   matching, or `--surface-positions anchor` + a per-window truth.

### Configured correctly, everything converges -- including aerosol

`submit_smoke_representable_final.sbatch` (job 1886xxx), rows 189-197,
`--anchor-density 16 --vary-albedo`.

**Part R** -- rm-ad16 truth, every frozen row on the ad16 anchor grid
(`--surface-positions anchor --frozen-atmosphere-positions anchor`):

| config | resid | co2 error (true 416.7) | p_surface error |
|---|---|---|---|
| exact prior, co2 free | 4.8e-3 | +0.01 ppm | -- |
| exact prior, co2+p_surface | 4.8e-3 | +0.03 ppm | +0.02 hPa |
| wrong-level prior (co2 +1%/p_surf -1%), co2+p_surface | 2.5e-2 | prior +4.2 -> **-0.5 ppm**; p_surf prior -10 -> **+2 hPa** | |

4.8e-3 is the grid-match floor (co2 on 3 knots + anchor quadrature).
The imperfect-prior case is the "watch convergence" result: GN pulls
co2 and p_surface almost exactly onto truth, small residual prior-pull.

**Part G** -- aerosol re-tested with the albedo gap removed (continuous
truth, ad16, free albedo + `--sub-bin-anomaly truth`):

| config | resid | co2 error | tau_aerosol (true 0.05) |
|---|---|---|---|
| free albedo + oracle, no aerosol | 4.1e-3 | -0.004 ppm | -- |
| + aerosol FROZEN at truth | 3.6e-3 | -0.004 ppm | 0.05 |
| + aerosol FREE (tau, height) | 3.2e-3 | +0.13 ppm | **0.054 (err 0.004)** |

**Aerosol converges perfectly once the scene is representable.**
Compare Sec.14: resid 0.17-0.23, tau -> 0.32, co2 off 47-110 ppm.

### Conclusion

Sec.14's "aerosol still does not converge" and its leading hypothesis
(`height_aerosol` layer-boundary threshold mismatch between the truth
and forward paths) were **wrong**. The aerosol forward model
(`geocarb_gert.spectrum.simulate_spectrum`, `_build_aerosol_kwargs`,
`P_aerosol`, the `above_mask` threshold) is fine. Every aerosol test
inherited a ~0.2-0.27 unrepresentable-albedo residual floor via the
forced `--vary-albedo`, and GN abused `tau_aerosol`/`height_aerosol`/
`co2_ppm` trying to fit it (tau -> 0.32, albedo -> 0.96 in the worst
case). The 4 bugs Sec.11 fixed were real but not the blocker.

**ALGORITHM_ROADMAP.md Sec.4 item 7 (aerosol non-convergence) is
closed.** Aerosol is a working member of the joint state, validated in
a representable-truth retrieval. Open follow-ups: (1) the g-ratio-bins
representability gap above; (2) whether background aerosol should become
a frozen nuisance row always present (like `p_surface`/`t_offset_k`) or
stay opt-in -- deferred in Sec.14, now unblocked; (3) aerosol has not
been run against a genuinely non-representable (dense) truth with a
proper sub-bin albedo treatment -- the representable results here are a
ceiling, not a production estimate.

**Repro**: `scripts/submit_smoke_aerosol_onoff_impprior.sbatch`,
`submit_aerosol_confound_diag.sbatch`,
`submit_aerosol_confound_isolate.sbatch`,
`submit_rmtruth_gr3_build.sbatch` + `submit_smoke_rmtruth_gr3.sbatch`,
`submit_rmtruth_ad16.sbatch` + `submit_smoke_rmtruth_ad16{,_anchorpos}.sbatch`,
`submit_smoke_representable_final.sbatch` (all checked in,
`scratch_work/smoke_*` / `scratch_work/aero_confound_*` outputs).

## 16. `impprior_ws` whole-slit sweep: c3/c4/c5 TIMEOUTs were a daemon-pool bug, not non-convergence (2026-09-11)

The whole-slit imperfect-prior sweep (`submit_impprior_ws.sbatch`, 58
windows x configs c1-c5, more free params per config -- see that
script's header) started hitting 12h TIMEOUTs (co-occurring with OOM
kills) on c3/c4/c5's wider windows. User: "I think the timeout is due
to the window sizes" -- true in direction but not the actual mechanism.

**Timing breakdown, isolated:** the completed-window logs (`rows
X-Y: G=Z t_hires=Ws`) show cost scaling superlinearly with width even
for c1/c2 (`t ~ G^1.3-1.5` log-log fit) and getting much worse with
more free rows (c1 mean 100s/bin, c2 154s/bin, c3 349s/bin, c4 288s/bin,
c5's one sample 1428s/bin) -- consistent with "bigger/more-parameter
windows cost more," but extrapolating c4's own narrow-window trend to
the widths that timed out (15-23 rows) only predicts ~2.5-4h, not 12h.

**Root cause, found by reproducing the exact stuck window** (rows
346-362, c4, `--free co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,
albedo`) live with `gauss_newton_state`'s `verbose` forced on and timing
wrapped around `forward()`/`jac.linearize()`: one `forward()` call took
16.6s, one `jac.linearize()` (the analytic GN Jacobian) took **1736.4s
-- ~100x**. `max_iter=15` outer GN iterations each need one
`linearize()`, so that alone approaches the 12h budget before counting
LM retries.

Why `linearize()` was ~100x `forward()`: confirmed directly (printed
`mp.current_process().daemon` from inside `_worker`) that **every
single-tile `--task-id` array task was running its one window inside a
daemon multiprocessing.Pool worker**. `main()` unconditionally wraps
`_worker` in `ctx.Pool(n_workers)` (`n_workers = args.n_workers or
available_cpus()`, i.e. 24 here) even when there is only one tile to
hand out -- `Pool()` pre-forks `n_workers` daemon processes regardless.
`anchor_spectra_and_derivs`'s and `linearize`'s own nested-pool guards
(`if mp.current_process().daemon: n_workers = 1`) then silently forced
**every** anchor-level RT call and every Jacobian L()-projection column
(one per free state scalar, `spec.n_free` of them) to run sequentially
in that one process, regardless of `--anchor-workers`. The existing
warning for this exact collision (`--anchor-workers has no effect
here`) only fires `if args.task_id is None` -- exactly backwards from
where the sweep lives, so it never printed for the jobs that were
actually affected.

**Fix** (`scripts/gd_joint_block_retrieve.py`, `main()`): when a task
owns exactly one tile (`len(tiles) == 1 and args.task_id is not None`),
skip the outer `Pool` and call `_worker` directly in-process, so it is
never a daemon and `--anchor-workers` can actually parallelize the one
window the task owns.

**Validated** by rerunning the exact stuck window (rows 346-362, c4,
`--anchor-workers 24`) against the fixed code: completed in **1008s**
(previously: no result -- 12h TIMEOUT + OOM kill every attempt), a
genuine converged solve (`resid_hires_rms=9.4e-4`,
`chi2_hires_reduced=4.7e-3`, `G_eff=385`), not a stub.

**Follow-up**: resubmit the outstanding c3/c4/c5 `impprior_ws` array
tasks (everything that TIMEOUT'd or never got past `JobArrayTaskLimit`)
now that they should finish in minutes rather than hours; audit
`submit_impprior_ws_prerender_aero.sbatch` and other single-shot
callers of `gd_joint_block_retrieve.py` for the same pattern.

## 17. Sec.16's fix unmasked an OOM problem; found and fixed one real duplication in `linearize`'s L()-loop (2026-09-11)

Resubmitting c3/c4/c5 with Sec.16's fix (real `--anchor-workers`
parallelism) OOM'd on nearly every task within ~1 minute against the
old `--mem=16G` (sized for the accidentally-serial pre-fix behavior).
Operational mitigation shipped first: `--anchor-workers` 8->4,
`--mem` 16G->128G (partition nodes have 504G) -- validated on a
moderate c3 window (19 rows, 3 free rows): completed in 2886s, peak
~41.5GB attributable, no plateau reached within the window's own
runtime (kept climbing step-wise, once per GN iteration).

User then asked directly: "are there duplicates in memory that could
be avoided?" Root-caused with a cheap, fast repro (task 0, the
SMALLEST window, 9 rows) instrumented with `resource.getrusage(...)
.ru_maxrss` at the top of `linearize()`/`anchor_spectra_and_derivs`:

- The MAIN process's own memory does NOT grow per GN iteration --
  iteration 0's `linearize()` jumped 3074MB->7554MB (one-time), but
  iteration 1's only moved 7554MB->7579MB (+25MB). No leak in the
  parent.
- Each of the 4 anchor-workers sat at ~8GB RSS even for this tiny
  9-row window -- but `/proc/<pid>/smaps_rollup` PSS (proportional,
  de-duplicates copy-on-write pages) showed the true UNIQUE share per
  worker was only ~0.4-0.7GB in an earlier probe on a different window
  -- most of that 8GB RSS is the ABSCO table (`gert.ABSCOTable.
  load_all`, eagerly `np.array(...)`-materialized once before any
  forking) correctly SHARED via copy-on-write, not duplicated. Naive
  per-process RSS summing overcounts shared pages; system-wide `free`
  `used` is the reliable signal, and PSS the reliable per-process one.

**One real, fixed duplication**: `linearize`'s L()-loop dispatched
`pool.map(_L_worker, jobs, chunksize=1)` where each job was `(dest,
field)` and `field` a full `(n_anchor, n_hires)` array -- ONE per free
state SCALAR (e.g. one per co2 bin, one per albedo bin; 30+ for
c3/c4). `multiprocessing.Pool.map` pickles task arguments through an
IPC pipe to reach workers EVEN under the `fork` context (fork only
governs how worker *processes* are created, not how individual task
payloads move) -- so every `field` array was serialized in the parent
and deserialized again in the worker, real copies neither COW nor
`anchor_spectra_and_derivs`'s own already-correct global-stash pattern
(used just above in the same module) needed to pay.

**Fix** (`geocarb_gert/jacobians.py`): `jobs` itself now goes into the
same fork-inherited `_LINEARIZE_L_G` global that already carried the
small metadata; `_L_worker` takes a plain integer index instead of the
`(dest, field)` tuple, and reads its own job from the global (inherited
via copy-on-write, no pickling) rather than having it pickled to it.
Only the small `(y.size,)` result `col` still crosses the IPC boundary,
same as before.

**Validated** on the same c3 window used above (rows 380-398,
`--anchor-workers 4`): peak attributable memory **41.5GB -> 23.3GB**
(-44%), runtime unchanged (2854s vs 2886s -- this only changes how data
moves, not what gets computed), and the result is bit-identical
(`resid_hires_rms`, `x_hires` to every printed digit) to both the
pre-fix run and a separately-checked c2 window's pre-existing
completed result. No correctness risk: same arithmetic, cheaper IPC.

**Not yet fixed** (open, deeper architectural item): both
`anchor_spectra_and_derivs` and the L()-loop create a BRAND NEW
`multiprocessing.Pool` on every single call, and `linearize()` is
called once per outer GN iteration (up to `max_iter=15`) -- up to ~30
fork/teardown cycles per window solve. This isn't shown to leak (the
parent plateaus, per above), but repeated forking of a live process
image is real, avoidable overhead; a persistent pool created once
before the GN loop and reused across iterations (fed fresh work via
lightweight IPC each iteration instead of a fresh fork) is the next
lever if `--mem=128G` ever proves insufficient for c4/c5's widest
windows.
