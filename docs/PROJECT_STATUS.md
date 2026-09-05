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
