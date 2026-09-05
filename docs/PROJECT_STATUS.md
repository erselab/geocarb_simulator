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
