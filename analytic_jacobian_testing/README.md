# Analytic-Jacobian solver regression — CLOSED, 2026-08-18

This folder is a self-contained, closed-out record of the whole-slit
regression matrix that validated `geocarb_gert.jacobians` (the analytic
Gauss-Newton Jacobian, wired into `gauss_newton_state` via `jacobian_fn`)
against the existing finite-difference solver. **Result: PASS on all 16
runs.** Everything the matrix produced lives here rather than in the
project's shared `results/`/`docs/` pool, specifically so a future joint-block
run doesn't collide with or get confused for one of these closed runs.

This is a closed record, not a stale one — `gd_solver_regression.py` still
runs from here if it's ever needed again (a real code change to
`geocarb_gert/jacobians.py`, `joint_state.py`, or another scene revision
would be the reason to re-open it). It writes back into this same folder,
never back out into `results/`/`docs/`. See §5 below for what it does *not*
cover, which is exactly the situation that would call for that.

## 1. Why this exists

Finite-difference Jacobians have three real costs that motivated building
an analytic alternative (see `geocarb_gert/jacobians.py`'s own module
docstring for the full case): a step size that can interact badly with a
hard physical bound (exactly what happened before the along-slit scene's
`P_HEADROOM_HPA` fix — the FD probe crossed the standard-atmosphere ceiling
and killed 26% of a `p_surface`-free sweep), `n_free + 1` forward
evaluations per iteration instead of 1, and a structural block on
`kind="absolute"` state rows (which need their own step size, not the one
shared FD step `kind="scale"` gets away with).

The analytic path removes all three, assembling `dI_hires/dparam` from
`gert`'s own per-layer arrays (`ForwardModel.run(jacobians=True)`) composed
with this package's linear detector operator
(`gd_render.predict_neighborhood`) — no `gert` changes anywhere. Per-column
validation against finite differences (not full solves, individual
Jacobian columns) lives separately in `scripts/gd_jacobian_validate.py`,
which **stays in `scripts/`** as ongoing infrastructure — see §5.

This matrix is the next level up: does a *full Gauss-Newton solve* using
the analytic Jacobian converge to the same place as the existing
finite-difference solver, across the axes most likely to stress it.

## 2. The design matrix

Three factors, each with two levels, run against both solvers — a full
2×2×2×2 = 16-run matrix:

| factor | levels | why these two |
|---|---|---|
| free rows | `co2_ppm` (control) / `co2_ppm,p_surface_hpa` | CO2 and surface pressure are near-perfectly anti-correlated in this band alone (`run_free_state_sweeps.sh`'s own note) — freeing pressure is the harder, more degenerate case; CO2-only is the control that isolates whether that degeneracy is what's driving any disagreement |
| window count | 58 (`window_scale=1.0`) / 29 (`window_scale=2.9`) | 58 is every prior whole-slit run's tiling, unchanged; 29 is a new capability built this session (`scale_for_window_count`) — wider windows, more anchors per window, more free elements per solve |
| `g_ratio` | 1 (`G=width`) / 3 (`G=width/3`) | repeats `JOINT_BLOCK_MIGRATION_PLAN.md` §10's own `G=width` vs `G=width/3` production comparison, now against the analytic solver and the current scene |
| solver | `analytic` / `fd` | the actual thing under test |

Held fixed across every run, deliberately not swept here: `state_interp=True`
and `anchor_density=1` (the historically-validated `*_stateinterp` legacy
family's own settings), `uniform=False` (realistic scene). See §5 for what
that leaves untested.

Config naming: `{free}-{n_windows}-g{g_ratio}`, e.g. `co2p-29-g3` =
`p_surface_hpa` free, 29 windows, `g_ratio=3`.

## 3. Files in this folder

| file | what it is |
|---|---|
| `gd_solver_regression.py` | the driver — builds the 8 configs, runs each through both solvers via `scripts/gd_joint_block_whole_slit_sweep.py` as a subprocess, scores accuracy against truth and solver agreement, writes the two outputs below |
| `SOLVER_REGRESSION.md` | the generated report: accuracy-against-truth table (16 rows × 2 solves), solver-agreement table (8 configs, PASS/FAIL), wall-time table |
| `results/gd_solver_regression.pkl` | the scored/timings dict the report was built from — `{"scored": {(config, jacobian): {...}}, "timings": {...}, "tol": 1e-05}` |
| `results/<config>_<jacobian>.pkl` | the 16 raw sweep outputs (8 configs × {`analytic`, `fd`}) — full `gd_joint_block_whole_slit_sweep.py` payloads, one per config/solver pair, ~17 MB each |
| `plots/co2p-58-g1_analytic_state.png` | one illustrative `gd_joint_block_state_plot.py` render (CO2 + p_surface, retrieved vs. truth) from `co2p-58-g1_analytic.pkl` — a spot-check made while deciding which plotting script handles multi-row-free configs, not a systematic plot battery over the whole matrix |

## 4. Result

All 16 runs: **PASS**. Full numbers in `SOLVER_REGRESSION.md`; headline:

- **State agreement** (analytic vs. FD, worst relative difference over every
  detector row): `co2_ppm` 7×10⁻⁸ to 3×10⁻⁷, `p_surface_hpa` 0 to 7×10⁻⁸,
  across every config.
- **Residual agreement**: median per-row relative difference 1.8×10⁻⁷ to
  5×10⁻⁶.
- One config (`co2p-29-g3`) initially reported FAIL under the first version
  of the comparison metric (`abs(median_A − median_B)/median_A`, diffing two
  already-collapsed scalars) — traced to a rank-flip artifact of comparing
  medians near the CO2/p_surface degenerate ridge, not a real disagreement;
  confirmed by hand via the full per-row arrays (median relative difference
  3×10⁻⁶, consistent with every passing config). Fixed by comparing the
  *median of the per-row relative differences* instead — see `compare()`'s
  own docstring in `gd_solver_regression.py` for the full account.
- Timing: analytic and FD were within a few percent of each other across
  the matrix — the win here is correctness (no step size, `kind="absolute"`
  unblocked), not speed. See `geocarb_gert/jacobians.py`'s module docstring.

## 5. What this matrix does *not* cover

Real gaps, not hypothetical ones — worth knowing before trusting the
analytic solver outside exactly the settings tested here:

- **`anchor_density=4`** (the `*_adens4` legacy config family) — never run
  through the analytic solver at all, only `anchor_density=1`. Nothing in
  `jacobians.linearize` depends on anchor count, so it should work
  by construction, but "should" isn't "verified."
- **`uniform=True`** (the `uniform_gratio1` legacy sanity check, constant
  atmosphere everywhere) — also not covered.
- **`state_interp=False`** for hires — `gauss_newton_state`'s per-solve
  gating means analytic silently falls back to FD there (coarse still gets
  analytic; hires does not, and each solve's snapshot records which one it
  actually used via `jacobian_used`) — this matrix never exercises that
  fallback path because it always passes `--state-interp`.
- **Albedo (`surface` target) and dispersion (`instrument` target) rows** —
  `jacobians.py` implements and `gd_jacobian_validate.py` validates the
  albedo columns per-Jacobian, but no full-slit *solve* has ever freed
  albedo, and dispersion has no forward-model row to solve for yet at all
  (see `jacobians.py`'s own coverage table). Neither has been through
  anything like this matrix.
- **This scene only.** `along_slit_scene`'s truth changed twice during this
  same work session (`P_HEADROOM_HPA` 0.1→10.0 hPa, then Laprise eta → pure
  sigma levels in `geocarb_gert/levels.py`) — this matrix ran entirely on
  the scene as it stands after both changes. A third scene revision would
  need its own check, not an assumption that this PASS still holds.

Any of the above would be a legitimate reason to come back to this folder,
extend `gd_solver_regression.py`'s config matrix, and re-run — from here,
into here.
