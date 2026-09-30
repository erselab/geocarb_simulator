"""One-off generator for notebooks/onboarding_retrievals_tour.ipynb (2026-09-27, user: a
walkthrough notebook for new collaborators -- environment checks, a single-band and a
multi-band example sized to run interactively, reading/plotting the .pkl results, and a
from-scratch minimal-window section so they can size an example for their own machine).
Run once to (re)generate the notebook; not imported by anything.
"""
import json
import uuid
from pathlib import Path


def md(*lines):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": _src(lines)}


def code(*lines):
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "execution_count": None, "metadata": {}, "outputs": [], "source": _src(lines)}


def _src(lines):
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


cells = []

cells.append(md(
"# GeoCarb joint-retrieval simulator -- onboarding tour",
"",
"This notebook is a guided, runnable tour of `geocarb_simulator` for a new collaborator: it checks your",
"environment, runs one small **single-band** retrieval and one small **multi-band** retrieval interactively,",
"shows you how to read and plot the `.pkl` result files the real sweeps produce, and ends with a from-scratch",
"walkthrough of the pieces you'd assemble to size a **minimum working example (MWE)** on your own window/state",
"vector.",
"",
"**Terminology, up front** (so the code below reads naturally):",
"- **FPA** -- one of the four detector focal planes: FPA0 (O2-A, 0.765 um), FPA1 (CO2 weak, 1.61 um), FPA2 (CO2",
"  strong, 2.06 um), FPA3 (CH4/CO, 2.32 um). Each is `1024` detector columns x `1024` rows.",
"- **Tile** -- a contiguous range of detector rows solved jointly (what the code itself still calls a",
"  \"window\" in most function/flag names -- `--row-min/--row-max`, `build_window_tiles*` -- but we say \"tile\"",
"  when talking about it, to keep \"window\" free for `gert`'s own `SpectralWindow`/spectral windows).",
"- **State vector** -- the physical quantities being retrieved (or held fixed) inside a tile: trace gases",
"  (`co2_ppm`, `ch4_ppb`, `co_ppb`), `p_surface_hpa`, `h2o_surface_vmr`, `t_offset_k`, per-band `albedo`, and,",
"  when aerosol is on, `amplitude_aerosol`/`height_aerosol`. Each row of the state vector sits at one or more",
"  positions along the slit (in along-slit km, called `eta` internally) and carries a **prior** (mean +",
"  covariance) and a **truth** (what the synthetic scene actually has there) -- retrieving means starting from",
"  the (imperfect) prior and using simulated radiances to move toward the (unknown-to-the-retrieval) truth.",
"",
"**Before you run this**, read the next two cells (environment checks, then RAM/time budget) -- do not skip",
"straight to \"run the retrieval\"; a wrong environment fails confusingly deep inside an RT call, and an",
"oversized tile can exhaust your node's memory.",
))

cells.append(md(
"## 1. Environment checks",
"",
"Run this cell first. It should print `all checks passed` at the end; if anything fails, fix it before going",
"further -- everything downstream assumes it.",
))

cells.append(code(
"import os",
"import sys",
"from pathlib import Path",
"",
"# ---- 1a. Repo + PYTHONPATH -----------------------------------------------------------",
"# Adjust these two if your checkout lives somewhere else -- everything else in this",
"# notebook is written relative to them, never hardcoded again below.",
"REPO_ROOT = Path(\"/scratch/scrowel3_lab/geocarb_simulator\")",
"GERT_HINT = Path(\"/scratch/scrowel3_lab/gert\")   # only used if geocarb_gert.paths can't find it itself",
"",
"assert REPO_ROOT.is_dir(), f\"repo not found at {REPO_ROOT} -- edit REPO_ROOT above\"",
"for p in (str(REPO_ROOT), str(REPO_ROOT / \"scripts\")):",
"    if p not in sys.path:",
"        sys.path.insert(0, p)",
"os.environ.setdefault(\"PYTHONPATH\", f\"{REPO_ROOT}:{GERT_HINT}\")",
"print(f\"python executable : {sys.executable}\")",
"print(f\"python version    : {sys.version.split()[0]}\")",
"",
"# ---- 1b. Can we import the project's own packages? -----------------------------------",
"try:",
"    import gert  # noqa: F401  -- the RT/retrieval library geocarb_simulator is built on",
"    import geocarb_gert  # noqa: F401",
"    from geocarb_gert.paths import describe_gert_root",
"    print(\"import gert, geocarb_gert : OK\")",
"except ImportError as e:",
"    raise SystemExit(\n        f\"Could not import gert/geocarb_gert ({e}). Make sure you're running the 'analysis' conda\\n\"",
"        f\"        environment's own Jupyter kernel, and that {GERT_HINT} (or wherever your gert checkout\\n\"",
"        \"        lives) is reachable -- see geocarb_gert/paths.py's own docstring for how it's located.\")",
"",
"# ---- 1c. The big input files gert needs (absco.h5 ~2.3 GB, solar.h5 ~33 MB) -----------",
"print(describe_gert_root())",
"absco_path = None",
"from geocarb_gert.paths import gert_root",
"absco_path = gert_root() / \"input/absco/absco.h5\"",
"solar_path = gert_root() / \"input/solar/solar.h5\"",
"for p in (absco_path, solar_path):",
"    assert p.exists(), f\"missing {p} -- this notebook cannot run any retrieval without it\"",
"    print(f\"  {p}  ({p.stat().st_size / 1e9:.2f} GB)\")",
"",
"# ---- 1d. CPUs and RAM actually available to *this* process ----------------------------",
"import multiprocessing as mp",
"n_cpus = mp.cpu_count()",
"try:",
"    import resource",
"    # ru_maxrss is peak-so-far, not a limit; the real number we want is /proc/meminfo.",
"except ImportError:",
"    pass",
"mem_total_gb = mem_avail_gb = None",
"try:",
"    with open(\"/proc/meminfo\") as f:",
"        info = dict(line.split(\":\", 1) for line in f if \":\" in line)",
"    mem_total_gb = int(info[\"MemTotal\"].strip().split()[0]) / 1e6",
"    mem_avail_gb = int(info[\"MemAvailable\"].strip().split()[0]) / 1e6",
"except Exception:",
"    pass",
"print(f\"CPUs visible to this process : {n_cpus}\")",
"if mem_avail_gb is not None:",
"    print(f\"RAM total / available on this node : {mem_total_gb:.0f} GB / {mem_avail_gb:.0f} GB\")",
"else:",
"    print(\"could not read /proc/meminfo (not on Linux?) -- check your RAM another way, see Section 2\")",
"",
"# ---- 1e. Are we actually on an interactive allocation, not a login node? --------------",
"slurm_job = os.environ.get(\"SLURM_JOB_ID\")",
"if slurm_job:",
"    print(f\"Running inside Slurm job {slurm_job} \"",
"          f\"(cpus_on_node={os.environ.get('SLURM_CPUS_ON_NODE', '?')}, \"",
"          f\"nodelist={os.environ.get('SLURM_NODELIST', '?')})\")",
"else:",
"    print(\"WARNING: no SLURM_JOB_ID in the environment -- if this is a shared login node rather than an\\n\"",
"          \"         interactive salloc/srun session or a Jupyter kernel launched from one, the examples\\n\"",
"          \"         below may be slow or get killed for using too much CPU/RAM on a shared node.\")",
"",
"print(\"\\nall checks passed\" if (mem_avail_gb is None or mem_avail_gb > 4) else \"\\nWARNING: very little RAM available\")",
))

cells.append(md(
"## 2. RAM and time budget -- read before running anything below",
"",
"- **Loading the inputs alone costs real memory.** `absco.h5` (the absorption cross-section lookup table) is",
"  ~2.3 GB on disk and is loaded whole into memory (`gert.ABSCOTable.load_all`); with `solar.h5` and Python/",
"  numpy overhead, budget **at least 6-8 GB just to import and load inputs**, before any retrieval runs.",
"- **Each parallel \"anchor worker\" adds its own RT-solver memory.** Retrievals here parallelize over spatial",
"  anchor points (`anchor_workers=N` forks `N` worker processes, each running the radiative-transfer solver",
"  independently). Budget roughly **1-3 GB per anchor worker** for a small, no-aerosol tile like the examples",
"  below (`solver=\"single_scatter\"`); if you turn aerosol on, the `xrtm` multiple-scattering solver used",
"  instead is far more expensive in both time (~30x slower per call) and memory -- do not experiment with",
"  aerosol interactively without first reading `docs/PROJECT_STATUS.md`'s aerosol sections.",
"- **Time scales with tile width x number of bands x number of free state-vector rows.** The two examples",
"  below (single-band, ~46 rows; multi-band, ~23+23 rows) are deliberately small and should each finish in a",
"  few minutes on 4-8 anchor workers. A *whole-slit* sweep (all ~33 tiles across the full 1024 rows) is a",
"  batch job, not an interactive one -- see `scripts/submit_multiband_sweep.sbatch` and Section 6 below.",
"- **Start with fewer anchor workers than CPUs you have**, especially the first time you run this notebook --",
"  `anchor_workers=4` is a safe starting point on a shared node; raise it once you've watched one run's actual",
"  memory footprint (e.g. `htop` in another terminal, or wrap the call in `/usr/bin/time -v` from a shell).",
))

cells.append(md(
"## 3. Load the shared driver code",
"",
"Both examples below go through **one** function, `gd_multiband_window.solve_window_multiband`, which despite",
"its name handles the single-band case too (pass it a dict with just one FPA) -- there is deliberately no",
"separate single-band code path to learn. It builds the synthetic scene, runs the forward radiative-transfer",
"model, and does the Gauss-Newton retrieval, returning one Python `dict` you can inspect directly or `pickle`",
"to disk (exactly what the real sweeps do -- the `.pkl` files under `results/` have this same shape).",
))

cells.append(code(
"import time",
"import pickle",
"",
"import numpy as np",
"import matplotlib.pyplot as plt",
"",
"import gd_multiband_window as gmw            # the shared single-/multi-band retrieval driver",
"import gd_joint_block_retrieve as gjr        # FPA row/column geometry, tiling helpers -- NOTE: this module",
"                                              # itself calls matplotlib.use(\"Agg\") at import time (its own CLI",
"                                              # plotting is headless-only), which is why we re-select the inline",
"                                              # backend explicitly below, AFTER these imports rather than before.",
"from geocarb_gert import along_slit_scene as als   # the synthetic scene's truth + prior functions",
"",
"# Loading absco/solar (Section 2's ~2.3 GB) happens here, once, and is reused by both examples below.",
"INPUTS = gmw.load_inputs()",
"print(\"inputs loaded (absco, solar, geometry, background atmosphere)\")",
))

cells.append(code(
"%matplotlib inline",
"# Re-select the inline backend now that gd_joint_block_retrieve's own matplotlib.use(\"Agg\") (above) has run --",
"# this cell must come AFTER that import, not before, or Agg wins and plt.show() below silently no-ops.",
))

cells.append(md(
"## 4. Example 1 -- a single-band retrieval",
"",
"We retrieve on FPA2 (CO2 strong) rows 374-396 -- the FPA2 half of \"tile 12\" of the standard FPA0+FPA2 tiling,",
"a small, already-validated 23-row range (no aerosol, so the cheaper `single_scatter` solver applies; the CLI",
"driver picks this automatically, but since we're calling the Python function directly we set it explicitly).",
"The state vector free set is the usual five rows: `co2_ppm`, `p_surface_hpa`, `h2o_surface_vmr`, `t_offset_k`,",
"and this band's own `albedo` (each starts from the `\"realistic\"` prior, which -- unlike `\"structural\"` --",
"never exactly equals the truth, so convergence is a real test rather than a no-op).",
))

cells.append(code(
"FREE_SINGLE = [\"co2_ppm\", \"p_surface_hpa\", \"h2o_surface_vmr\", \"t_offset_k\", \"albedo\"]",
"",
"t0 = time.time()",
"result_single = gmw.solve_window_multiband(",
"    {2: (374, 396)},               # {fpa: (row_lo, row_hi)} -- FPA2 only (the FPA2 half of tile 12 below)",
"    FREE_SINGLE,",
"    inputs=INPUTS,",
"    prior_fields=\"realistic\",",
"    solver=\"single_scatter\",      # no aerosol here -- see Section 2 before ever passing solver=\"xrtm\"",
"    anchor_workers=4,              # raise once you've checked your own memory headroom (Section 2)",
"    verbose=False,",
")",
"print(f\"wall time (this cell): {time.time() - t0:.0f} s\")",
"print(f\"reported t_solve={result_single['t_solve']:.0f} s, n_free={result_single['n_free']}, \"",
"      f\"resid_rms={result_single['resid_rms']:.3e}\")",
))

cells.append(md(
"Save it exactly the way a real sweep would -- one `.pkl` per tile, under `results/`:",
))

cells.append(code(
"OUT_DIR = REPO_ROOT / \"results/onboarding_examples\"",
"OUT_DIR.mkdir(parents=True, exist_ok=True)",
"single_pkl = OUT_DIR / \"example_singleband_fpa2_r374-396.pkl\"",
"with open(single_pkl, \"wb\") as f:",
"    pickle.dump(result_single, f)",
"print(f\"saved {single_pkl}\")",
))

cells.append(md(
"## 5. Example 2 -- a multi-band (joint) retrieval",
"",
"Same idea, but now FPA0 (O2-A) and FPA2 (CO2 strong) are retrieved **jointly** -- one shared state vector for",
"the gas/surface rows, with each band keeping its own `albedo` row. We use \"tile 12\" of the standard",
"FPA0+FPA2 tiling (`rows_by_fpa = {0: (378, 400), 2: (374, 396)}`) -- the exact tile used throughout this",
"project's own aerosol/optimizer diagnostics, so it's known-good and fast (previously measured at ~390 s for",
"this free set on this solver).",
))

cells.append(code(
"FREE_MULTI = [\"co2_ppm\", \"p_surface_hpa\", \"h2o_surface_vmr\", \"t_offset_k\", \"albedo\"]",
"",
"t0 = time.time()",
"result_multi = gmw.solve_window_multiband(",
"    {0: (378, 400), 2: (374, 396)},   # tile 12 of the FPA0+FPA2 tiling -- see Section 7 for where this comes from",
"    FREE_MULTI,",
"    inputs=INPUTS,",
"    prior_fields=\"realistic\",",
"    solver=\"single_scatter\",",
"    anchor_workers=4,",
"    g_ratio=1.0,                       # 1 retrieval bin per reference-band row; see Section 7",
"    verbose=False,",
")",
"print(f\"wall time (this cell): {time.time() - t0:.0f} s\")",
"print(f\"reported t_solve={result_multi['t_solve']:.0f} s, n_free={result_multi['n_free']}, \"",
"      f\"resid_rms={result_multi['resid_rms']:.3e} (by band {result_multi['resid_rms_by_band']})\")",
"",
"multi_pkl = OUT_DIR / \"example_multiband_fpa0-2_tile12.pkl\"",
"with open(multi_pkl, \"wb\") as f:",
"    pickle.dump(result_multi, f)",
"print(f\"saved {multi_pkl}\")",
))

cells.append(md(
"## 6. Reading the `.pkl` files and plotting results",
"",
"Both files just saved (and every whole-slit sweep's per-tile `.pkl` under `results/realistic_prior/multiband/`)",
"share the same shape: the top level is run metadata (`rows_by_fpa`, `free`, `solver`, `t_solve`, `resid_rms`,",
"...); `result[\"joint\"]` is the retrieved `StateSpec` snapshot, and `result[\"joint\"][\"params\"][row_name]` is a",
"dict with:",
"",
"- `positions` -- where along the slit this row's own bins sit, in normalized eta (multiply by",
"  `als.SLIT_HALF_KM` to get along-slit km -- the helper below does this for you);",
"- `values` -- the **retrieved** value at each position;",
"- `prior` -- the **prior mean** at each position (what the retrieval started from);",
"- `free` -- whether this row was actually retrieved (`True`) or held fixed at its prior (`False`);",
"- `kind`/`sigma` -- how the row is parameterized internally (not needed for basic plotting).",
"",
"The **truth** isn't stored in the file (it's a synthetic scene, defined in code, not data) -- get it back from",
"`geocarb_gert.along_slit_scene`, keyed the same way the scene itself is: `als.STATE_FIELDS[name](x_km)` for",
"gas/surface-atmosphere rows, `als.SURFACE_FIELDS[\"albedo\"](x_km, band_label)` for a band's own albedo row",
"(its name in `params` is `\"albedo\"` for a single-band result, or `f\"albedo_{band_label}\"` for a multi-band",
"one -- e.g. `\"albedo_O2_A\"`, `\"albedo_CO2_strong\"`).",
))

cells.append(code(
"def truth_of(name, x_km, band_label=None):",
"    \"\"\"The synthetic scene's TRUE value of state-vector row `name` at along-slit position(s) `x_km`.\"\"\"",
"    if band_label is not None:",
"        return np.asarray(als.SURFACE_FIELDS[\"albedo\"](x_km, band_label))",
"    if name in als.SURFACE_FIELDS:",
"        return np.asarray(als.SURFACE_FIELDS[name](x_km))",
"    return np.asarray(als.STATE_FIELDS[name](x_km))",
"",
"",
"def row_frame(result, name, band_label=None):",
"    \"\"\"positions [km], retrieved values, prior values, truth values -- one dict, ready to plot.\"\"\"",
"    p = result[\"joint\"][\"params\"][name]",
"    x_km = np.asarray(p[\"positions\"], dtype=float) * als.SLIT_HALF_KM",
"    return dict(x_km=x_km, retrieved=np.asarray(p[\"values\"], dtype=float),",
"                prior=np.asarray(p[\"prior\"], dtype=float), truth=truth_of(name, x_km, band_label),",
"                free=bool(p.get(\"free\", True)))",
"",
"",
"def plot_row(ax, result, name, band_label=None, title=None):",
"    d = row_frame(result, name, band_label)",
"    ax.plot(d[\"x_km\"], d[\"truth\"], \"k-\", lw=1.5, label=\"truth\")",
"    ax.plot(d[\"x_km\"], d[\"prior\"], \"--\", color=\"tab:gray\", lw=1.2, label=\"prior\")",
"    ax.plot(d[\"x_km\"], d[\"retrieved\"], \"o-\", color=\"tab:blue\", ms=3, lw=1, label=\"retrieved\")",
"    ax.set_xlabel(\"along-slit position [km]\")",
"    ax.set_title(title or name)",
"    ax.legend(fontsize=8)",
))

cells.append(code(
"with open(single_pkl, \"rb\") as f:",
"    d = pickle.load(f)",
"",
"# A single-band result still names its albedo row by band label (e.g. \"albedo_CO2_strong\"), never plain",
"# \"albedo\" -- `--free albedo` is just shorthand for \"this band's own albedo row, whatever it's called\".",
"rows_to_plot_single = [\"co2_ppm\", \"p_surface_hpa\", \"h2o_surface_vmr\", \"t_offset_k\", \"albedo_CO2_strong\"]",
"fig, axes = plt.subplots(1, len(rows_to_plot_single), figsize=(4 * len(rows_to_plot_single), 3.2))",
"for ax, name in zip(axes, rows_to_plot_single):",
"    band_label = \"CO2_strong\" if name.startswith(\"albedo\") else None",
"    plot_row(ax, d, name, band_label=band_label, title=name)",
"fig.suptitle(\"Example 1: single-band (FPA2) retrieval, rows 374-396\")",
"fig.tight_layout()",
"plt.show()",
))

cells.append(code(
"with open(multi_pkl, \"rb\") as f:",
"    d = pickle.load(f)",
"",
"rows_to_plot = [\"co2_ppm\", \"p_surface_hpa\", \"h2o_surface_vmr\", \"t_offset_k\", \"albedo_O2_A\", \"albedo_CO2_strong\"]",
"fig, axes = plt.subplots(2, 3, figsize=(14, 6))",
"for ax, name in zip(axes.ravel(), rows_to_plot):",
"    band_label = \"O2_A\" if name == \"albedo_O2_A\" else (\"CO2_strong\" if name == \"albedo_CO2_strong\" else None)",
"    plot_row(ax, d, name, band_label=band_label, title=name)",
"fig.suptitle(\"Example 2: joint FPA0+FPA2 retrieval, tile 12\")",
"fig.tight_layout()",
"plt.show()",
"",
"print(f\"per-band residual rms: {d['resid_rms_by_band']}\")",
))

cells.append(md(
"### 6a. Error histograms and a pairplot",
"",
"Two more views of the same multi-band result, both common enough in this project's own sweep-analysis",
"scripts to be worth knowing from the start:",
"",
"- A **histogram** of `retrieved - truth` for each row -- shows whether a row's error is tight and centred on",
"  zero (a healthy retrieval) or biased/spread out, without needing to eyeball a noisy along-slit curve.",
"- A **pairplot** -- every row's error plotted against every other row's error -- surfaces real degeneracies",
"  directly (e.g. a surface-pressure error that correlates with an aerosol-amplitude error means the two are",
"  compensating for each other) rather than needing to read a posterior covariance matrix.",
"",
"The pairplot needs every row on the SAME set of along-slit positions, but rows are retrieved on different",
"grids (albedo is on the finest one -- see Section 6's own schema note) -- so we interpolate the coarser rows",
"onto albedo's own grid first (`np.interp`; already smooth, since it's a coarse retrieval grid to begin with).",
))

cells.append(code(
"import pandas as pd",
"",
"with open(multi_pkl, \"rb\") as f:",
"    d = pickle.load(f)",
"",
"hist_rows = [(\"co2_ppm\", None), (\"p_surface_hpa\", None), (\"h2o_surface_vmr\", None), (\"t_offset_k\", None),",
"            (\"albedo_O2_A\", \"O2_A\"), (\"albedo_CO2_strong\", \"CO2_strong\")]",
"fig, axes = plt.subplots(2, 3, figsize=(14, 6))",
"for ax, (name, band_label) in zip(axes.ravel(), hist_rows):",
"    err = row_frame(d, name, band_label)",
"    e = err[\"retrieved\"] - err[\"truth\"]",
"    ax.hist(e, bins=20, color=\"tab:blue\", alpha=0.8)",
"    ax.axvline(0.0, color=\"k\", lw=0.8)",
"    ax.set_title(f\"{name} error  (mean {e.mean():.3g}, sd {e.std():.3g})\", fontsize=9)",
"fig.suptitle(\"Error histograms: joint FPA0+FPA2 retrieval, tile 12\")",
"fig.tight_layout()",
"plt.show()",
))

cells.append(code(
"import seaborn as sns",
"",
"# One shared eta grid (albedo's own, the finest) -- interpolate the coarser rows onto it.",
"x_fine = row_frame(d, \"albedo_O2_A\", \"O2_A\")[\"x_km\"]",
"pair_df = {}",
"for name, band_label in hist_rows:",
"    r = row_frame(d, name, band_label)",
"    e = r[\"retrieved\"] - r[\"truth\"]",
"    pair_df[name] = e if name.startswith(\"albedo\") else np.interp(x_fine, r[\"x_km\"], e)",
"pair_df = pd.DataFrame(pair_df)",
"",
"g = sns.PairGrid(pair_df, height=1.8)",
"g.map_diag(plt.hist, bins=15)",
"g.map_offdiag(plt.scatter, s=8, alpha=0.5)",
"g.figure.suptitle(\"Error pairplot: joint FPA0+FPA2 retrieval, tile 12\", y=1.02)",
"plt.show()",
))

cells.append(md(
"## 7. Where the example tiles/windows came from, and how to define your own",
"",
"Both examples above used a tile someone else already picked. This section shows the machinery behind that",
"choice, so you can size a genuinely NEW minimum working example -- your own row range, your own free-state",
"choice -- for whatever question you're actually trying to answer.",
"",
"### 7a. Detector geometry",
"Each FPA is `1024` rows x `1024` columns. A detector **row** maps to a range of along-slit positions because",
"of **keystone** (the dispersion trace's spatial footprint tilts and stretches across the row's own columns);",
"`rows_crossed(fpa, row)` tells you how many physical rows' worth of along-slit smear a given row has -- 0 at",
"FPA2's keystone-null row (row 25), growing toward the slit ends. This is why tiles near the slit edges are",
"wider than tiles near the middle: a wider tile is needed there to still cover a useful along-slit extent once",
"you average out the keystone smear.",
))

cells.append(code(
"from geocarb_gert.gd_polynomials import rows_crossed",
"",
"for row in (25, 200, 500, 890, 1000):",
"    print(f\"FPA2 row {row:4d}: rows_crossed = {float(rows_crossed(2, np.array([float(row)]))[0]):.2f}\")",
))

cells.append(md(
"### 7b. How the standard tiling actually picked tile 12",
"",
"`gd_joint_block_retrieve.build_window_tiles(fpa, row_min, row_max)` (single-band) and",
"`geocarb_gert.multiband_geometry.build_window_tiles_multiband(fpas, ...)` (multi-band, used by the sweeps)",
"turn the whole slit into a list of non-overlapping `(row_lo, row_hi)` tiles, each sized off the LOCAL keystone",
"at its own center (`build_window_tiles`'s own docstring has the exact formula). You don't need to call these",
"to define your own MWE -- pick any `(row_lo, row_hi)` you like, as narrow as you want -- but it's worth seeing",
"where 378-400/374-396 (tile 12 above) actually came from:",
))

cells.append(code(
"from geocarb_gert.multiband_geometry import build_window_tiles_multiband",
"",
"tiles = build_window_tiles_multiband((0, 2), gjr.MIN_WINDOW, 1.0, 2)   # fpas, min_window, window_scale, overlap",
"print(f\"{len(tiles)} tiles cover the whole FPA0+FPA2 slit\")",
"print(\"tile 12's own row ranges:\", tiles[12].rows)",
"print(\"a few tile widths (rows):\", [t.rows[0][1] - t.rows[0][0] + 1 for t in tiles[:6]])",
))

cells.append(md(
"### 7c. Building your own minimal window\n",
"To define YOUR OWN tile from scratch, you only need to choose:\n",
"1. **Which FPA(s)** -- one for a single-band MWE, two or more sharing a `rows_by_fpa` dict for joint.\n",
"2. **A row range per FPA**, `(row_lo, row_hi)` -- start NARROW (a dozen rows or so) for your first MWE; widen",
"   once you've confirmed it runs and understand its cost. There's no lower limit that breaks anything -- a",
"   handful of rows still gives you a complete, if noisy, retrieval to poke at.",
"3. **Which state-vector rows are free** -- any subset of `als.STATE_FIELDS`/`als.SURFACE_FIELDS`'s keys",
"   (listed below). Fewer free rows = fewer unknowns = faster and easier to reason about; start with just one",
"   or two if you're debugging something specific.",
"4. **A prior** -- `prior_fields=\"structural\"` (the default; a smoothed, non-exact background -- easiest to",
"   reason about) or `\"realistic\"` (an ACOS-like prior that never equals the truth anywhere -- what the",
"   examples above use, and what the production sweeps use).",
"",
"Everything else (`g_ratio`, `anchor_density`, `anchor_workers`, ...) has a working default; change them only",
"once the basic call above already runs for you.",
))

cells.append(code(
"print(\"atmosphere/surface state rows you can free:\")",
"print(\" \", sorted(set(als.STATE_FIELDS) | set(als.SURFACE_FIELDS)))",
"print()",
"print(\"prior sets you can pass as prior_fields=...:\")",
"print(\" \", sorted(als.PRIOR_FIELD_SETS))",
))

cells.append(code(
"# A genuinely minimal example: FPA0 only, a 12-row tile, ONE free state row.",
"# This is the pattern to copy for your own first MWE -- change ROW_LO/ROW_HI, FPA, and FREE_MWE, then run.",
"ROW_LO, ROW_HI, FPA_MWE = 500, 511, 0",
"FREE_MWE = [\"co2_ppm\"]",
"",
"t0 = time.time()",
"result_mwe = gmw.solve_window_multiband(",
"    {FPA_MWE: (ROW_LO, ROW_HI)},",
"    FREE_MWE,",
"    inputs=INPUTS,",
"    prior_fields=\"structural\",",
"    solver=\"single_scatter\",",
"    anchor_workers=2,",
"    verbose=False,",
")",
"print(f\"wall time: {time.time() - t0:.0f} s, n_free={result_mwe['n_free']}, resid_rms={result_mwe['resid_rms']:.3e}\")",
"",
"fig, ax = plt.subplots(figsize=(5, 3.5))",
"plot_row(ax, result_mwe, \"co2_ppm\", title=f\"minimal example: FPA{FPA_MWE} rows {ROW_LO}-{ROW_HI}\")",
"plt.show()",
))

cells.append(md(
"## 8. Next steps",
"",
"- **Whole-slit / many-tile sweeps are batch jobs, not notebook cells.** Use `scripts/submit_multiband_sweep.sbatch`",
"  (one Slurm array task per tile) -- read its own header comment for the environment variables it takes",
"  (`FPAS`, `FREE`, `PRIOR`, `AEROSOL`, `AEROSOL_TYPE`, ...). The `.pkl` files it produces read exactly the same",
"  way Section 6 above does.",
"- **`docs/PROJECT_STATUS.md`** is this project's running log -- design decisions, validation results, and the",
"  reasoning behind non-obvious choices (why `\"realistic\"` prior, why `single_scatter` vs `xrtm`, the aerosol",
"  degeneracies, ...), each dated. Read the most recent sections first for the current state of things.",
"- **`docs/MULTIBAND_PLAN.md`** covers the multi-band joint-retrieval design specifically.",
"- **`notebooks/pkl_results_explorer.ipynb`** is an older, single-config-sweep-focused companion notebook (a",
"  different, earlier result-file schema from `gd_joint_block_whole_slit_sweep.py`) -- worth a look once",
"  you're past this one.",
"- If something in this notebook breaks for you in a way that looks like an environment/setup issue rather than",
"  a real bug, re-run Section 1's checks first and read what they actually printed before asking -- most",
"  environment issues show up there directly.",
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (analysis env)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parents[1] / "notebooks" / "onboarding_retrievals_tour.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
