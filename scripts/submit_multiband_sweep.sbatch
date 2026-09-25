#!/bin/bash
#SBATCH --partition=atmos
#SBATCH --account=ltmurray_lab
#SBATCH --job-name=mb_sweep
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=5:00:00
#SBATCH --output=slurm_out/mb_sweep_%x_%A_%a.out
# Multi-band (FPA0+FPA2, no aerosol) whole-slit sweep: ONE array task = ONE tile of
# build_window_tiles_multiband (33 tiles). Override --cpus-per-task/--mem/--time/--array per
# width tier on the command line (docs/MULTIBAND_PLAN.md phase 5/6). A failed window exits
# non-zero (Slurm FAILED) -- unlike the single-band driver it does not record an error and
# exit 0. Pass the free set through the environment (--export splits at commas):
#   [FPAS=0,2 default; e.g. FPAS=0,3] FREE=co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo [PRIOR=realistic] sbatch --export=ALL --array=... scripts/submit_multiband_sweep.sbatch
# SOLVER: if unset, gd_multiband_window.py itself now defaults to xrtm when AEROSOL=1 else
# single_scatter (2026-09-23, user: "when aerosol is not included, the RT model should default to
# single_scatter" -- single_scatter's own I_scatter term is unconditionally zero without aerosol
# kwargs, so it's already exactly absorption/extinction-only there; xrtm's multiple-scattering cost
# only buys real accuracy once aerosol is actually free). Set SOLVER explicitly to override either way.
# AEROSOL=1 requires AEROSOL_TYPE=<smoke|dust|sulfate|sea_salt|cloud_water|registry_smoke...> (gd_multiband_window.py --aerosol-type;
# no default). realistic per-band properties for the named types; registry_<type> reproduces the earlier two-slot runs.
# PRIOR = als.PRIOR_FIELD_SETS name (default structural); realistic = ACOS-like prior that never equals the truth.
# GEOM=<geometry config json> selects tile geometry+bin centres from a config (the tile index is then a config index; use for single-band runs so bins match a multi-band run).
# MAXTASKS overrides GEOCARB_ANCHOR_POOL_MAXTASKS (2026-09-22: xrtm's own C save_tree cache leaks real,
# reachable memory on every RT call, never freed until the model is destroyed -- confirmed via malloc_trim
# NOT reclaiming it; jacobians.LinearizePool respawns an anchor-pool worker after this many anchor tasks
# so the OS reclaims what xrtm itself never does. Default here (15) sized for THIS sbatch's own 180G/16
# workers/2 bands (32 total XRTM-calling workers/tile): worst-case ~150MB/task (aerosol + the p_surface
# fallback's 2 extra RT calls) x 15 tasks =~2.2GB/worker peak between respawns, well under 180G/32 =~5.6GB.
# Override for a different --mem/--cpus-per-task/band-count combination.
set -euo pipefail
cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GEOCARB_ANCHOR_POOL_MAXTASKS="${MAXTASKS:-15}"
: "${FREE:?set FREE=... in the environment}"
SHM_NAME="/dev/shm/geocarb_L_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
trap 'rm -f "${SHM_NAME}"*' EXIT
/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python -u scripts/gd_multiband_window.py \
    --fpas "${FPAS:-0,2}" --tile "${SLURM_ARRAY_TASK_ID}" --anchor-mode cover --g-ratio 1 \
    --free "${FREE}" ${GEOM:+--geometry-config "${GEOM}"} --prior-fields "${PRIOR:-structural}" --anchor-workers "${SLURM_CPUS_PER_TASK}" \
    ${SOLVER:+--solver "${SOLVER}"} ${AEROSOL:+--aerosol} ${AEROSOL_TYPE:+--aerosol-type "${AEROSOL_TYPE}"} ${NOISE_SEED:+--noise-seed "${NOISE_SEED}"}
