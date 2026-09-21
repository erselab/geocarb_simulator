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
#   FREE=co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo [PRIOR=realistic] sbatch --export=ALL --array=... scripts/submit_multiband_sweep.sbatch
# SOLVER=xrtm and AEROSOL=1 select the aerosol arm (required together with free amplitude/height_aerosol).
# PRIOR = als.PRIOR_FIELD_SETS name (default structural); realistic = ACOS-like prior that never equals the truth.
set -euo pipefail
cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
: "${FREE:?set FREE=... in the environment}"
SHM_NAME="/dev/shm/geocarb_L_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
trap 'rm -f "${SHM_NAME}"*' EXIT
/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python -u scripts/gd_multiband_window.py \
    --fpas 0,2 --tile "${SLURM_ARRAY_TASK_ID}" --anchor-mode cover --g-ratio 1 \
    --free "${FREE}" --prior-fields "${PRIOR:-structural}" --anchor-workers "${SLURM_CPUS_PER_TASK}" \
    --solver "${SOLVER:-single_scatter}" ${AEROSOL:+--aerosol}
