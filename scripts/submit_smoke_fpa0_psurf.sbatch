#!/bin/bash
# Verification smoke test (2026-09-04): single-window retrieval on FPA0
# (O2_A band -- molecules o2,h2o, no CO2 sensitivity at all) with the
# physically correct target gas, p_surface_hpa, instead of the earlier
# smoke test's default --free co2_ppm (which trivially gave dof=0 there,
# not a bug -- FPA0 just can't see CO2). Closes the multi-FPA retrieval-
# side verification gap from docs/PROJECT_STATUS.md Sec.13/14.
#
#SBATCH --job-name=gd_smoke_fpa0_psurf
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_out/smoke_fpa0_psurf_%j.out

set -euo pipefail
module load git/2.50.1

cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${PYTHON}" scripts/gd_joint_block_retrieve.py \
    --fpa 0 --free p_surface_hpa --task-id 1 --n-tasks 2 \
    --hires-only --jacobian analytic --anchor-workers "${SLURM_CPUS_PER_TASK}" \
    --out scratch_work/smoke_mid_window_fpa0_psurf.pkl
