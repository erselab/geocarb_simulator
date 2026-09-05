#!/bin/bash
# Verification smoke test (2026-09-04): single-window retrieval on FPA3
# (CH4_CO band -- molecules ch4,co,h2o,n2o, no CO2 sensitivity) with the
# physically correct target gases, ch4_ppb+co_ppb, instead of the earlier
# smoke test's default --free co2_ppm (trivially gave dof=0 there, not a
# bug -- FPA3 just can't see CO2). Closes the multi-FPA retrieval-side
# verification gap from docs/PROJECT_STATUS.md Sec.13/14.
#
#SBATCH --job-name=gd_smoke_fpa3_ch4co
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_out/smoke_fpa3_ch4co_%j.out

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

"${PYTHON}" scripts/gd_joint_block_whole_slit_sweep.py \
    --fpa 3 --free ch4_ppb,co_ppb --task-id 1 --n-tasks 2 \
    --hires-only --jacobian analytic --anchor-workers "${SLURM_CPUS_PER_TASK}" \
    --out scratch_work/smoke_mid_window_fpa3_ch4co.pkl
