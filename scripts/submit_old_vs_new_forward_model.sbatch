#!/bin/bash
# Old-vs-new forward-model regression comparison (2026-09-04) -- the last
# open item on the footprint-integrated-forward-model merge checklist.
# See scratch_work/old_vs_new_forward_model.py.
#
#SBATCH --job-name=gd_old_vs_new_forward_model
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_out/old_vs_new_forward_model_%j.out

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

"${PYTHON}" scratch_work/old_vs_new_forward_model.py
