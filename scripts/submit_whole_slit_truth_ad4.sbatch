#!/bin/bash
# Build Mode-2 representative truth at anchor_density=4 (2026-09-04, user
# request: speed up the imperfect-prior ablation tests). See
# scratch_work/build_whole_slit_truth_ad4.py.
#
#SBATCH --job-name=gd_whole_slit_truth_ad4
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_out/whole_slit_truth_ad4_%j.out

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

"${PYTHON}" scratch_work/build_whole_slit_truth_ad4.py
