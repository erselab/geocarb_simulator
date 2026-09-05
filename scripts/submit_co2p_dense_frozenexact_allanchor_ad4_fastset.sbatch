#!/bin/bash
# Experiment B, all-frozen-rows-on-anchor-grid variant (2026-09-05).
# Fast 3-window subset only. See
# scratch_work/retrieval_co2p_dense_truth_frozenexact_allanchor_ad4.py.
#
#SBATCH --job-name=gd_co2p_dense_frozenexact_allanchor_ad4
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --array=21,26,57
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_out/co2p_dense_frozenexact_allanchor_ad4_%A_%a.out

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

export SLURM_ARRAY_TASK_COUNT=58
export ANCHOR_WORKERS="${SLURM_CPUS_PER_TASK}"

"${PYTHON}" scratch_work/retrieval_co2p_dense_truth_frozenexact_allanchor_ad4.py
