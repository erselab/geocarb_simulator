#!/bin/bash
# Experiment B, free-albedo variant (2026-09-05, user: "let's try
# unfreezing albedo" / "just the smaller test set"). Fast 3-window
# subset (task_ids 21,26,57 -- rows 189-197, 238-248, 1013-1023),
# identified this session as small/fast/high-signal for this class of
# frozen-row/representability experiment. See
# scratch_work/retrieval_co2palbedo_dense_truth_frozenexact_ad4.py.
#
#SBATCH --job-name=gd_co2palbedo_dense_frozenexact_fastset
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --array=21,26,57
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_out/co2palbedo_dense_frozenexact_fastset_%A_%a.out

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

"${PYTHON}" scratch_work/retrieval_co2palbedo_dense_truth_frozenexact_ad4.py
