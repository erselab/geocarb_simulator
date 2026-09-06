#!/bin/bash
# Finer g_ratio exploration (2026-09-05/06, user), fast 3-window subset
# only (task_ids 21,26,57 -- rows 189-197, 238-248, 1013-1023), capped
# after the widest window (task_id 56, width=45) took >100min at
# g_ratio=0.25 with no sign of finishing and the user chose to cap
# scope to the fast subset rather than pay for a full 58-window array.
# GRATIO env var selects 0.5 or 0.25 -- submit once per value. See
# scratch_work/retrieval_co2palbedo_dense_truth_anchorfrozen_gratio.py.
#
#SBATCH --job-name=gd_co2palbedo_anchorfrozen_gratio
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --array=21,26,57
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_out/co2palbedo_anchorfrozen_gratio_%A_%a.out

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
# GRATIO must be set in the environment before submitting (0.5 or 0.25).

"${PYTHON}" scratch_work/retrieval_co2palbedo_dense_truth_anchorfrozen_gratio.py
