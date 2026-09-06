#!/bin/bash
# Keystone-vs-truth-structure control experiment (2026-09-06): the same
# Sec.6 g_ratio=1 config with every truth field spatially reversed.
# Fast 3-window subset only. See
# scratch_work/retrieval_co2palbedo_reversed_truth_anchorfrozen_gratio1.py.
#
#SBATCH --job-name=gd_co2palbedo_reversed_gratio1
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --array=21,26,57
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_out/co2palbedo_reversed_gratio1_%A_%a.out

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

"${PYTHON}" scratch_work/retrieval_co2palbedo_reversed_truth_anchorfrozen_gratio1.py
