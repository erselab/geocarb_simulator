#!/bin/bash
# Whole-slit joint-block sweep, realistic scene, G tied to the along-slit
# ground-pixel footprint (G = width, i.e. --g-ratio 1.0 -- see
# docs/JOINT_BLOCK_MIGRATION_PLAN.md Sec.9). Anchors unchanged from the
# original production method (row-uniform, G_eff = width + 2*PAD).
#
# One SLURM array task per whole-slit window -- the simplest possible
# split (array index maps directly to a window, no load-balancing logic,
# no shared state during the run). Each task independently rebuilds the
# realistic-scene band from scratch (redundant across tasks, but that
# redundant cost is paid in PARALLEL across nodes, not serially -- a
# deliberate simplicity-over-CPU-hours tradeoff; see the chat discussion
# for the alternative if that redundancy ever needs trimming) and writes
# its own small output file -- no cross-task I/O happens until the
# separate merge step below.
#
# CHECK BEFORE SUBMITTING: --account is confirmed (ltmurray_lab); --partition
# is a guess (atmos/atmos2 look like the right domain-matched partitions
# on this cluster, `sinfo` for the full list) -- adjust to whatever
# allocation you actually want to use.
#
#SBATCH --job-name=gd_whole_slit_1x
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --array=0-57
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_out/whole_slit_1x_%A_%a.out

set -euo pipefail
module load git/2.50.1

cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python

# Must match --array above (N_TASKS-1): 58 windows from build_window_tiles(FPA=2)
# as of this writing -- confirmed via a quick geometry-only check, not assumed.
N_TASKS=58

"${PYTHON}" scripts/gd_joint_block_whole_slit_sweep.py \
    --g-ratio 1.0 \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --n-tasks "${N_TASKS}" \
    --n-workers "${SLURM_CPUS_PER_TASK}"

# Don't submit this file directly for the full pipeline -- use
# `bash scripts/run_whole_slit_1x.sh` instead, which submits this array
# job AND chains submit_whole_slit_1x_postprocess.sbatch (merge + all
# plots) after it via --dependency=afterok, so nothing needs to be run
# by hand once it's launched.
