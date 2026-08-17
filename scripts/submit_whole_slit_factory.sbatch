#!/bin/bash
# Whole-slit joint-block sweep, realistic scene, G at the "factory"
# production default (G = width/3, --g-ratio 3.0 -- the same convention
# results/gd_joint_block_whole_slit_fpa2.pkl was originally run with,
# before this script saved full residual arrays/state vectors). Rerun
# here only to pick up that saved data, not because the retrieval itself
# changed -- bias numbers should match the original run to within GN
# tolerance. Anchors unchanged (row-uniform, G_eff = width + 2*PAD), same
# as every other run in this investigation.
#
# One SLURM array task per whole-slit window, same simplicity tradeoff as
# submit_whole_slit_1x.sbatch (each task independently rebuilds the band;
# redundant compute paid in parallel, not serially). Since G=width/3 is
# smaller than G=width, per-task cost here should be LOWER than the 1x
# run's measured worst case (106s) -- the --time budget below is kept the
# same anyway for safety margin, not because it's needed.
#
# CHECK BEFORE SUBMITTING: --account is confirmed (ltmurray_lab); --partition
# is a guess (atmos/atmos2 look like the right domain-matched partitions
# on this cluster, `sinfo` for the full list) -- adjust to whatever
# allocation you actually want to use.
#
#SBATCH --job-name=gd_whole_slit_factory
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --array=0-57
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_out/whole_slit_factory_%A_%a.out

set -euo pipefail
module load git/2.50.1

cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python

# Must match --array above (N_TASKS-1): 58 windows from build_window_tiles(FPA=2)
# as of this writing -- confirmed via a quick geometry-only check, not assumed.
N_TASKS=58

"${PYTHON}" scripts/gd_joint_block_whole_slit_sweep.py \
    --g-ratio 3.0 \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --n-tasks "${N_TASKS}" \
    --n-workers "${SLURM_CPUS_PER_TASK}"

# Don't submit this file directly for the full pipeline -- use
# `bash scripts/run_whole_slit_factory.sh` instead, which submits this
# array job AND chains submit_whole_slit_factory_postprocess.sbatch
# (merge + plot) after it via --dependency=afterok.
