#!/bin/bash
# Merge + plot step for the whole-slit factory-default (G=width/3) rerun.
# Meant to run as a SLURM dependency AFTER submit_whole_slit_factory.sbatch's
# array job completes (see scripts/run_whole_slit_factory.sh, which submits
# both with the dependency wired up automatically).
#
# `--dependency=afterok:<array_jobid>` on a job ARRAY only fires once EVERY
# array task has exited 0 -- if any task failed, this job stays pending
# (or is cancelled, depending on cluster policy) rather than silently
# merging a partial/broken run. Check `sacct -j <array_jobid>` and rerun
# any failed task IDs directly (same --task-id/--n-tasks flags) before
# resubmitting this step by hand if that happens.
#
#SBATCH --job-name=gd_whole_slit_factory_post
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=slurm_out/whole_slit_factory_postprocess_%j.out

set -euo pipefail
module load git/2.50.1

cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python

"${PYTHON}" scripts/gd_joint_block_whole_slit_merge.py \
    results/gd_joint_block_whole_slit_fpa2_gratio3_parts

"${PYTHON}" scripts/gd_joint_block_whole_slit_plot.py \
    results/gd_joint_block_whole_slit_fpa2_gratio3.pkl
