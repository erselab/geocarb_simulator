#!/bin/bash
# Scoring + REPORT.md + plots for the Phase 1 bingrid retrieval sweep
# (PROJECT_STATUS.md Sec.5). Meant to run as a SLURM dependency AFTER
# submit_retrieval_bingrid_v1.sbatch's array job completes (see
# scripts/run_retrieval_bingrid_v1.sh, which submits both with the
# dependency wired up automatically) -- not meant to be submitted
# standalone unless the array job already finished on its own.
#
# No merge step needed here (unlike submit_whole_slit_1x_postprocess.
# sbatch) -- each array task already writes ONE complete, ready-to-score
# .pkl (gd_joint_block_whole_slit_sweep.py's own internal multiprocessing
# Pool handles all 58 windows within that one task; --task-id/--n-tasks
# were never passed).
#
# Two steps:
#   1. gd_joint_block_matrix.py, SAME axes/tag, WITHOUT --force -- its own
#      resolve_existing() finds every .pkl the array job already wrote and
#      skips straight to scoring/REPORT.md/summary.pkl, so the exact same
#      scoring code gets reused instead of duplicated here, and nothing
#      gets recomputed.
#   2. gd_retrieval_bingrid_plot.py -- reads summary.pkl (posterior-error-
#      vs-slit-position and RMS-across-the-sweep figures) and every
#      per-config .pkl directly (full-detector + hotspot/mountain zoom
#      residual figures, 30 configs x 4 figures -- the slow part; this is
#      why --cpus-per-task/--time/--mem are larger than the scoring-only
#      version this step used to be).
#
# `--dependency=afterok:<array_jobid>` on a job ARRAY only fires once
# EVERY array task has exited 0 -- if any task failed, this job stays
# pending (or is cancelled, depending on cluster policy) rather than
# silently scoring/plotting a partial/broken sweep. Check
# `sacct -j <array_jobid>` and rerun any failed task IDs directly (same
# flags, computed from that task's SLURM_ARRAY_TASK_ID per
# submit_retrieval_bingrid_v1.sbatch's own index math) before resubmitting
# this step by hand if that happens.
#
#SBATCH --job-name=gd_retrieval_bingrid_v1_post
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_out/retrieval_bingrid_v1_postprocess_%j.out

set -euo pipefail
module load git/2.50.1

cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python

"${PYTHON}" scripts/gd_joint_block_matrix.py \
    --tag retrieval_bingrid_v1 --free co2,co2p \
    --g-ratio 0.5,1,3,6,12 --anchor-density 1,4,16 \
    --hires-only --jacobian analytic

"${PYTHON}" scripts/gd_retrieval_bingrid_plot.py --tag retrieval_bingrid_v1
