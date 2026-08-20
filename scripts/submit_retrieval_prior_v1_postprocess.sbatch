#!/bin/bash
# Scoring + REPORT.md + plots for the Phase 2 "first try" imperfect-prior
# sweep (PROJECT_STATUS.md Sec.5, prior_fields=co2_plus1pct /
# co2_plus1pct_psurf_minus1pct). Meant to run as a SLURM dependency AFTER
# submit_retrieval_prior_v1.sbatch's array job completes (see
# scripts/run_retrieval_prior_v1.sh, which submits both with the
# dependency wired up automatically) -- not meant to be submitted
# standalone unless the array job already finished on its own.
#
# Two SEPARATE tags here (unlike retrieval_bingrid_v1's single tag with
# both free-sets) since each setup pairs a different --free with a
# different --prior-fields (see submit_retrieval_prior_v1.sbatch's header
# for why they aren't cross-multiplied into one matrix run):
#   retrieval_prior_co2plus1pct_v1: free=co2_ppm,      prior=co2_plus1pct
#   retrieval_prior_co2p_v1:        free=co2_ppm+p_surface_hpa,
#                                    prior=co2_plus1pct_psurf_minus1pct
#
# Each gd_joint_block_matrix.py call below runs WITHOUT --force, same
# axes/tag the array job itself used, so resolve_existing() finds every
# .pkl already on disk and skips straight to scoring/REPORT.md/summary.pkl
# -- no re-run, no duplicated scoring logic (same pattern as
# submit_retrieval_bingrid_v1_postprocess.sbatch).
#
# gd_retrieval_bingrid_plot.py is generic over --tag (reads whatever
# free-sets/params are actually present in that tag's summary.pkl, skips
# the rest gracefully -- verified by reading the script directly,
# 2026-08-20) so it's reused as-is for each tag rather than writing a
# prior_v1-specific plotting script.
#
# `--dependency=afterok:<array_jobid>` on a job ARRAY only fires once
# EVERY array task has exited 0 -- if any task failed, this job stays
# pending/cancelled rather than silently scoring a partial sweep. Check
# `sacct -j <array_jobid>` and rerun any failed task IDs directly before
# resubmitting this step by hand if that happens.
#
#SBATCH --job-name=gd_retrieval_prior_v1_post
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_out/retrieval_prior_v1_postprocess_%j.out

set -euo pipefail
module load git/2.50.1

cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python

"${PYTHON}" scripts/gd_joint_block_matrix.py \
    --tag retrieval_prior_co2plus1pct_v1 --free co2 \
    --g-ratio 0.5,1,3,6,12 --anchor-density 1,4,16 \
    --prior-fields co2_plus1pct \
    --hires-only --jacobian analytic

"${PYTHON}" scripts/gd_joint_block_matrix.py \
    --tag retrieval_prior_co2p_v1 --free co2p \
    --g-ratio 0.5,1,3,6,12 --anchor-density 1,4,16 \
    --prior-fields co2_plus1pct_psurf_minus1pct \
    --hires-only --jacobian analytic

"${PYTHON}" scripts/gd_retrieval_bingrid_plot.py --tag retrieval_prior_co2plus1pct_v1
"${PYTHON}" scripts/gd_retrieval_bingrid_plot.py --tag retrieval_prior_co2p_v1
