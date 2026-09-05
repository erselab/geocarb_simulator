#!/bin/bash
# Scoring + REPORT.md + plots for the first real albedo-in-state-vector
# sweep (docs/PROJECT_STATUS.md Sec.6). Meant to run as a SLURM dependency
# AFTER submit_retrieval_prior_albedo_v1.sbatch's array job completes (see
# scripts/run_retrieval_prior_albedo_v1.sh, which submits both with the
# dependency wired up automatically) -- not meant to be submitted
# standalone unless the array job already finished on its own.
#
# --vary-albedo --overlap 2 are passed to EVERY gd_joint_block_matrix.py
# call below even though it isn't regenerating anything (the array job
# already wrote every .pkl) -- resolve_existing() checks each cached
# file's own vary_albedo/overlap/prior_form metadata (same pattern as the
# --flat-sy-inv check), so omitting either here would make this step
# decide the array job's own pkls don't match a default (False/0) request
# and try to regenerate them from scratch under the WRONG settings.
#
# Two SEPARATE tags (one per setup, own --prior-fields each):
#   retrieval_prior_albedo_exact_v1:      prior=exact (truth)
#   retrieval_prior_albedo_structural_v1: prior=structural (patches known,
#                                         fine texture unknown)
# Both use the SAME free set (co2p_albedo = co2_ppm,p_surface_hpa,albedo).
#
# `--dependency=afterok:<array_jobid>` on a job ARRAY only fires once
# EVERY array task has exited 0 -- if any task failed, this job stays
# pending/cancelled rather than silently scoring a partial sweep. Check
# `sacct -j <array_jobid>` and rerun any failed task IDs directly before
# resubmitting this step by hand if that happens.
#
# 2026-08-25 postmortem: the FIRST run of this step TIMED OUT (20 min) and
# OOM'd, because the array job's own sbatch originally wrote CIDs prefixed
# "co2p-" while --free co2p_albedo here builds CIDs prefixed "co2p_albedo-"
# (config_id() uses the --free KEY string itself as the prefix) -- a real
# mismatch, not a false alarm, so resolve_existing() correctly decided the
# array job's pkls weren't there and tried to REGENERATE all 30 configs
# from scratch inside this job's tiny 8GB/1-cpu allocation. Fixed by (1)
# renaming every already-computed pkl to the co2p_albedo- prefix
# (submit_retrieval_prior_albedo_v1.sbatch's own CID now matches too, for
# any future rerun) and (2) bumping --mem/--time here as a safety margin
# in case scoring+plotting 30 three-free-row configs genuinely needs more
# than the co2p-only sweeps did.
#
#SBATCH --job-name=gd_retrieval_prior_albedo_v1_post
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=slurm_out/retrieval_prior_albedo_v1_postprocess_%j.out

set -euo pipefail
module load git/2.50.1

cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python

"${PYTHON}" scripts/gd_joint_block_matrix.py \
    --tag retrieval_prior_albedo_exact_v1 --free co2p_albedo \
    --g-ratio 0.5,1,3,6,12 --anchor-density 1,4,16 \
    --prior-fields exact --overlap 2 --vary-albedo \
    --hires-only --jacobian analytic

"${PYTHON}" scripts/gd_joint_block_matrix.py \
    --tag retrieval_prior_albedo_structural_v1 --free co2p_albedo \
    --g-ratio 0.5,1,3,6,12 --anchor-density 1,4,16 \
    --prior-fields structural --overlap 2 --vary-albedo \
    --hires-only --jacobian analytic

"${PYTHON}" scripts/gd_retrieval_bingrid_plot.py --tag retrieval_prior_albedo_exact_v1
"${PYTHON}" scripts/gd_retrieval_bingrid_plot.py --tag retrieval_prior_albedo_structural_v1
