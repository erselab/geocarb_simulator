#!/bin/bash
# Scoring + REPORT.md + plots for the "v3" imperfect-prior sweep
# (docs/PROJECT_STATUS.md Sec.6): co2plus1pct/co2p redone under an
# explicit exponential prior_form + overlap=2, plus two new
# prior_fields=structural setups. Meant to run as a SLURM dependency AFTER
# submit_retrieval_prior_v3.sbatch's array job completes (see
# scripts/run_retrieval_prior_v3.sh, which submits both with the
# dependency wired up automatically) -- not meant to be submitted
# standalone unless the array job already finished on its own.
#
# --overlap 2 is passed to EVERY gd_joint_block_matrix.py call below even
# though it isn't regenerating anything (the array job already wrote every
# .pkl) -- resolve_existing() checks each cached file's own overlap/
# prior_form metadata (docs/PROJECT_STATUS.md Sec.6, same pattern as the
# --flat-sy-inv check), so omitting it here would make this step decide
# the array job's overlap=2 pkls don't match an overlap=0 request and try
# to regenerate them from scratch under the WRONG setting.
#
# Four SEPARATE tags (one per setup, own --free/--prior-fields each):
#   retrieval_prior_co2plus1pct_v3:     free=co2_ppm,      prior=co2_plus1pct
#   retrieval_prior_co2p_v3:            free=co2_ppm+p_surface_hpa,
#                                        prior=co2_plus1pct_psurf_minus1pct
#   retrieval_prior_structural_co2_v3:  free=co2_ppm,      prior=structural
#   retrieval_prior_structural_co2p_v3: free=co2_ppm+p_surface_hpa,
#                                        prior=structural
#
# `--dependency=afterok:<array_jobid>` on a job ARRAY only fires once
# EVERY array task has exited 0 -- if any task failed, this job stays
# pending/cancelled rather than silently scoring a partial sweep. Check
# `sacct -j <array_jobid>` and rerun any failed task IDs directly before
# resubmitting this step by hand if that happens.
#
#SBATCH --job-name=gd_retrieval_prior_v3_post
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_out/retrieval_prior_v3_postprocess_%j.out

set -euo pipefail
module load git/2.50.1

cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python

"${PYTHON}" scripts/gd_joint_block_matrix.py \
    --tag retrieval_prior_co2plus1pct_v3 --free co2 \
    --g-ratio 0.5,1,3,6,12 --anchor-density 1,4,16 \
    --prior-fields co2_plus1pct --overlap 2 \
    --hires-only --jacobian analytic

"${PYTHON}" scripts/gd_joint_block_matrix.py \
    --tag retrieval_prior_co2p_v3 --free co2p \
    --g-ratio 0.5,1,3,6,12 --anchor-density 1,4,16 \
    --prior-fields co2_plus1pct_psurf_minus1pct --overlap 2 \
    --hires-only --jacobian analytic

"${PYTHON}" scripts/gd_joint_block_matrix.py \
    --tag retrieval_prior_structural_co2_v3 --free co2 \
    --g-ratio 0.5,1,3,6,12 --anchor-density 1,4,16 \
    --prior-fields structural --overlap 2 \
    --hires-only --jacobian analytic

"${PYTHON}" scripts/gd_joint_block_matrix.py \
    --tag retrieval_prior_structural_co2p_v3 --free co2p \
    --g-ratio 0.5,1,3,6,12 --anchor-density 1,4,16 \
    --prior-fields structural --overlap 2 \
    --hires-only --jacobian analytic

"${PYTHON}" scripts/gd_retrieval_bingrid_plot.py --tag retrieval_prior_co2plus1pct_v3
"${PYTHON}" scripts/gd_retrieval_bingrid_plot.py --tag retrieval_prior_co2p_v3
"${PYTHON}" scripts/gd_retrieval_bingrid_plot.py --tag retrieval_prior_structural_co2_v3
"${PYTHON}" scripts/gd_retrieval_bingrid_plot.py --tag retrieval_prior_structural_co2p_v3
