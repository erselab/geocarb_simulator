#!/bin/bash
# One-command launcher for the first real albedo-in-state-vector sweep
# (docs/PROJECT_STATUS.md Sec.6): --free co2_ppm,p_surface_hpa,albedo
# against the real spatially-varying surface truth scene (--vary-albedo),
# under an exact prior and a structural (patches known, fine texture
# unknown) prior, across g_ratio in {0.5,1,3,6,12} x anchor_density in
# {1,4,16}. Submits the 30-task array
# (submit_retrieval_prior_albedo_v1.sbatch), then submits the scoring/
# REPORT.md/plots step (submit_retrieval_prior_albedo_v1_postprocess.sbatch,
# re-invoking gd_joint_block_matrix.py once per tag -- no separate merge
# needed, each array task already writes one complete .pkl -- then
# gd_retrieval_bingrid_plot.py per tag for figures) with a SLURM
# dependency so it fires automatically once every array task completes --
# no polling, no manual follow-up.
#
# Run:  bash scripts/run_retrieval_prior_albedo_v1.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ARRAY_JOBID=$(sbatch --parsable scripts/submit_retrieval_prior_albedo_v1.sbatch)
echo "submitted retrieval prior albedo v1 array job ${ARRAY_JOBID} (30 tasks: 2 setups x 5"
echo "g_ratios x 3 anchor_densities -- setup 0: prior=exact, setup 1: prior=structural --"
echo "both free=co2_ppm,p_surface_hpa,albedo, --vary-albedo, --overlap 2)"

POST_JOBID=$(sbatch --parsable --dependency=afterok:"${ARRAY_JOBID}" \
    scripts/submit_retrieval_prior_albedo_v1_postprocess.sbatch)
echo "submitted scoring/report/plots job ${POST_JOBID} (runs after ${ARRAY_JOBID} completes: "
echo "gd_joint_block_matrix.py per tag for scoring, then gd_retrieval_bingrid_plot.py per tag "
echo "for figures -- no re-run of any config)"
echo
echo "check progress:  sacct -j ${ARRAY_JOBID},${POST_JOBID} --format=JobID,JobName,State,Elapsed"
echo "reports land at: results/config_matrix/retrieval_prior_albedo_exact_v1/REPORT.md"
echo "                 results/config_matrix/retrieval_prior_albedo_structural_v1/REPORT.md"
echo "plots land at:   plots/config_matrix/retrieval_prior_albedo_exact_v1/"
echo "                 plots/config_matrix/retrieval_prior_albedo_structural_v1/"
