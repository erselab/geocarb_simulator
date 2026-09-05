#!/bin/bash
# One-command launcher for the Phase 2 "first try" imperfect-prior sweep
# (PROJECT_STATUS.md Sec.5: two uniform multiplicative-bias priors --
# CO2+1%, and CO2+1%/p_surface-1% -- each retrieved against exactly the
# free row(s) the prior mis-states, across g_ratio in {0.5,1,3,6,12} x
# anchor_density in {1,4,16}, matching Phase 1's own full axis). Submits
# the 30-task array
# (submit_retrieval_prior_v1.sbatch), then submits the scoring/REPORT.md/
# plots step (submit_retrieval_prior_v1_postprocess.sbatch, which
# re-invokes gd_joint_block_matrix.py once per tag for scoring -- no
# separate merge needed, each array task already writes one complete .pkl
# -- then gd_retrieval_bingrid_plot.py per tag for figures) with a SLURM
# dependency so it fires automatically once every array task completes --
# no polling, no manual follow-up.
#
# Run:  bash scripts/run_retrieval_prior_v1.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ARRAY_JOBID=$(sbatch --parsable scripts/submit_retrieval_prior_v1.sbatch)
echo "submitted retrieval prior v1 array job ${ARRAY_JOBID} (30 tasks: 2 setups x 5 g_ratios x"
echo "3 anchor_densities -- setup A: free=co2_ppm/prior=co2_plus1pct, setup B: "
echo "free=co2_ppm+p_surface_hpa/prior=co2_plus1pct_psurf_minus1pct)"

POST_JOBID=$(sbatch --parsable --dependency=afterok:"${ARRAY_JOBID}" \
    scripts/submit_retrieval_prior_v1_postprocess.sbatch)
echo "submitted scoring/report/plots job ${POST_JOBID} (runs after ${ARRAY_JOBID} completes: "
echo "gd_joint_block_matrix.py per tag for scoring, then gd_retrieval_bingrid_plot.py per tag "
echo "for figures -- no re-run of any config)"
echo
echo "check progress:  sacct -j ${ARRAY_JOBID},${POST_JOBID} --format=JobID,JobName,State,Elapsed"
echo "reports land at: results/config_matrix/retrieval_prior_co2plus1pct_v1/REPORT.md"
echo "                 results/config_matrix/retrieval_prior_co2p_v1/REPORT.md"
echo "plots land at:   plots/config_matrix/retrieval_prior_co2plus1pct_v1/"
echo "                 plots/config_matrix/retrieval_prior_co2p_v1/"
