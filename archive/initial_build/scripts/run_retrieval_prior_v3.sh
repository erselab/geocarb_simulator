#!/bin/bash
# One-command launcher for the "v3" imperfect-prior sweep
# (docs/PROJECT_STATUS.md Sec.6): co2plus1pct and co2p REDONE under an
# explicitly-consistent exponential prior_form + window-overlap merging
# (--overlap 2), plus two new prior_fields=structural setups (prior loses
# ALL localized plume/hot-spot/topography detail, a genuinely different
# failure mode from co2plus1pct/co2p's uniform multiplicative bias) --
# across g_ratio in {0.5,1,3,6,12} x anchor_density in {1,4,16}, matching
# every prior sweep's own full axis. Submits the 60-task array
# (submit_retrieval_prior_v3.sbatch), then submits the scoring/REPORT.md/
# plots step (submit_retrieval_prior_v3_postprocess.sbatch, re-invoking
# gd_joint_block_matrix.py once per tag -- no separate merge needed, each
# array task already writes one complete .pkl -- then gd_retrieval_
# bingrid_plot.py per tag for figures) with a SLURM dependency so it fires
# automatically once every array task completes -- no polling, no manual
# follow-up.
#
# Run:  bash scripts/run_retrieval_prior_v3.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ARRAY_JOBID=$(sbatch --parsable scripts/submit_retrieval_prior_v3.sbatch)
echo "submitted retrieval prior v3 array job ${ARRAY_JOBID} (60 tasks: 4 setups x 5 g_ratios x"
echo "3 anchor_densities -- setup 0: free=co2_ppm/prior=co2_plus1pct, setup 1: "
echo "free=co2_ppm+p_surface_hpa/prior=co2_plus1pct_psurf_minus1pct, setup 2: "
echo "free=co2_ppm/prior=structural, setup 3: free=co2_ppm+p_surface_hpa/prior=structural --"
echo "all four with prior_form=exponential, overlap=2)"

POST_JOBID=$(sbatch --parsable --dependency=afterok:"${ARRAY_JOBID}" \
    scripts/submit_retrieval_prior_v3_postprocess.sbatch)
echo "submitted scoring/report/plots job ${POST_JOBID} (runs after ${ARRAY_JOBID} completes: "
echo "gd_joint_block_matrix.py per tag for scoring, then gd_retrieval_bingrid_plot.py per tag "
echo "for figures -- no re-run of any config)"
echo
echo "check progress:  sacct -j ${ARRAY_JOBID},${POST_JOBID} --format=JobID,JobName,State,Elapsed"
echo "reports land at: results/config_matrix/retrieval_prior_co2plus1pct_v3/REPORT.md"
echo "                 results/config_matrix/retrieval_prior_co2p_v3/REPORT.md"
echo "                 results/config_matrix/retrieval_prior_structural_co2_v3/REPORT.md"
echo "                 results/config_matrix/retrieval_prior_structural_co2p_v3/REPORT.md"
echo "plots land at:   plots/config_matrix/retrieval_prior_co2plus1pct_v3/"
echo "                 plots/config_matrix/retrieval_prior_co2p_v3/"
echo "                 plots/config_matrix/retrieval_prior_structural_co2_v3/"
echo "                 plots/config_matrix/retrieval_prior_structural_co2p_v3/"
