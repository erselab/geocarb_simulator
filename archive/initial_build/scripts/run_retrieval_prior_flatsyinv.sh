#!/bin/bash
# One-command launcher for the flat-sy-inv reproduction of the Phase 2
# "first try" imperfect-prior sweep (docs/PROJECT_STATUS.md Sec.6) --
# deliberately reproduces the pre-Phase-D flat-scalar Sy_inv weighting
# (--flat-sy-inv) for direct comparison against the corrected
# retrieval_prior_{co2plus1pct,co2p}_v2 results, since the ORIGINAL raw
# pkls from the real old run were overwritten with no way to recover them.
# Same 30-config axes as run_retrieval_prior_v1.sh (2 setups x 5 g_ratios
# x 3 anchor_densities), written to separate _flatsyinv tags so this can
# never collide with (or be silently skipped in favor of) any
# corrected-weighting result. Submits the array job
# (submit_retrieval_prior_flatsyinv.sbatch), then chains the scoring/
# REPORT.md/plots step (submit_retrieval_prior_flatsyinv_postprocess.
# sbatch) via a SLURM dependency -- no polling, no manual follow-up.
#
# Should run noticeably faster than the original v1 run: results/
# truth_cache/ already holds this exact scene from the corrected rerun,
# so every task here gets a cache HIT and skips the render step entirely.
#
# Run:  bash scripts/run_retrieval_prior_flatsyinv.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ARRAY_JOBID=$(sbatch --parsable scripts/submit_retrieval_prior_flatsyinv.sbatch)
echo "submitted retrieval prior flatsyinv array job ${ARRAY_JOBID} (30 tasks: 2 setups x 5 "
echo "g_ratios x 3 anchor_densities -- setup A: free=co2_ppm/prior=co2_plus1pct, setup B: "
echo "free=co2_ppm+p_surface_hpa/prior=co2_plus1pct_psurf_minus1pct, all with --flat-sy-inv)"

POST_JOBID=$(sbatch --parsable --dependency=afterok:"${ARRAY_JOBID}" \
    scripts/submit_retrieval_prior_flatsyinv_postprocess.sbatch)
echo "submitted scoring/report/plots job ${POST_JOBID} (runs after ${ARRAY_JOBID} completes: "
echo "gd_joint_block_matrix.py --flat-sy-inv per tag for scoring, then "
echo "gd_retrieval_bingrid_plot.py per tag for figures -- no re-run of any config)"
echo
echo "check progress:  sacct -j ${ARRAY_JOBID},${POST_JOBID} --format=JobID,JobName,State,Elapsed"
echo "reports land at: results/config_matrix/retrieval_prior_co2plus1pct_flatsyinv/REPORT.md"
echo "                 results/config_matrix/retrieval_prior_co2p_flatsyinv/REPORT.md"
echo "plots land at:   plots/config_matrix/retrieval_prior_co2plus1pct_flatsyinv/"
echo "                 plots/config_matrix/retrieval_prior_co2p_flatsyinv/"
