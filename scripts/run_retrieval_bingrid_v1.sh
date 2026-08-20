#!/bin/bash
# One-command launcher for the Phase 1 bingrid retrieval sweep
# (PROJECT_STATUS.md Sec.5: prior=truth at the bin scale, state_interp=
# "linear", CO2-only and CO2+p_surface, g_ratio x anchor_density). Submits
# the 30-task array (submit_retrieval_bingrid_v1.sbatch), then submits the
# scoring/REPORT.md/plots step (submit_retrieval_bingrid_v1_postprocess.
# sbatch, which re-invokes gd_joint_block_matrix.py for scoring -- no
# separate merge needed, each array task already writes one complete .pkl
# -- then gd_retrieval_bingrid_plot.py for figures) with a SLURM
# dependency so it fires automatically once every array task completes --
# no polling, no manual follow-up.
#
# Run:  bash scripts/run_retrieval_bingrid_v1.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ARRAY_JOBID=$(sbatch --parsable scripts/submit_retrieval_bingrid_v1.sbatch)
echo "submitted retrieval bingrid v1 array job ${ARRAY_JOBID} (30 tasks: 2 free-sets x "
echo "5 g_ratios x 3 anchor_densities)"

POST_JOBID=$(sbatch --parsable --dependency=afterok:"${ARRAY_JOBID}" \
    scripts/submit_retrieval_bingrid_v1_postprocess.sbatch)
echo "submitted scoring/report/plots job ${POST_JOBID} (runs after ${ARRAY_JOBID} completes: "
echo "gd_joint_block_matrix.py --tag retrieval_bingrid_v1 for scoring, then "
echo "gd_retrieval_bingrid_plot.py for figures -- no re-run of any config)"
echo
echo "check progress:  sacct -j ${ARRAY_JOBID},${POST_JOBID} --format=JobID,JobName,State,Elapsed"
echo "report lands at: results/config_matrix/retrieval_bingrid_v1/REPORT.md"
echo "plots land at:   plots/config_matrix/retrieval_bingrid_v1/"
