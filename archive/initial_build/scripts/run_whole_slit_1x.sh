#!/bin/bash
# One-command launcher for the whole-slit 1x (G=width) production run:
# submits the 58-task array (submit_whole_slit_1x.sbatch), then submits
# the merge+plot step (submit_whole_slit_1x_postprocess.sbatch) with a
# SLURM dependency so it fires automatically once every array task
# completes -- no polling, no manual follow-up.
#
# Run:  bash scripts/run_whole_slit_1x.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ARRAY_JOBID=$(sbatch --parsable scripts/submit_whole_slit_1x.sbatch)
echo "submitted whole-slit array job ${ARRAY_JOBID} (58 tasks)"

POST_JOBID=$(sbatch --parsable --dependency=afterok:"${ARRAY_JOBID}" scripts/submit_whole_slit_1x_postprocess.sbatch)
echo "submitted merge+plot job ${POST_JOBID} (runs after ${ARRAY_JOBID} completes: merge, "
echo "gd_joint_block_whole_slit_plot.py, gd_joint_block_whole_slit_gratio_compare.py)"
echo
echo "check progress:  sacct -j ${ARRAY_JOBID},${POST_JOBID} --format=JobID,JobName,State,Elapsed"
