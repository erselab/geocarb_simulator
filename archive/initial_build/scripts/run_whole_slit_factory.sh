#!/bin/bash
# One-command launcher for the whole-slit factory-default (G=width/3)
# rerun -- picks up full residual arrays/state vectors, which the
# original results/gd_joint_block_whole_slit_fpa2.pkl predates. Submits
# the 58-task array (submit_whole_slit_factory.sbatch), then submits the
# merge+plot step (submit_whole_slit_factory_postprocess.sbatch) with a
# SLURM dependency so it fires automatically once every array task
# completes -- no polling, no manual follow-up.
#
# Run:  bash scripts/run_whole_slit_factory.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ARRAY_JOBID=$(sbatch --parsable scripts/submit_whole_slit_factory.sbatch)
echo "submitted whole-slit factory array job ${ARRAY_JOBID} (58 tasks)"

POST_JOBID=$(sbatch --parsable --dependency=afterok:"${ARRAY_JOBID}" scripts/submit_whole_slit_factory_postprocess.sbatch)
echo "submitted merge+plot job ${POST_JOBID} (runs after ${ARRAY_JOBID} completes: merge, "
echo "gd_joint_block_whole_slit_plot.py)"
echo
echo "check progress:  sacct -j ${ARRAY_JOBID},${POST_JOBID} --format=JobID,JobName,State,Elapsed"
