#!/bin/bash
# Scoring + REPORT.md + plots for the flat-sy-inv reproduction run (see
# submit_retrieval_prior_flatsyinv.sbatch's own header). Meant to run as a
# SLURM dependency AFTER that array job completes (see
# scripts/run_retrieval_prior_flatsyinv.sh, which submits both with the
# dependency wired up automatically) -- not meant to be submitted
# standalone unless the array job already finished on its own.
#
# --flat-sy-inv is passed to BOTH gd_joint_block_matrix.py calls below,
# not just the array job -- resolve_existing() checks each candidate
# file's own flat_sy_inv metadata against what's REQUESTED (see its own
# docstring), so omitting --flat-sy-inv here would make it treat every
# file this array job just wrote as "not found" (metadata says
# flat_sy_inv=True, request would say False) and re-run a full solve for
# all 30 configs from scratch -- exactly the redundant-compute case
# resolve_existing() exists to avoid.
#
#SBATCH --job-name=gd_retrieval_prior_flatsyinv_post
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_out/retrieval_prior_flatsyinv_postprocess_%j.out

set -euo pipefail
module load git/2.50.1

cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python

"${PYTHON}" scripts/gd_joint_block_matrix.py \
    --tag retrieval_prior_co2plus1pct_flatsyinv --free co2 \
    --g-ratio 0.5,1,3,6,12 --anchor-density 1,4,16 \
    --prior-fields co2_plus1pct \
    --hires-only --jacobian analytic --flat-sy-inv

"${PYTHON}" scripts/gd_joint_block_matrix.py \
    --tag retrieval_prior_co2p_flatsyinv --free co2p \
    --g-ratio 0.5,1,3,6,12 --anchor-density 1,4,16 \
    --prior-fields co2_plus1pct_psurf_minus1pct \
    --hires-only --jacobian analytic --flat-sy-inv

"${PYTHON}" scripts/gd_retrieval_bingrid_plot.py --tag retrieval_prior_co2plus1pct_flatsyinv
"${PYTHON}" scripts/gd_retrieval_bingrid_plot.py --tag retrieval_prior_co2p_flatsyinv
