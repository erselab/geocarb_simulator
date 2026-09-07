#!/bin/bash
# One-off diagnostic: rerun a matched defocus config under a much wider
# window (MIN_WINDOW, e.g. 20 -> ~41-row windows) to check whether the
# edge-clamp confound found in Sec.9's per-bin analysis shrinks as
# expected when pad/width ratio drops (2026-09-06, user: "Yes, let's do
# that"). See scratch_work/retrieval_co2palbedo_defocus_truth_
# widewindow_check.py for the full mechanism writeup.
#
# TRUTH_PSF_FWHM_PX, RETRIEVAL_PSF_FWHM_PX, and MIN_WINDOW must all be
# set explicitly at submission time (no silent default). task_id/n_tasks
# must be pre-computed for the target MIN_WINDOW's own tiling (NOT the
# standard 58-window one) -- see build_window_tiles(fpa,
# min_window=<MW>, window_scale=1.0) to find the right task_id.
#
# N_TASKS is the total window count for MIN_WINDOW's own tiling (e.g. 25
# at min_window=20) -- must be set explicitly too: SLURM auto-sets
# SLURM_ARRAY_TASK_COUNT from --array's own index count (1 for a single
# index like --array=5), which is NOT the right value here and would
# silently corrupt the tiling if relied on.
#
# Example (nominal PSF, matched, window idx 5 of 25 at min_window=20):
#   sbatch --export=ALL,TRUTH_PSF_FWHM_PX=1.5,RETRIEVAL_PSF_FWHM_PX=1.5,MIN_WINDOW=20,N_TASKS=25 \
#     --array=5 submit_co2palbedo_defocus_truth_widewindow_check.sbatch
#
#SBATCH --job-name=gd_co2palbedo_defocus_widecheck
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_out/co2palbedo_defocus_widecheck_%A_%a.out

set -euo pipefail
module load git/2.50.1

cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

: "${TRUTH_PSF_FWHM_PX:?TRUTH_PSF_FWHM_PX must be set}"
: "${RETRIEVAL_PSF_FWHM_PX:?RETRIEVAL_PSF_FWHM_PX must be set}"
: "${MIN_WINDOW:?MIN_WINDOW must be set}"
: "${N_TASKS:?N_TASKS must be set (total window count for the MIN_WINDOW tiling)}"
export ANCHOR_WORKERS="${SLURM_CPUS_PER_TASK}"
export SLURM_ARRAY_TASK_COUNT="${N_TASKS}"

"${PYTHON}" scratch_work/retrieval_co2palbedo_defocus_truth_widewindow_check.py
