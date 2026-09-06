#!/bin/bash
# Defocus (wide-PSF) experiment, fast 3-window subset (task_ids 21,26,57 --
# rows 189-197, 238-248, 1013-1023 -- 2026-09-06, user: "plan some
# experiments with wider PSFs to simulate a defocusing effect"). Both
# TRUTH_PSF_FWHM_PX and RETRIEVAL_PSF_FWHM_PX must be set explicitly at
# submission time (no silent default -- this session's own GRATIO
# near-miss lesson). See
# scratch_work/retrieval_co2palbedo_defocus_truth_anchorfrozen.py.
#
# Example (matched/corrected defocus):
#   sbatch --export=ALL,TRUTH_PSF_FWHM_PX=5,RETRIEVAL_PSF_FWHM_PX=5 \
#     submit_co2palbedo_defocus_truth_anchorfrozen_fastset.sbatch
# Example (mismatched/uncorrected defocus):
#   sbatch --export=ALL,TRUTH_PSF_FWHM_PX=5,RETRIEVAL_PSF_FWHM_PX=1.5 \
#     submit_co2palbedo_defocus_truth_anchorfrozen_fastset.sbatch
#
#SBATCH --job-name=gd_co2palbedo_defocus
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --array=21,26,57
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_out/co2palbedo_defocus_%A_%a.out

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

export SLURM_ARRAY_TASK_COUNT=58
export ANCHOR_WORKERS="${SLURM_CPUS_PER_TASK}"

: "${TRUTH_PSF_FWHM_PX:?TRUTH_PSF_FWHM_PX must be set, e.g. --export=ALL,TRUTH_PSF_FWHM_PX=5,RETRIEVAL_PSF_FWHM_PX=5}"
: "${RETRIEVAL_PSF_FWHM_PX:?RETRIEVAL_PSF_FWHM_PX must be set, e.g. --export=ALL,TRUTH_PSF_FWHM_PX=5,RETRIEVAL_PSF_FWHM_PX=1.5}"

"${PYTHON}" scratch_work/retrieval_co2palbedo_defocus_truth_anchorfrozen.py
