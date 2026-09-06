#!/bin/bash
# Build Mode-1 dense (500m) truth with a WIDER along-slit PSF, simulating
# telescope defocus (2026-09-06, user: "plan some experiments with wider
# PSFs to simulate a defocusing effect ... doesn't affect the spectral
# resolution, just the spatial blurring"). See
# scratch_work/build_whole_slit_truth_defocused.py.
#
# PSF_FWHM_PX must be set explicitly at submission time (no silent
# default) -- submit once per FWHM value, e.g.:
#   sbatch --export=ALL,PSF_FWHM_PX=3 submit_whole_slit_truth_defocused.sbatch
#   sbatch --export=ALL,PSF_FWHM_PX=5 submit_whole_slit_truth_defocused.sbatch
#   sbatch --export=ALL,PSF_FWHM_PX=8 submit_whole_slit_truth_defocused.sbatch
#
#SBATCH --job-name=gd_whole_slit_truth_defocused
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_out/whole_slit_truth_defocused_%j.out

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

# PSF_FWHM_PX must be set in the environment before submitting.
: "${PSF_FWHM_PX:?PSF_FWHM_PX must be set, e.g. --export=ALL,PSF_FWHM_PX=5}"

"${PYTHON}" scratch_work/build_whole_slit_truth_defocused.py
