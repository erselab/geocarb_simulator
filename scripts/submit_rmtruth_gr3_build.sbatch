#!/bin/bash
# Build the whole-slit TRUTH image band-limited to the retrieval's own
# state BIN grid at g_ratio=3 (the retrieval default) -- "truth matched
# to the resolution of the retrieval" (2026-09-10, user). This removes
# the bin-grid-vs-continuous-truth representability gap entirely: the
# retrieval's bin-to-bin piecewise-linear state reconstruction becomes
# bit-for-bit the same function the truth was built from.
#
# Cached under resolution_tag gr3bins-<hash>; the retrieval side then
# reconstructs the identical whole_slit_bin_centers grid to hit this
# cache via --resolution-matched-g-ratio-bins 3.
#
#SBATCH --job-name=rmtruth_gr3_build
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_out/rmtruth_gr3_build_%j.out

set -euo pipefail
module load git/2.50.1

cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

git rev-parse HEAD
"${PYTHON}" scripts/gd_build_resolution_matched_truth.py --fpa 2 --g-ratio-bins 3
echo "BUILD DONE"
cat results/resolution_matched_truth/manifest.json
