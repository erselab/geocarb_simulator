#!/bin/bash
# Build whole-slit TRUTH images band-limited to the retrieval's ANCHOR
# grid at anchor_density 4 and 16 (2026-09-10). The g_ratio-bins=3 truth
# tried earlier does NOT give a representable scene at G=3: (a) the
# whole-slit bin-center union interleaves neighbour-window albedo knots
# a 3-knot window state can't match, (b) it band-limits albedo only to
# ~13km within a 27km window while als._albedo_fine_field is a 0.5km-
# correlated 10% field. Matching to a fine ANCHOR grid (ad16 -> ~0.19km)
# band-limits albedo below its own correlation length, so the retrieval
# at --anchor-density 16 can represent it exactly.
#
#SBATCH --job-name=rmtruth_ad_build
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_out/rmtruth_ad_build_%j.out

set -euo pipefail
module load git/2.50.1
cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

git rev-parse HEAD
"${PYTHON}" scripts/gd_build_resolution_matched_truth.py --fpa 2 --anchor-density 4 16
echo "BUILD DONE"
cat results/resolution_matched_truth/manifest.json
