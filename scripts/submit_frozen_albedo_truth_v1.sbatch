#!/bin/bash
# Consistency check (2026-08-29, user): whole-slit sweep with albedo FROZEN
# at the true per-position values (not part of the free/retrieved state),
# only co2_ppm and p_surface_hpa free -- a regression check that the new
# sub_bin_anomaly-era codebase reproduces the same behavior as older,
# pre-heterogeneous-surface baselines for this simpler configuration.
#
# Required gd_joint_block_whole_slit_sweep.py fix (2026-08-29, this same
# check): band_label used to be gated on "albedo" in --free alone, so
# --vary-albedo without freeing albedo silently fell back to a single
# fixed scalar albedo in the forward model despite a spatially-varying
# TRUTH -- unreachable before (every prior sweep either froze albedo AND
# rendered it constant, or freed it AND varied it). Now band_label is
# added whenever --vary-albedo is set, regardless of --free, so a frozen
# row's own prior is genuinely the true per-bin-center value (surface_
# fields defaults to als.SURFACE_FIELDS, exact truth) -- verified directly:
# max|frozen albedo value - truth| == 0.0 on a smoke-test window.
#
# anchor_density=1 (array index 0 would map here) was already run directly
# (not via this array) and completed successfully -- this array only
# covers the two that got interrupted by an allocation cycle: 4 and 16.
#
#SBATCH --job-name=gd_frozen_albedo_truth_v1
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --array=0-1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_out/frozen_albedo_truth_v1_%A_%a.out

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

ADENSS=(4 16)
AD="${ADENSS[$SLURM_ARRAY_TASK_ID]}"

OUT_DIR="results/realistic_prior/config_matrix/frozen_albedo_truth_v1/results"
mkdir -p "${OUT_DIR}"

echo "task ${SLURM_ARRAY_TASK_ID}: anchor_density=${AD}"

"${PYTHON}" scripts/gd_joint_block_whole_slit_sweep.py \
    --g-ratio 3 --free co2_ppm,p_surface_hpa --vary-albedo \
    --n-windows 58 --anchor-density "${AD}" \
    --hires-only --jacobian analytic --prior-form exponential --overlap 2 \
    --n-workers "${SLURM_CPUS_PER_TASK}" \
    --out "${OUT_DIR}/co2p-58-g3-ad${AD}-valb-frozenalbedo-truth_analytic.pkl"
