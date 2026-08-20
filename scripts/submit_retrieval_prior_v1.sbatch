#!/bin/bash
# Phase 2, first try (PROJECT_STATUS.md Sec.5): two imperfect, uniformly-
# biased priors (geocarb_gert/along_slit_scene.py's PRIOR_FIELD_SETS,
# 2026-08-20), each paired with the free row(s) it actually mis-informs --
# NOT a full free x prior_fields cross product (gd_joint_block_matrix.py's
# own itertools.product would also generate the two MISMATCHED pairings,
# e.g. free=co2 scored against a prior that's also wrong about p_surface
# but frozen there -- a real, different experiment, just not this one):
#
#   setup A: --free co2  --prior-fields co2_plus1pct
#            (the only free row is exactly the one the prior mis-states)
#   setup B: --free co2p --prior-fields co2_plus1pct_psurf_minus1pct
#            (both free rows are exactly the two the prior mis-states)
#
# Both swept across g_ratio (0.5,1,3,6,12) x anchor_density (1,4,16),
# matching Phase 1's own full axis exactly. 5 g_ratios x 3 anchor_densities
# x 2 setups = 30 independent configs, one SLURM array task per config,
# same "one task = one full 58-window sweep, internal multiprocessing Pool
# across windows" pattern as submit_retrieval_bingrid_v1.sbatch.
#
# Output filenames match gd_joint_block_matrix.py's own config_id() naming
# exactly (verified directly before writing this file), under TWO tags
# (one per setup, since the two setups use different --free/--prior-fields
# and the matrix tool's own report is per-tag) --
# results/config_matrix/retrieval_prior_co2plus1pct_v1/results/ and
# results/config_matrix/retrieval_prior_co2p_v1/results/ -- so
# submit_retrieval_prior_v1_postprocess.sbatch's own resolve_existing()
# finds every .pkl already on disk and skips straight to scoring, no
# re-run, no duplicated scoring logic.
#
# Per-task cost: no direct timing for a 1%-bias-prior config yet, but nothing
# about a wrong prior LEVEL changes RT cost -- expect similar to Phase 1's
# own ad4 numbers (co2p-58-g1-ad4 timed ~4.8 min, see PROJECT_STATUS.md
# Sec.5). --time below is a safety margin, not a tight estimate.
#
# CHECK BEFORE SUBMITTING: --account is confirmed (ltmurray_lab); --partition
# is a guess (atmos/atmos2 look like the right domain-matched partitions on
# this cluster, `sinfo` for the full list) -- adjust to whatever allocation
# you actually want to use.
#
#SBATCH --job-name=gd_retrieval_prior_v1
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --array=0-29
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_out/retrieval_prior_v1_%A_%a.out

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

# -- axes, in the same nested order (setup slowest, g_ratio, anchor_density
# fastest) submit_retrieval_bingrid_v1.sbatch uses for its own free x
# g_ratio x anchor_density array; array-index -> config mapping only needs
# to be a bijection over all 30, which this is.
GRATIOS=(0.5 1 3 6 12)
ADENSS=(1 4 16)
N_G=${#GRATIOS[@]}
N_AD=${#ADENSS[@]}

IDX="${SLURM_ARRAY_TASK_ID}"
SETUP_IDX=$(( IDX / (N_G * N_AD) ))
REM=$(( IDX % (N_G * N_AD) ))
G_IDX=$(( REM / N_AD ))
AD_IDX=$(( REM % N_AD ))
G="${GRATIOS[$G_IDX]}"
AD="${ADENSS[$AD_IDX]}"

if [[ "${SETUP_IDX}" == "0" ]]; then
    TAG=retrieval_prior_co2plus1pct_v1
    FREE_SET=co2_ppm
    PRIOR_FIELDS=co2_plus1pct
    CID="co2-58-g${G}"
else
    TAG=retrieval_prior_co2p_v1
    FREE_SET=co2_ppm,p_surface_hpa
    PRIOR_FIELDS=co2_plus1pct_psurf_minus1pct
    CID="co2p-58-g${G}"
fi
if [[ "${AD}" != "1" ]]; then
    CID="${CID}-ad${AD}"
fi
CID="${CID}-pf-${PRIOR_FIELDS}"

OUT_DIR="results/config_matrix/${TAG}/results"
mkdir -p "${OUT_DIR}"

echo "task ${IDX}: tag=${TAG} free=${FREE_SET} g_ratio=${G} anchor_density=${AD} prior_fields=${PRIOR_FIELDS} cid=${CID}"

"${PYTHON}" scripts/gd_joint_block_whole_slit_sweep.py \
    --g-ratio "${G}" --free "${FREE_SET}" \
    --n-windows 58 --anchor-density "${AD}" \
    --hires-only --jacobian analytic \
    --prior-fields "${PRIOR_FIELDS}" \
    --n-workers "${SLURM_CPUS_PER_TASK}" \
    --out "${OUT_DIR}/${CID}_analytic.pkl"

# Don't submit this file directly for the full pipeline -- use
# `bash scripts/run_retrieval_prior_v1.sh` instead, which submits this
# array job AND chains submit_retrieval_prior_v1_postprocess.sbatch
# (scoring + REPORT.md for each tag, via gd_joint_block_matrix.py itself)
# after it via --dependency=afterok.
