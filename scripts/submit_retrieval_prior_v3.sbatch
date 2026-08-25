#!/bin/bash
# Phase 2, "v3" (docs/PROJECT_STATUS.md Sec.6): the two existing imperfect-
# prior setups (co2plus1pct, co2p -- previously run as *_v1/*_v2/*_flatsyinv)
# REDONE under one explicitly-consistent, weakly spatially-correlated
# exponential prior_form (2026-08-21 user request: "redo all of them with
# the same spatially correlated prior uncertainty" -- confirming, not
# changing, today's actual default: input/retrieval_defaults.yml's
# prior_form=exponential, corr_length_eta[co2_ppm]=0.0071 ~10km), PLUS two
# new setups using prior_fields="structural" (geocarb_gert/along_slit_
# scene.py's PRIOR_FIELD_SETS: the prior keeps the truth's background but
# loses ALL localized plume/hot-spot/topography detail -- a genuinely
# different failure mode from co2plus1pct/co2p's uniform multiplicative
# bias, which keeps 100% of the truth's spatial detail and is wrong only
# in absolute level). All four ALSO use --overlap 2 (build_window_tiles/
# along_slit_query.query_state, Sec.6) -- the window-overlap merge fix for
# the boundary bias this same exponential prior_form is documented to
# cause, so this run exercises prior + fix together rather than banking a
# result that will need yet another rerun once merge is turned on:
#
#   setup 0: --free co2   --prior-fields co2_plus1pct
#   setup 1: --free co2p  --prior-fields co2_plus1pct_psurf_minus1pct
#   setup 2: --free co2   --prior-fields structural
#   setup 3: --free co2p  --prior-fields structural
#
# All four swept across g_ratio (0.5,1,3,6,12) x anchor_density (1,4,16),
# matching every prior sweep's own axis exactly. 5 g_ratios x 3
# anchor_densities x 4 setups = 60 independent configs, one SLURM array
# task per config, same "one task = one full 58-window sweep, internal
# multiprocessing Pool across windows" pattern as submit_retrieval_prior_v1
# .sbatch.
#
# Output under FOUR separate tags (one per setup, own --free/--prior-fields
# each) so gd_joint_block_matrix.py's per-tag report/resolve_existing keeps
# working unchanged:
#   results/config_matrix/retrieval_prior_co2plus1pct_v3/results/
#   results/config_matrix/retrieval_prior_co2p_v3/results/
#   results/config_matrix/retrieval_prior_structural_co2_v3/results/
#   results/config_matrix/retrieval_prior_structural_co2p_v3/results/
#
# Per-task cost: no direct timing for overlap=2 + structural yet -- overlap
# widens each internal window boundary by 2 rows on each side, a small
# fraction of even the narrowest (9-row) window, so expect similar to the
# untimed but same-shape co2p-58-g1-ad4 (~4.8 min, Sec.5). --time below is
# a safety margin, not a tight estimate.
#
# CHECK BEFORE SUBMITTING: --account is confirmed (ltmurray_lab); --partition
# is a guess (atmos/atmos2 look right on this cluster, `sinfo` for the full
# list) -- adjust to whatever allocation you actually want to use.
#
#SBATCH --job-name=gd_retrieval_prior_v3
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --array=0-59
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_out/retrieval_prior_v3_%A_%a.out

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

# -- axes, in nested order (setup slowest, g_ratio, anchor_density fastest);
# array-index -> config mapping only needs to be a bijection over all 60.
GRATIOS=(0.5 1 3 6 12)
ADENSS=(1 4 16)
N_G=${#GRATIOS[@]}
N_AD=${#ADENSS[@]}
N_SETUPS=4

IDX="${SLURM_ARRAY_TASK_ID}"
SETUP_IDX=$(( IDX / (N_G * N_AD) ))
REM=$(( IDX % (N_G * N_AD) ))
G_IDX=$(( REM / N_AD ))
AD_IDX=$(( REM % N_AD ))
G="${GRATIOS[$G_IDX]}"
AD="${ADENSS[$AD_IDX]}"

case "${SETUP_IDX}" in
  0)
    TAG=retrieval_prior_co2plus1pct_v3
    FREE_SET=co2_ppm
    PRIOR_FIELDS=co2_plus1pct
    CID="co2-58-g${G}"
    ;;
  1)
    TAG=retrieval_prior_co2p_v3
    FREE_SET=co2_ppm,p_surface_hpa
    PRIOR_FIELDS=co2_plus1pct_psurf_minus1pct
    CID="co2p-58-g${G}"
    ;;
  2)
    TAG=retrieval_prior_structural_co2_v3
    FREE_SET=co2_ppm
    PRIOR_FIELDS=structural
    CID="co2-58-g${G}"
    ;;
  3)
    TAG=retrieval_prior_structural_co2p_v3
    FREE_SET=co2_ppm,p_surface_hpa
    PRIOR_FIELDS=structural
    CID="co2p-58-g${G}"
    ;;
esac
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
    --prior-form exponential --overlap 2 \
    --n-workers "${SLURM_CPUS_PER_TASK}" \
    --out "${OUT_DIR}/${CID}_analytic.pkl"

# Don't submit this file directly for the full pipeline -- use
# `bash scripts/run_retrieval_prior_v3.sh` instead, which submits this
# array job AND chains submit_retrieval_prior_v3_postprocess.sbatch
# (scoring + REPORT.md for each of the four tags, via gd_joint_block_matrix
# .py itself) after it via --dependency=afterok.
