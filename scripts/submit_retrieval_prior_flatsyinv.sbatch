#!/bin/bash
# Deliberate reproduction of the pre-Phase-D "flat scalar Sy_inv" imperfect-
# prior results (docs/PROJECT_STATUS.md Sec.6), for direct comparison
# against the corrected (real noise model) rerun -- the ORIGINAL raw pkls
# from that old run were overwritten before anyone thought to save them
# (results/ is gitignored, no git history either), so this is the only way
# to get old-vs-new comparable per-window data again. Otherwise identical
# to submit_retrieval_prior_v1.sbatch (same axes, same two setups, same
# --n-windows/--anchor-density/--jacobian) -- the only difference is
# --flat-sy-inv on the sweep call and a distinct _flatsyinv tag suffix, so
# this can NEVER collide with (or be silently skipped in favor of) the
# corrected-weighting results at retrieval_prior_{co2plus1pct,co2p}_v1/v2.
#
#   setup A: --free co2  --prior-fields co2_plus1pct
#   setup B: --free co2p --prior-fields co2_plus1pct_psurf_minus1pct
#
# 5 g_ratios (0.5,1,3,6,12) x 3 anchor_densities (1,4,16) x 2 setups = 30
# independent configs, one SLURM array task per config, same pattern as
# submit_retrieval_prior_v1.sbatch.
#
# Output: results/config_matrix/retrieval_prior_co2plus1pct_flatsyinv/results/
#         results/config_matrix/retrieval_prior_co2p_flatsyinv/results/
# gd_joint_block_matrix.py's own resolve_existing() (extended for this) also
# checks each candidate file's own flat_sy_inv metadata before treating it
# as a match, so even reusing an existing tag by mistake can't silently
# score the wrong weighting -- see gd_joint_block_matrix.py's own
# resolve_existing() docstring.
#
# Per-task cost: should generally be FASTER than the original v1 run --
# results/truth_cache/ already holds this exact scene (fpa2/realistic/
# n_lookup_samples=5600/vary_albedo=False) from the corrected rerun, so
# every task here gets a truth-cache HIT and skips the render step
# entirely; only the Gauss-Newton solve itself (unchanged by --flat-sy-inv,
# same cost either way) remains. --time is kept at the same value as
# submit_retrieval_prior_v1.sbatch's own (post-incident) budget as a safety
# margin, not re-tuned down, since solve time is what dominated the
# longest configs there (co2p-58-g0.5-ad16 came within a minute of the
# original 30-min budget) and that part hasn't changed.
#
# CHECK BEFORE SUBMITTING: --account is confirmed (ltmurray_lab); --partition
# is a guess (atmos/atmos2 look like the right domain-matched partitions on
# this cluster, `sinfo` for the full list) -- adjust to whatever allocation
# you actually want to use.
#
#SBATCH --job-name=gd_retrieval_prior_flatsyinv
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --array=0-29
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_out/retrieval_prior_flatsyinv_%A_%a.out

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

# -- axes, same nested order (setup slowest, g_ratio, anchor_density
# fastest) as submit_retrieval_prior_v1.sbatch's own array-index mapping.
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
    TAG=retrieval_prior_co2plus1pct_flatsyinv
    FREE_SET=co2_ppm
    PRIOR_FIELDS=co2_plus1pct
    CID="co2-58-g${G}"
else
    TAG=retrieval_prior_co2p_flatsyinv
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

echo "task ${IDX}: tag=${TAG} free=${FREE_SET} g_ratio=${G} anchor_density=${AD} prior_fields=${PRIOR_FIELDS} cid=${CID} (flat-sy-inv)"

"${PYTHON}" scripts/gd_joint_block_whole_slit_sweep.py \
    --g-ratio "${G}" --free "${FREE_SET}" \
    --n-windows 58 --anchor-density "${AD}" \
    --hires-only --jacobian analytic --flat-sy-inv \
    --prior-fields "${PRIOR_FIELDS}" \
    --n-workers "${SLURM_CPUS_PER_TASK}" \
    --out "${OUT_DIR}/${CID}_analytic.pkl"

# Don't submit this file directly for the full pipeline -- use
# `bash scripts/run_retrieval_prior_flatsyinv.sh` instead, which submits
# this array job AND chains submit_retrieval_prior_flatsyinv_postprocess.
# sbatch (scoring + REPORT.md for each tag) after it via
# --dependency=afterok.
