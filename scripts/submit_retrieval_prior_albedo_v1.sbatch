#!/bin/bash
# First real sweep with albedo in the free/retrieved state vector
# (docs/PROJECT_STATUS.md Sec.6, 2026-08-25): --free co2_ppm,p_surface_hpa,
# albedo against the real spatially-varying surface truth scene
# (--vary-albedo, geocarb_gert/along_slit_scene.py's SURFACE_PATCHES + fine
# texture), all previous smoke/debug runs were throwaway single-window
# checks, never a real g_ratio x anchor_density sweep. Two setups:
#
#   setup 0: --prior-fields exact       (prior = truth, including the same
#            patch/fine-texture realization -- validates that a free
#            albedo row can recover the known truth before anything is
#            deliberately wrong)
#   setup 1: --prior-fields structural  (prior knows the patch layout, not
#            the fine texture -- als.SURFACE_FIELDS_PRIOR/albedo_for_label
#            _prior -- the imperfect-surface-prior test this whole feature
#            was built for)
#
# Both swept across g_ratio (0.5,1,3,6,12) x anchor_density (1,4,16),
# matching every other prior sweep's own axis. 5 g_ratios x 3
# anchor_densities x 2 setups = 30 independent configs, one SLURM array
# task per config, same "one task = one full 58-window sweep, internal
# multiprocessing Pool across windows" pattern as
# submit_retrieval_prior_v3.sbatch.
#
# Both setups also use --vary-albedo (required whenever albedo is free --
# gd_joint_block_whole_slit_sweep.py's own validation errors out otherwise),
# --prior-form exponential (today's default, explicit for reproducibility),
# and --overlap 2 (window-boundary merge, same convention as _v3).
#
# Output under TWO tags (one per setup):
#   results/config_matrix/retrieval_prior_albedo_exact_v1/results/
#   results/config_matrix/retrieval_prior_albedo_structural_v1/results/
#
# Per-task cost: UNKNOWN at fine g_ratio -- only ever timed single windows
# at g_ratio=3 (a water-boundary window took 173s hires; a G=15 window at
# the same g_ratio took 244s). Freeing albedo roughly triples the free-row
# count per window (co2+p_surface+albedo vs co2+p_surface alone) and
# surface_density=3 makes the albedo grid itself denser still, so expect
# meaningfully slower than _v3's own co2p numbers, worse at g_ratio=0.5.
# --time below is a generous safety margin, not a real estimate -- if any
# task times out, that's the signal to actually measure a real config and
# come back with a tighter number.
#
# CHECK BEFORE SUBMITTING: --account is confirmed (ltmurray_lab); --partition
# is a guess (atmos/atmos2 look right on this cluster, `sinfo` for the full
# list) -- adjust to whatever allocation you actually want to use.
#
#SBATCH --job-name=gd_retrieval_prior_albedo_v1
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --array=0-29
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_out/retrieval_prior_albedo_v1_%A_%a.out

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
# array-index -> config mapping only needs to be a bijection over all 30.
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

FREE_SET=co2_ppm,p_surface_hpa,albedo
if [[ "${SETUP_IDX}" == "0" ]]; then
    TAG=retrieval_prior_albedo_exact_v1
    PRIOR_FIELDS=exact
    CID="co2p_albedo-58-g${G}"
else
    TAG=retrieval_prior_albedo_structural_v1
    PRIOR_FIELDS=structural
    CID="co2p_albedo-58-g${G}"
fi
if [[ "${AD}" != "1" ]]; then
    CID="${CID}-ad${AD}"
fi
# config_id() (gd_joint_block_matrix.py) elides "-pf-exact" specifically --
# "exact" is the documented default, so its own cid has no -pf- suffix at
# all. Found 2026-08-25: this script originally appended -pf-${PRIOR_FIELDS}
# unconditionally, so the "exact" tag's own files never matched what the
# postprocess step's resolve_existing() actually looked for, forcing a
# (memory-exhausting) regeneration attempt. Match config_id() exactly here.
if [[ "${PRIOR_FIELDS}" != "exact" ]]; then
    CID="${CID}-pf-${PRIOR_FIELDS}"
fi

OUT_DIR="results/config_matrix/${TAG}/results"
mkdir -p "${OUT_DIR}"

echo "task ${IDX}: tag=${TAG} free=${FREE_SET} g_ratio=${G} anchor_density=${AD} prior_fields=${PRIOR_FIELDS} cid=${CID}"

"${PYTHON}" scripts/gd_joint_block_whole_slit_sweep.py \
    --g-ratio "${G}" --free "${FREE_SET}" \
    --n-windows 58 --anchor-density "${AD}" \
    --hires-only --jacobian analytic \
    --prior-fields "${PRIOR_FIELDS}" \
    --prior-form exponential --overlap 2 --vary-albedo \
    --n-workers "${SLURM_CPUS_PER_TASK}" \
    --out "${OUT_DIR}/${CID}_analytic.pkl"

# Don't submit this file directly for the full pipeline -- use
# `bash scripts/run_retrieval_prior_albedo_v1.sh` instead, which submits
# this array job AND chains
# submit_retrieval_prior_albedo_v1_postprocess.sbatch (scoring + REPORT.md
# for each of the two tags) after it via --dependency=afterok.
