#!/bin/bash
# PROJECT_STATUS.md Sec.5 Phase 1: prior=truth at the bin scale,
# state_interp="linear" downscale onto a fixed anchor grid, real
# gauss_newton_state solves (not forward-only) -- CO2-only and
# CO2+p_surface, swept across g_ratio x anchor_density. 2 free-sets x 5
# g_ratios x 3 anchor_densities = 30 independent configs, one SLURM array
# task per config (unlike submit_whole_slit_1x.sbatch/
# submit_whole_slit_factory.sbatch, which array over WINDOWS within a
# single config -- here each task runs its own config's FULL 58-window
# solve in one process, via gd_joint_block_whole_slit_sweep.py's own
# internal multiprocessing Pool across windows, --n-tasks/--task-id left
# unset).
#
# Output filenames match gd_joint_block_matrix.py's own config_id()
# naming exactly (`{free}-58-g{gratio}[-ad{adens}]_analytic.pkl` under
# results/config_matrix/retrieval_bingrid_v1/results/) so
# submit_retrieval_bingrid_v1_postprocess.sbatch can score everything via
# gd_joint_block_matrix.py itself afterward -- that script's own
# resolve_existing() finds every .pkl already on disk and skips straight
# to scoring/REPORT.md, no re-run, no duplicated scoring logic needed here.
#
# Per-task cost: timed directly on a local M4 Pro (14 cores, 12 workers)
# before submitting -- co2p/g_ratio=1/anchor_density=4: 4.8 min,
# anchor_density=16: 8.1 min (results/config_matrix/anchor_density_v1/,
# retrieval_bingrid_timing/ REPORT.md). anchor_density is the dominant
# cost driver (RT evaluations per window); g_ratio should matter much
# less (only sets state size G, a small per-window linear-solve cost, not
# an RT cost) -- see PROJECT_STATUS.md Sec.5's own reasoning. --time below
# is a safety margin on that measured worst case, not a tight estimate.
#
# CHECK BEFORE SUBMITTING: --account is confirmed (ltmurray_lab); --partition
# is a guess (atmos/atmos2 look like the right domain-matched partitions
# on this cluster, `sinfo` for the full list) -- adjust to whatever
# allocation you actually want to use. --cpus-per-task=16 mirrors this
# repo's other single-task-does-all-windows jobs (gd_test.slurm uses 32
# for a heavier multi-band case) rather than the 4-per-task used by the
# window-array scripts, since here each array task parallelizes its own
# 58 windows internally -- reduce if 30 tasks x 16 cpus oversubscribes
# your allocation.
#
#SBATCH --job-name=gd_retrieval_bingrid_v1
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --array=0-29
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_out/retrieval_bingrid_v1_%A_%a.out

set -euo pipefail
module load git/2.50.1

cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python

# Avoid BLAS oversubscription against the window-level multiprocessing
# Pool inside gd_joint_block_whole_slit_sweep.py itself -- same reasoning
# gd_joint_block_matrix.py's own run_one() documents for its subprocess
# env.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TAG=retrieval_bingrid_v1
OUT_DIR="results/config_matrix/${TAG}/results"
mkdir -p "${OUT_DIR}"

# -- axes, in the SAME nested order as gd_joint_block_matrix.py's own
# itertools.product(frees, nwins, gratios, adenss, sis, scenes) would
# produce for --free co2,co2p --g-ratio 0.5,1,3,6,12 --anchor-density
# 1,4,16 --n-windows 58 (nwins/si/scene are singleton axes here, so this
# reduces to free (slowest) x g_ratio x anchor_density (fastest)) --
# purely for readability/parity with the matrix tool; array-index ->
# config mapping only needs to be a bijection over all 30, which this is.
FREE_KEYS=(co2 co2p)
GRATIOS=(0.5 1 3 6 12)
ADENSS=(1 4 16)
N_G=${#GRATIOS[@]}
N_AD=${#ADENSS[@]}

IDX="${SLURM_ARRAY_TASK_ID}"
FREE_IDX=$(( IDX / (N_G * N_AD) ))
REM=$(( IDX % (N_G * N_AD) ))
G_IDX=$(( REM / N_AD ))
AD_IDX=$(( REM % N_AD ))

FREE_KEY="${FREE_KEYS[$FREE_IDX]}"
G="${GRATIOS[$G_IDX]}"
AD="${ADENSS[$AD_IDX]}"

case "${FREE_KEY}" in
    co2)  FREE_SET="co2_ppm" ;;
    co2p) FREE_SET="co2_ppm,p_surface_hpa" ;;
    *) echo "unknown free key ${FREE_KEY}" >&2; exit 1 ;;
esac

# gd_joint_block_matrix.py's config_id(): elides adens==1 (never elides
# g_ratio, state_interp="linear"/scene="realistic" are this sweep's only
# values so never appended here either).
CID="${FREE_KEY}-58-g${G}"
if [[ "${AD}" != "1" ]]; then
    CID="${CID}-ad${AD}"
fi

echo "task ${IDX}: free=${FREE_KEY} (${FREE_SET})  g_ratio=${G}  anchor_density=${AD}  cid=${CID}"

"${PYTHON}" scripts/gd_joint_block_whole_slit_sweep.py \
    --g-ratio "${G}" \
    --free "${FREE_SET}" \
    --n-windows 58 \
    --anchor-density "${AD}" \
    --hires-only \
    --jacobian analytic \
    --n-workers "${SLURM_CPUS_PER_TASK}" \
    --out "${OUT_DIR}/${CID}_analytic.pkl"

# Don't submit this file directly for the full sweep -- use
# `bash scripts/run_retrieval_bingrid_v1.sh` instead, which submits this
# array job AND chains submit_retrieval_bingrid_v1_postprocess.sbatch
# (scoring + REPORT.md, via gd_joint_block_matrix.py itself) after it via
# --dependency=afterok.
