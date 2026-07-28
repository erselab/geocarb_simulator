#!/bin/bash
#SBATCH --job-name=gd_joint
#SBATCH --partition=atmos2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sean.crowell@rochester.edu

#
# Joint two-band (no-aerosol) stress test -- one SLURM task per scene/noise
# combination, for a single fixed band pair (default FPA0+FPA2, the pair
# KEYSTONE_SMILE_BIAS_PLAN.md Sec. 11 proposes). All three pipelines
# (native/rectified/undistorted) run within one task, mirroring
# gd_band_stress_test.slurm's per-FPA task structure.
#
# Env vars (all optional, unset/0 = script default):
#   FPA_A=<int>, FPA_B=<int>   -- band pair (default 0, 2)
#   UNIFORM=1                  -- flat truth composition (null test)
#   BARCODE=1                  -- diffuser-style barcode illumination
#     (mutually exclusive with UNIFORM=1, matches gd_band_stress_test.py)
#   BARCODE_BARS=<int>         -- bar count for BARCODE=1 (default 32)
#   NOISE=1                    -- add SNR-based Gaussian noise to both bands'
#     raw rendered images (shared across pipelines within each band, same
#     mechanism as gd_band_stress_test.py's --noise)
#   SNR=<float>                -- override both bands' SNR (default: the
#     more conservative of the two bands' own DEFAULT_SNR_BY_FPA)
#   NOISE_SEED=<int>           -- seed for NOISE=1's draw (default 0)
#   PIPELINES=<csv>            -- subset of native,rectified,undistorted
#     (default: all three)
#   ROW_STEP=<int>             -- subsample fpa_a's native rows (default 1,
#     every row -- also controls the rectified shared-grid's sampling
#     stride, see gd_joint_band_test.py)
#
# Runtime: realistic-scene rendering is the slow part (~110s per band on
# this pair, ~220s for both, vs ~10s total for uniform/barcode) -- checked
# 2026-07-27. Full native+undistorted (nearest-row-paired, ~1007 rows each)
# plus rectified (~1024 shared-grid points) is roughly 3000 joint
# retrievals per scene/noise combination; the one full run done
# interactively (native only, order=0, uniform, 1007 rows) took 585s at
# ~16-worker parallelism -- scale accordingly for 3 pipelines at this
# job's 32 cpus. Time limit above is generous, not a tight estimate --
# tighten once a task's actual runtime is known from its own .out file.
#
# Submit a single combination directly, e.g. realistic scene + noise:
#   NOISE=1 sbatch scripts/gd_joint_band_test.slurm
# Or use scripts/gd_joint_band_test_submit_all.sh for the full 6-combination
# battery (3 scenes x 2 noise settings) on the default FPA0+FPA2 pair.

set -euo pipefail

REPO_ROOT="/scratch/scrowel3_lab/geocarb_simulator"
GERT_ROOT="/scratch/scrowel3_lab/gert"
PYTHON="/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python"

cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${GERT_ROOT}"

FPA_A="${FPA_A:-0}"
FPA_B="${FPA_B:-2}"
ROW_STEP="${ROW_STEP:-1}"

SCENE_FLAGS=()
if [[ "${UNIFORM:-0}" == "1" ]]; then
    SCENE_FLAGS+=(--uniform)
fi
if [[ "${BARCODE:-0}" == "1" ]]; then
    SCENE_FLAGS+=(--barcode)
fi
if [[ -n "${BARCODE_BARS:-}" ]]; then
    SCENE_FLAGS+=(--barcode-bars "${BARCODE_BARS}")
fi

EXTRA_FLAGS=()
if [[ "${NOISE:-0}" == "1" ]]; then
    EXTRA_FLAGS+=(--noise)
fi
if [[ -n "${SNR:-}" ]]; then
    EXTRA_FLAGS+=(--snr "${SNR}")
fi
if [[ -n "${NOISE_SEED:-}" ]]; then
    EXTRA_FLAGS+=(--noise-seed "${NOISE_SEED}")
fi
if [[ -n "${PIPELINES:-}" ]]; then
    EXTRA_FLAGS+=(--pipelines "${PIPELINES}")
fi

# n-workers left unset -- gd_joint_band_test.py auto-detects via
# geocarb_gert.gd_render.available_cpus() (os.sched_getaffinity), same
# convention as gd_band_stress_test.slurm.
CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "Host: $(hostname)   FPA_A: ${FPA_A}   FPA_B: ${FPA_B}   CPUs: ${CPUS}" \
     "Uniform: ${UNIFORM:-0}   Barcode: ${BARCODE:-0}   Noise: ${NOISE:-0}" \
     "SNR: ${SNR:-default}   RowStep: ${ROW_STEP}"

"$PYTHON" scripts/gd_joint_band_test.py --fpa-a "${FPA_A}" --fpa-b "${FPA_B}" \
    --row-step "${ROW_STEP}" \
    "${SCENE_FLAGS[@]}" "${EXTRA_FLAGS[@]}"
