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
# Multi-band (no-aerosol) stress test -- one SLURM task per scene/noise
# combination, for a fixed, ordered set of >=1 GeoCarb bands (default
# FPA0+FPA2, the pair KEYSTONE_SMILE_BIAS_PLAN.md Sec. 11 proposes; a single
# FPA runs the plain single-band battery, any >=2 bands run a joint
# retrieval). All three pipelines (native/rectified/undistorted) run
# within one task.
#
# Env vars (all optional, unset/0 = script default):
#   FPAS=<csv>                  -- ordered, comma-separated FPA indices,
#     >=1 of them (default "0,2"); the first is the along-slit position
#     reference used for row-pairing and plots (a no-op for a single FPA)
#   UNIFORM=1                  -- flat truth composition (null test)
#   BARCODE=1                  -- diffuser-style barcode illumination
#     (mutually exclusive with UNIFORM=1)
#   BARCODE_BARS=<int>         -- bar count for BARCODE=1 (default 32)
#   NOISE=1                    -- add SNR-based Gaussian noise to every
#     band's raw rendered image (shared across pipelines within each band)
#   SNR=<float>                -- override every band's SNR (default: the
#     most conservative of all bands' own DEFAULT_SNR_BY_FPA)
#   NOISE_SEED=<int>           -- seed for NOISE=1's draw (default 0)
#   PIPELINES=<csv>            -- subset of native,rectified,undistorted
#     (default: all three)
#   ROW_STEP=<int>             -- subsample the reference band's native rows
#     (default 1, every row -- also controls the rectified shared-grid's
#     sampling stride, see gd_test.py)
#
# Runtime: realistic-scene rendering is the slow part (~110s per band on
# the FPA0+FPA2 pair, checked 2026-07-27) and scales with band count since
# each band is rendered independently. Native+undistorted (nearest-row-
# paired across all N bands) plus rectified (N-way shared-grid) is roughly
# 3 x N_rows joint retrievals per scene/noise combination -- row count
# itself only shrinks (never grows) as more bands are added, since pairing/
# rectified coverage is an intersection, but each retrieval gets more
# expensive with more Jacobian columns (one dispersion/albedo block per
# band). Time limit above is generous, not a tight estimate -- tighten once
# a task's actual runtime is known from its own .out file, especially for
# >2-band runs.
#
# Submit a single combination directly, e.g. realistic scene + noise:
#   NOISE=1 sbatch scripts/gd_test.slurm
# Or use scripts/gd_test_submit_all.sh for the full 6-combination
# battery (3 scenes x 2 noise settings) on a given set of bands.

set -euo pipefail

REPO_ROOT="/scratch/scrowel3_lab/geocarb_simulator"
GERT_ROOT="/scratch/scrowel3_lab/gert"
PYTHON="/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python"

cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${GERT_ROOT}"

FPAS="${FPAS:-0,2}"
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

# n-workers left unset -- gd_test.py auto-detects via
# geocarb_gert.gd_render.available_cpus() (os.sched_getaffinity), same
# convention as gd_band_stress_test.slurm.
CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "Host: $(hostname)   FPAs: ${FPAS}   CPUs: ${CPUS}" \
     "Uniform: ${UNIFORM:-0}   Barcode: ${BARCODE:-0}   Noise: ${NOISE:-0}" \
     "SNR: ${SNR:-default}   RowStep: ${ROW_STEP}"

"$PYTHON" scripts/gd_test.py --fpas "${FPAS}" \
    --row-step "${ROW_STEP}" \
    "${SCENE_FLAGS[@]}" "${EXTRA_FLAGS[@]}"
