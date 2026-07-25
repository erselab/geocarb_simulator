#!/bin/bash
#SBATCH --job-name=gd_band_stress
#SBATCH --partition=atmos2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --array=0,1,2,3
#SBATCH --output=%x_fpa%a_%j.out
#SBATCH --error=%x_fpa%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sean.crowell@rochester.edu

#
# Composition/pressure stress test (scripts/gd_band_stress_test.py), one
# SLURM array task per FPA. Submit with:
#
#   sbatch scripts/gd_band_stress_test.slurm
#
# --array=0,1,2,3 : all four bands are now clear to run (as of 2026-07-24):
#   FPA0/O2_A: was on a razor-thin ABSCO margin (1.99 cm-1 at margin_cm1=10)
#     -- ABSCO's O2 block has since been padded to 12745-13300. Also fixed:
#     o2_scale was floating alongside p_scale (both nearly degenerate for a
#     well-mixed gas), causing near-100% non-convergence -- o2 is now
#     excluded from the retrieved gas list (see WELL_MIXED_GASES in
#     gd_band_stress_test.py); p_scale alone carries the surface-pressure
#     signal, as in an operational O2-A retrieval.
#   FPA1/CO2_weak: unaffected by the above, already validated at full scale
#     (see KEYSTONE_SMILE_BIAS_PLAN.md) -- included here for a fresh run
#     picking up the undistorted-pipeline sampling-density fix below.
#   FPA2/CO2_strong: previously excluded because it already had dense-sweep
#     results (gd_dense_sweep.py) from a uniform (composition-invariant)
#     scene -- that tests a different, narrower question (pure geometric
#     distortion in isolation) and was never run through *this* along-slit
#     scene. Added here for full 4-band coverage under the same realistic
#     truth atmosphere as the others.
#   FPA3/CH4_CO: was hard-blocked on ABSCO gaps in ch4/h2o/co (now extended
#     to 4400 cm-1). Also required correcting GEOCARB_BANDS[3]'s nominal
#     band -- it was [4208,4318], offset ~41 cm-1 from the real per-pixel
#     band (~[4259,4350], confirmed against ground-test wavelength data),
#     which silently discarded ~1/3 of every row's real spectral content in
#     the rectified/undistorted pipelines. Corrected to [4258.60,4350.69]
#     (see geocarb_gert/instrument.py).
#   All four bands also pick up: h2o_scale's prior uncertainty raised from
#   the 10% default to 60% (the along-slit scene's true h2o swing is up to
#   +-56%, ~5.6 prior-sigma at 10% -- an unrealistically tight prior for
#   this scene), and the undistorted pipeline now evaluates each row on its
#   own real per-pixel grid (matching native exactly) instead of a separate,
#   ~38%-coarser shared nominal grid -- removes a sampling-density confound
#   from the native-vs-undistorted comparison, on top of the distortion
#   difference that comparison is meant to isolate.
#
# --cpus-per-task: gd_band_stress_test.py auto-detects available cores
# (geocarb_gert.gd_render.available_cpus(), via os.sched_getaffinity) and
# uses all of them by default -- raise this to whatever the target node
# actually has (checked 2026-07-24: "standard" partition nodes have
# 56-96 cores; bhdrb6x/bhdrb8x nodes are the 96-core ones) for a
# proportional speedup. Memory stays low regardless of core count -- the
# ~2.2GB ABSCO table is shared read-only across all worker processes via
# fork copy-on-write, not duplicated per worker.
#
# Runtime: varies a lot by band -- FPA1's forward-model calls are cheap
# (~0.04s each) vs FPA0's O2-A calls (much more expensive; FPA0's first,
# pre-fix run took ~99 min at 32 cores, almost entirely wasted on non-
# convergent retrievals hitting max_iter every time -- the o2_scale/p_scale
# fix above should cut that dramatically since retrievals now converge in a
# handful of iterations instead of always running all 14, but this hasn't
# been verified at full 1024-row scale yet, only smoke-tested). At 32 cores
# every band should be well under the 4-hour limit above; increase --time
# if using --row-step 1 on a smaller core count, or decrease it once you've
# seen each array task's actual runtime in its .out file.
#
# --uniform comparison: for a with/without along-slit-variation comparison
# (isolates pure geometric-distortion bias from composition-tracking bias,
# see KEYSTONE_SMILE_BIAS_PLAN.md) set UNIFORM=1, e.g. to run FPA2 both ways:
#   sbatch --array=2 scripts/gd_band_stress_test.slurm
#   UNIFORM=1 sbatch --array=2 scripts/gd_band_stress_test.slurm
# Not part of the default --array=0,1,2,3 job above.
#
# Added 2026-07-25 (all opt-in via env var, unset/0 = off, matching UNIFORM
# above -- none of these are part of the default --array=0,1,2,3 job):
#   NOISE=1        -- add a random Gaussian noise realization (see SNR/
#                     NOISE_SEED below), shared consistently across all
#                     three pipelines. Sy_inv weighting is always SNR-based
#                     now regardless of NOISE (gd_band_stress_test.py's
#                     sigma_band) -- this only controls whether an actual
#                     noisy realization gets added to the simulated data.
#   SNR=<float>    -- override this band's default SNR (400/300/300/200
#                     for FPA0-3, gd_band_stress_test.py's
#                     DEFAULT_SNR_BY_FPA). Affects Sy_inv weighting always;
#                     affects the actual noise draw only when NOISE=1.
#   NOISE_SEED=<int> -- seed for NOISE=1's random draw (default 0,
#                     reproducible).
#   BARCODE=1      -- diffuser-style barcode illumination (Sec. 9j) instead
#                     of the along-slit composition scene -- mutually
#                     exclusive with UNIFORM=1 (the python script errors
#                     clearly if both are set).
#   BARCODE_BARS=<int> -- bar count for BARCODE=1 (default 32, Sec. 9j's
#                     original test).
# e.g. a noisy FPA1 run at the default SNR:
#   NOISE=1 sbatch --array=1 scripts/gd_band_stress_test.slurm

set -euo pipefail

REPO_ROOT="/scratch/scrowel3_lab/geocarb_simulator"
GERT_ROOT="/scratch/scrowel3_lab/gert"
PYTHON="/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python"

cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${GERT_ROOT}"

# Falls back to $FPA (or 0) when run directly via `bash` instead of
# `sbatch` -- SLURM_ARRAY_TASK_ID/SLURM_CPUS_PER_TASK are only set inside
# an actual SLURM allocation, and `set -u` above would otherwise treat
# either as a fatal "unbound variable" rather than running with a sane
# default. To test a specific band outside SLURM: `FPA=0 bash scripts/
# gd_band_stress_test.slurm`.
FPA="${SLURM_ARRAY_TASK_ID:-${FPA:-0}}"
CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"

# UNIFORM=1 disables along-slit composition/pressure variation (see the
# --uniform comparison note above) -- unset/0 by default.
UNIFORM_FLAG=()
if [[ "${UNIFORM:-0}" == "1" ]]; then
    UNIFORM_FLAG=(--uniform)
fi

# NOISE=1 / SNR / NOISE_SEED / BARCODE=1 / BARCODE_BARS -- see the "Added
# 2026-07-25" note above. All opt-in, unset/0 = gd_band_stress_test.py's
# own defaults (no noise, band-default SNR, along-slit scene).
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
if [[ "${BARCODE:-0}" == "1" ]]; then
    EXTRA_FLAGS+=(--barcode)
fi
if [[ -n "${BARCODE_BARS:-}" ]]; then
    EXTRA_FLAGS+=(--barcode-bars "${BARCODE_BARS}")
fi

echo "Host: $(hostname)   FPA: ${FPA}   CPUs: ${CPUS}   Uniform: ${UNIFORM:-0}" \
     "Noise: ${NOISE:-0}   SNR: ${SNR:-default}   Barcode: ${BARCODE:-0}"

"$PYTHON" scripts/gd_band_stress_test.py --fpa "${FPA}" "${UNIFORM_FLAG[@]}" "${EXTRA_FLAGS[@]}"
