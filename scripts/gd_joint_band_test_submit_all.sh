#!/bin/bash
# Submit the full joint two-band battery: 3 scenes (realistic, uniform,
# barcode) x 2 noise settings (off, on) = 6 sbatch calls, all three
# pipelines (native/rectified/undistorted) within each -- the joint-
# retrieval counterpart of gd_band_stress_test_submit_all.sh, for a single
# fixed band pair (default FPA0+FPA2; override with FPA_A/FPA_B env vars,
# see gd_joint_band_test.slurm's header for the full env-var list).
#
# Each combination writes its own uniquely-suffixed
# results/gd_joint_fpa<A>_fpa<B>[_uniform|_barcode][_noise].pkl
# (gd_joint_band_test.py's own filename logic), so nothing collides
# regardless of submission order or how many run at once.
#
# Run:  bash scripts/gd_joint_band_test_submit_all.sh
# Or a subset directly, e.g. realistic + noise only:
#   NOISE=1 sbatch scripts/gd_joint_band_test.slurm
#
# This only calls sbatch -- it does not wait for jobs to finish. Check
# progress with `squeue -u $USER`.

set -euo pipefail
cd "$(dirname "$0")/.."

FPA_A="${1:-${FPA_A:-0}}"
FPA_B="${2:-${FPA_B:-2}}"

for NOISE_VAL in 0 1; do
    noise_label="noise=${NOISE_VAL}"

    echo "=== realistic scene, ${noise_label} ==="
    FPA_A="${FPA_A}" FPA_B="${FPA_B}" NOISE="${NOISE_VAL}" \
        sbatch scripts/gd_joint_band_test.slurm

    echo "=== uniform scene, ${noise_label} ==="
    FPA_A="${FPA_A}" FPA_B="${FPA_B}" UNIFORM=1 NOISE="${NOISE_VAL}" \
        sbatch scripts/gd_joint_band_test.slurm

    echo "=== barcode scene, ${noise_label} ==="
    FPA_A="${FPA_A}" FPA_B="${FPA_B}" BARCODE=1 NOISE="${NOISE_VAL}" \
        sbatch scripts/gd_joint_band_test.slurm
done

echo
echo "Submitted 6 jobs (3 scenes x 2 noise settings) for FPA${FPA_A}+FPA${FPA_B}," \
    "all three pipelines (native/rectified/undistorted) each."
echo "Check status with: squeue -u \$USER"
