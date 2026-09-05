#!/bin/bash
# Submit the full multi-band battery: 3 scenes (realistic, uniform,
# barcode) x 2 noise settings (off, on) = 6 sbatch calls, all three
# pipelines (native/rectified/undistorted) within each -- for a fixed,
# ordered set of >=1 GeoCarb bands (default FPA0+FPA2; override with a
# positional arg or the FPAS env var, comma-separated, e.g. "0" for a
# single-band run or "0,1,2,3" for a joint one -- see gd_test.
# slurm's header for the full env-var list).
#
# Each combination writes its own uniquely-suffixed
# results/gd_joint_<fpas_tag>[_uniform|_barcode][_noise].pkl
# (gd_test.py's own filename logic, fpas_tag = "fpa0_fpa2",
# "fpa0_fpa1_fpa2_fpa3", etc.), so nothing collides regardless of
# submission order, band count, or how many run at once.
#
# Run:  bash scripts/gd_test_submit_all.sh
#   or: bash scripts/gd_test_submit_all.sh 0,1,2,3
# Or a subset directly, e.g. realistic + noise only:
#   NOISE=1 FPAS=0,2 sbatch scripts/gd_test.slurm
#
# This only calls sbatch -- it does not wait for jobs to finish. Check
# progress with `squeue -u $USER`.

set -euo pipefail
cd "$(dirname "$0")/.."

FPAS="${1:-${FPAS:-0,2}}"

for NOISE_VAL in 0 1; do
    noise_label="noise=${NOISE_VAL}"

    echo "=== realistic scene, ${noise_label} ==="
    FPAS="${FPAS}" NOISE="${NOISE_VAL}" \
        sbatch scripts/gd_test.slurm

    echo "=== uniform scene, ${noise_label} ==="
    FPAS="${FPAS}" UNIFORM=1 NOISE="${NOISE_VAL}" \
        sbatch scripts/gd_test.slurm

    echo "=== barcode scene, ${noise_label} ==="
    FPAS="${FPAS}" BARCODE=1 NOISE="${NOISE_VAL}" \
        sbatch scripts/gd_test.slurm
done

echo
echo "Submitted 6 jobs (3 scenes x 2 noise settings) for FPAs ${FPAS}," \
    "pipelines: ${PIPELINES:-native,rectified,undistorted} (each)."
echo "Check status with: squeue -u \$USER"
