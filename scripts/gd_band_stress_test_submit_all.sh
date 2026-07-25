#!/bin/bash
# Submit the full along-slit stress-test matrix: all 4 FPAs x 3 scenes
# (realistic along-slit composition, uniform, barcode) x 2 noise settings
# (off, on at each band's default SNR) -- 6 sbatch calls, each an
# --array=0,1,2,3 job, so 24 SLURM tasks total. Each combination writes to
# its own uniquely-suffixed results/gd_band_stress_test_fpa<N>[_uniform|
# _barcode][_noise].pkl (gd_band_stress_test.py's own filename logic), so
# nothing collides regardless of submission order or how many run at once.
#
# Run:  bash scripts/gd_band_stress_test_submit_all.sh
#
# This only calls sbatch -- it does not wait for jobs to finish. Check
# progress with `squeue -u $USER`, and see gd_band_stress_test.slurm's own
# header comment for what NOISE/SNR/UNIFORM/BARCODE do individually if you
# want to submit a subset by hand instead of the full matrix.

set -euo pipefail
cd "$(dirname "$0")/.."

for NOISE_VAL in 0 1; do
    noise_label="noise=${NOISE_VAL}"

    echo "=== realistic scene, ${noise_label} ==="
    NOISE="${NOISE_VAL}" sbatch --array=0,1,2,3 scripts/gd_band_stress_test.slurm

    echo "=== uniform scene, ${noise_label} ==="
    UNIFORM=1 NOISE="${NOISE_VAL}" sbatch --array=0,1,2,3 scripts/gd_band_stress_test.slurm

    echo "=== barcode scene, ${noise_label} ==="
    BARCODE=1 NOISE="${NOISE_VAL}" sbatch --array=0,1,2,3 scripts/gd_band_stress_test.slurm
done

echo
echo "Submitted 6 array jobs (24 tasks total: 4 FPAs x 3 scenes x 2 noise settings)."
echo "Check status with: squeue -u \$USER"
