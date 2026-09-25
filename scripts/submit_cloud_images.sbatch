#!/bin/bash
#SBATCH --partition=atmos
#SBATCH --account=ltmurray_lab
#SBATCH --job-name=cloud_img
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_out/cloud_img_%A_%a.out
# One array task per scenario: 0 = clear control, 1..9 = altitude x optical depth (cloud_scene.scenarios() order).
set -euo pipefail
cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NAMES=(clear low_thin low_moderate low_thick mid_thin mid_moderate mid_thick high_thin high_moderate high_thick)
/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python -u scripts/render_cloud_images.py --scenario "${NAMES[$SLURM_ARRAY_TASK_ID]}" --workers "${SLURM_CPUS_PER_TASK}"
