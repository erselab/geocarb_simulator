#!/bin/bash
#SBATCH --partition=atmos
#SBATCH --account=ltmurray_lab
#SBATCH --job-name=fpa_img
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_out/fpa_img_%A_%a.out
# One array task per FPA (0-3): scripts/render_fpa_images.py --fpa $SLURM_ARRAY_TASK_ID
set -euo pipefail
cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python -u scripts/render_fpa_images.py --fpa "${SLURM_ARRAY_TASK_ID}" --workers "${SLURM_CPUS_PER_TASK}"
