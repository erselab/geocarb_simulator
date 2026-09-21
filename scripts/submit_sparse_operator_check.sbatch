#!/bin/bash
#SBATCH --partition=atmos
#SBATCH --account=ltmurray_lab
#SBATCH --job-name=mb_sparse
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=3:00:00
#SBATCH --output=slurm_out/mb_sparse_%x_%j.out
# One multi-band window solve. ARGS is passed straight to check_sparse_operator.py:
#   ARGS="--fpas 2 --rows 2:25-37 --anchor-mode nominal --free co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo" sbatch --export=ALL scripts/submit_multiband_window.sbatch   (NOT --export=ALL,ARGS=...: sbatch splits that at commas)
set -euo pipefail
cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
SHM_NAME="/dev/shm/geocarb_L_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
trap 'rm -f "${SHM_NAME}"*' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python -u scripts/check_sparse_operator.py ${ARGS} --anchor-workers "${SLURM_CPUS_PER_TASK}"
