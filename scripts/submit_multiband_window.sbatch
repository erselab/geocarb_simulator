#!/bin/bash
#SBATCH --partition=atmos
#SBATCH --account=ltmurray_lab
#SBATCH --job-name=mb_window
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_out/mb_window_%x_%j.out
# One multi-band window solve. ARGS is passed straight to gd_multiband_window.py:
#   ARGS="--fpas 2 --rows 2:25-37 --anchor-mode nominal --free co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo" sbatch --export=ALL scripts/submit_multiband_window.sbatch   (NOT --export=ALL,ARGS=...: sbatch splits that at commas)
set -euo pipefail
cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python -u scripts/gd_multiband_window.py ${ARGS} --anchor-workers "${SLURM_CPUS_PER_TASK}"
