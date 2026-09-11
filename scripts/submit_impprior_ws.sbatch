#!/bin/bash
# Whole-slit imperfect-prior (structural) sweep, one SLURM array task per
# window (2026-09-10, user). Dense (continuous) truth, --vary-albedo, all
# frozen rows on the ad16 anchor grid so the scene is representable
# (docs/PROJECT_STATUS.md Sec.15), g_ratio=0.5 (ALGORITHM_ROADMAP
# production default).
#
# Select the free-parameter set with:  sbatch --export=ALL,CONF=c1 ...
#   c1  co2_ppm
#   c2  co2_ppm,p_surface_hpa
#   c3  co2_ppm,p_surface_hpa,albedo
#   c4  co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo   ("everything", no aerosol)
#   c5  c4 + tau_aerosol,height_aerosol  --aerosol                ("everything" + aerosol retrieved)
#
#SBATCH --job-name=impprior_ws
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --array=0-57%12
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_out/impprior_ws_%x_%A_%a.out

set -euo pipefail
module load git/2.50.1
cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

: "${CONF:?set CONF=c1|c2|c3|c4|c5 via --export=ALL,CONF=...}"
case "${CONF}" in
  c1) FREE="co2_ppm"; AERO="" ;;
  c2) FREE="co2_ppm,p_surface_hpa"; AERO="" ;;
  c3) FREE="co2_ppm,p_surface_hpa,albedo"; AERO="" ;;
  c4) FREE="co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo"; AERO="" ;;
  c5) FREE="co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo,tau_aerosol,height_aerosol"; AERO="--aerosol" ;;
  *) echo "unknown CONF=${CONF}"; exit 1 ;;
esac

echo "CONF=${CONF}  FREE=${FREE}  ${AERO}  task ${SLURM_ARRAY_TASK_ID}/58"
git rev-parse HEAD

${PYTHON} scripts/gd_joint_block_retrieve.py \
    --fpa 2 --prior-fields structural --vary-albedo \
    --anchor-density 16 --surface-positions anchor --frozen-atmosphere-positions anchor \
    --g-ratio 0.5 --hires-only --jacobian analytic \
    --anchor-workers "${SLURM_CPUS_PER_TASK}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" --n-tasks 58 \
    --free "${FREE}" ${AERO}
