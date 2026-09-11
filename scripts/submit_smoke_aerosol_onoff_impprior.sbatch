#!/bin/bash
# Aerosol on/off equivalence + imperfect-prior convergence smoke test
# (2026-09-10). Two questions:
#
#  (1) Does the current retrieval code (HEAD, with the Sec.14 --aerosol
#      opt-in mitigation) reproduce the pre-aerosol baseline when aerosol
#      is OFF?  Configs A1/A2 vs. PROJECT_STATUS Sec.14's table
#      (co2 -> resid ~2.0e-5 / err 0; co2+p_surface -> resid ~1.7e-5).
#      Config B re-runs the known-bad aerosol-ON case to confirm on != off.
#
#  (2) The "free" experiments (C1/C2) use a DELIBERATELY IMPERFECT prior
#      for the free parameters (--prior-fields co2_plus1pct_psurf_minus1pct
#      = right structure, wrong absolute level; and --prior-fields
#      structural = background/topography only, no localized/synoptic
#      content), so prior-vs-posterior error shows the solver actually
#      converging rather than sitting on truth.
#
# Single low-keystone window (rows 189-197), FPA2, hires-only, analytic
# Jacobian -- the same fast-subset window Sec.10/11/14 used.
#
#SBATCH --job-name=gd_smoke_aero_onoff
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_out/smoke_aero_onoff_%j.out

set -uo pipefail
module load git/2.50.1

cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

OUTDIR=scratch_work/smoke_aero_onoff
mkdir -p "${OUTDIR}"
COMMON="--fpa 2 --row-min 189 --row-max 197 --hires-only --jacobian analytic --anchor-workers 8"

run () {   # $1 = tag, rest = extra args
  tag="$1"; shift
  echo; echo "==================== ${tag} ===================="
  echo "args: $*"
  ${PYTHON} scripts/gd_joint_block_retrieve.py ${COMMON} "$@" \
      --out "${OUTDIR}/${tag}.pkl" 2>&1 || echo "!!! ${tag} FAILED"
  echo "-------------------- ${tag}: plot vs truth --------------------"
  ${PYTHON} scripts/gd_joint_block_retrieve.py --plot "${OUTDIR}/${tag}.pkl" 2>&1 \
      || echo "!!! ${tag} PLOT FAILED"
}

git rev-parse HEAD

# --- (1) aerosol OFF, exact prior: reproduce pre-aerosol baseline ---
run A1_exact_co2_aeroff        --free co2_ppm                       --prior-fields exact
run A2_exact_co2p_aeroff       --free co2_ppm,p_surface_hpa         --prior-fields exact

# --- aerosol ON (known-bad, confirms on != off) ---
run B_exact_co2p_aero_on       --free co2_ppm,p_surface_hpa,tau_aerosol,height_aerosol \
                               --vary-albedo --aerosol --prior-fields exact

# --- (2) free experiments with an IMPERFECT prior, aerosol OFF ---
run C1_wronglevel_co2p_aeroff  --free co2_ppm,p_surface_hpa \
                               --prior-fields co2_plus1pct_psurf_minus1pct
run C2_structural_allfree_aeroff --free co2_ppm,p_surface_hpa,h2o_surface_vmr,albedo \
                               --vary-albedo --prior-fields structural

echo; echo "ALL DONE"
