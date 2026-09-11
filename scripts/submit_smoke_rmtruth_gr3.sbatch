#!/bin/bash
# Re-run of the aerosol-off smoke / imperfect-prior experiment
# (submit_smoke_aerosol_onoff_impprior.sbatch, job 1885853), now with the
# TRUTH matched to the retrieval's own state bin grid at g_ratio=3
# (--resolution-matched-g-ratio-bins 3), rendered by
# submit_rmtruth_gr3_build.sbatch. --anchor-density 16 on the retrieval
# side so the forward model resolves this now-exact-but-coarser truth
# without an anchor-level stepping-bias confound (per that flag's help).
#
# Same window (rows 189-197), FPA2, hires-only, analytic Jacobian, aerosol
# OFF throughout. --resolution-matched-g-ratio-bins implies --vary-albedo.
# The aerosol-ON case (config B in the original) cannot be run
# resolution-matched -- that truth builder has no aerosol fields -- so it
# is omitted here (its only purpose, confirming on != off, is already done).
#
#SBATCH --job-name=gd_smoke_rmtruth_gr3
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=05:00:00
#SBATCH --output=slurm_out/smoke_rmtruth_gr3_%j.out

set -uo pipefail
module load git/2.50.1

cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

OUTDIR=scratch_work/smoke_rmtruth_gr3
mkdir -p "${OUTDIR}"
COMMON="--fpa 2 --row-min 189 --row-max 197 --hires-only --jacobian analytic \
        --resolution-matched-g-ratio-bins 3 --anchor-density 16 --anchor-workers 8"

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

# (1) exact prior, reproduces baseline -- now with representable truth
run A1_exact_co2         --free co2_ppm                 --prior-fields exact
run A2_exact_co2p        --free co2_ppm,p_surface_hpa    --prior-fields exact

# (2) imperfect prior on the free parameters -- watch convergence
run C1_wronglevel_co2p   --free co2_ppm,p_surface_hpa \
                         --prior-fields co2_plus1pct_psurf_minus1pct
run C2_structural_allfree --free co2_ppm,p_surface_hpa,h2o_surface_vmr,albedo \
                         --vary-albedo --prior-fields structural

echo; echo "ALL DONE"
