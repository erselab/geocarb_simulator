#!/bin/bash
# Aerosol convergence bug (PROJECT_STATUS Sec.14): isolate a suspected
# CONFOUND in the bisection (2026-09-10, code inspection).
#
# Every "aerosol broken" cell in Sec.14's table runs with --vary-albedo
# (forced: an aerosol run needs a surface row) and albedo FROZEN on the
# G=3 bin grid, scored against a DENSE truth whose albedo carries full
# fine-scale texture (albedo_at include_fine=True). That is exactly the
# frozen-albedo sub-bin-texture representability gap Sec.1-3 already
# measured at ~10x error / resid ~0.1-0.2 -- closed only by
# --surface-positions anchor. Every "clean" cell (resid ~1e-5) instead
# runs CONSTANT albedo (no --vary-albedo). So the two are not a clean A/B
# for aerosol: they also differ in the albedo grid.
#
# This fills the missing cells:
#   D1  --vary-albedo, NO aerosol           -> does the albedo gap alone blow resid up?
#   D2  D1 + --surface-positions anchor      -> does anchor-res albedo recover ~1e-5?
#   D3  --vary-albedo + --aerosol (frozen exact) + --surface-positions anchor
#                                            -> pure aerosol mismatch, albedo gap removed
#   D4  --vary-albedo + --aerosol (frozen exact), shared albedo  -> repro Sec.14 (~0.229)
#   D5  D3 + free tau_aerosol                 -> does tau converge to 0.05 once isolated?
#
# rows 189-197, FPA2, hires-only, analytic, --prior-fields exact, ad1
# (matching Sec.14's own config for comparability).
#
#SBATCH --job-name=aero_confound_diag
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_out/aero_confound_diag_%j.out

set -uo pipefail
module load git/2.50.1

cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

OUTDIR=scratch_work/aero_confound_diag
mkdir -p "${OUTDIR}"
COMMON="--fpa 2 --row-min 189 --row-max 197 --hires-only --jacobian analytic \
        --prior-fields exact --anchor-workers 8"

run () {
  tag="$1"; shift
  echo; echo "==================== ${tag} ===================="
  echo "args: $*"
  ${PYTHON} scripts/gd_joint_block_retrieve.py ${COMMON} "$@" \
      --out "${OUTDIR}/${tag}.pkl" 2>&1 || echo "!!! ${tag} FAILED"
  ${PYTHON} -c "
import pickle; d=pickle.load(open('${OUTDIR}/${tag}.pkl','rb'))
w=next(iter(d['results'].values()))
print('  >>> resid_hires_rms = %.4e   free=%s' % (w['resid_hires_rms'], d.get('free')))
for n,p in w['hires']['params'].items():
    import numpy as np
    print('      %-16s hires mean=%+.4g' % (n, np.mean(p)))
" 2>&1 || echo "!!! ${tag} SUMMARY FAILED"
}

git rev-parse HEAD

run D1_valb_noaero                --free co2_ppm --vary-albedo
run D2_valb_noaero_anchorpos      --free co2_ppm --vary-albedo --surface-positions anchor
run D3_valb_aero_anchorpos        --free co2_ppm --vary-albedo --aerosol --surface-positions anchor
run D4_valb_aero_sharedpos        --free co2_ppm --vary-albedo --aerosol
run D5_valb_aero_anchorpos_freetau --free co2_ppm,tau_aerosol --vary-albedo --aerosol --surface-positions anchor

echo; echo "ALL DONE"
