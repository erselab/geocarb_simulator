#!/bin/bash
# The experiment (A1/A2/C1/C2), truth matched to the retrieval's ANCHOR
# grid at anchor_density=16 -- fine enough to band-limit als._albedo_
# fine_field (0.5km correlation) below its own scale, so a retrieval at
# --anchor-density 16 can represent the scene exactly. Any residual is
# then real retrieval/model error, not an unrepresentable-truth confound.
#
# Same window (rows 189-197), FPA2, hires-only, analytic, aerosol OFF.
# --resolution-matched-anchor-density implies --vary-albedo.
#
#SBATCH --job-name=gd_smoke_rmtruth_ad16
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_out/smoke_rmtruth_ad16_%j.out

set -uo pipefail
module load git/2.50.1
cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

OUTDIR=scratch_work/smoke_rmtruth_ad16
mkdir -p "${OUTDIR}"
COMMON="--fpa 2 --row-min 189 --row-max 197 --hires-only --jacobian analytic \
        --resolution-matched-anchor-density 16 --anchor-density 16 --anchor-workers 8"

run () {
  tag="$1"; shift
  echo; echo "==================== ${tag} ===================="
  echo "args: $*"
  ${PYTHON} scripts/gd_joint_block_retrieve.py ${COMMON} "$@" \
      --out "${OUTDIR}/${tag}.pkl" 2>&1 || echo "!!! ${tag} FAILED"
  ${PYTHON} - "${OUTDIR}/${tag}.pkl" <<'PY' || echo "!!! ${tag} SUMMARY FAILED"
import pickle, sys, numpy as np
d = pickle.load(open(sys.argv[1], "rb")); w = next(iter(d["results"].values()))
print("  >>> resid_hires_rms = %.4e   free=%s" % (w["resid_hires_rms"], d.get("free")))
for n, e in w["hires"]["params"].items():
    v = np.asarray(e["values"], float); p = np.asarray(e["prior"], float)
    print("      %-16s %s val=%.6g pri=%.6g" % (n, "F" if e["free"] else "x", v.mean(), p.mean()))
PY
}

git rev-parse HEAD
run A1_exact_co2          --free co2_ppm                 --prior-fields exact
run A2_exact_co2p         --free co2_ppm,p_surface_hpa    --prior-fields exact
run C1_wronglevel_co2p    --free co2_ppm,p_surface_hpa    --prior-fields co2_plus1pct_psurf_minus1pct
run C2_structural_allfree --free co2_ppm,p_surface_hpa,h2o_surface_vmr,albedo --vary-albedo --prior-fields structural
echo; echo "ALL DONE"
