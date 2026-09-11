#!/bin/bash
# The experiment, GENUINELY representable this time. Prior runs missed
# that --resolution-matched-* only band-limits the TRUTH -- the retrieval
# still reconstructs a FROZEN albedo row from the window's 3 bin centers
# (~13km apart), which cannot represent even a 0.19km-band-limited albedo.
# Direct probe (2026-09-10): plain frozen 3-knot albedo -> resid(x0) 0.36
# at ANY anchor density; the SAME frozen row on the ad16 anchor grid
# (--surface-positions anchor) -> resid(x0) 0.004.
#
# So: truth band-limited to the ad16 anchor grid, AND every frozen row
# (albedo + atmosphere) reconstructed on that same ad16 anchor grid, AND
# the retrieval forward at --anchor-density 16. Everything matched. Any
# residual now is real retrieval/model error.
#
# rows 189-197, FPA2, hires-only, analytic, aerosol OFF.
#
#SBATCH --job-name=gd_smoke_rmt_ad16_ap
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_out/smoke_rmt_ad16_ap_%j.out

set -uo pipefail
module load git/2.50.1
cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

OUTDIR=scratch_work/smoke_rmt_ad16_ap
mkdir -p "${OUTDIR}"
COMMON="--fpa 2 --row-min 189 --row-max 197 --hires-only --jacobian analytic \
        --resolution-matched-anchor-density 16 --anchor-density 16 \
        --surface-positions anchor --frozen-atmosphere-positions anchor --anchor-workers 8"

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
run C2_structural_free    --free co2_ppm,p_surface_hpa,h2o_surface_vmr,albedo --vary-albedo --prior-fields structural
echo; echo ALL DONE
