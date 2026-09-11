#!/bin/bash
# Representable-truth baseline, done correctly this time (2026-09-10).
#
# BUG FOUND: gd_joint_block_retrieve.py's --resolution-matched-* renders a
# spatially-VARYING-albedo truth (vary_albedo=True hardcoded at the build
# call) but does NOT force args.vary_albedo, so without an explicit
# --vary-albedo the retrieval carries NO albedo row and uses a single
# constant scalar -> ~0.37 residual vs the varying truth, co2 driven to
# 400. Its own --help says "Implies --vary-albedo"; it doesn't. Every
# A1/A2/C1 run before this was invalid for that reason (C2 passed
# --vary-albedo explicitly and worked: resid 0.020 with anchor positions).
#
# Part R: rm-ad16 truth + every frozen row on the ad16 anchor grid +
#         --vary-albedo on ALL configs. Expect ~5e-3 (grid-match floor).
# Part G: aerosol re-test on a clean platform -- continuous truth,
#         --anchor-density 16, free albedo + exact fine-texture oracle
#         (--sub-bin-anomaly truth), which the direct probe showed makes
#         the --vary-albedo scene representable (resid(x0) 0.004 at ad16).
#         --aerosol is blocked with --resolution-matched-*, so this uses
#         the continuous truth + oracle instead.
#
#SBATCH --job-name=gd_smoke_repr_final
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_out/smoke_repr_final_%j.out

set -uo pipefail
module load git/2.50.1
cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

OUTDIR=scratch_work/smoke_repr_final
mkdir -p "${OUTDIR}"
BASE="--fpa 2 --row-min 189 --row-max 197 --hires-only --jacobian analytic \
      --anchor-density 16 --anchor-workers 8 --vary-albedo"
RM="${BASE} --resolution-matched-anchor-density 16 \
    --surface-positions anchor --frozen-atmosphere-positions anchor"

run () {
  tag="$1"; shift
  echo; echo "==================== ${tag} ===================="
  echo "args: $*"
  ${PYTHON} scripts/gd_joint_block_retrieve.py "$@" --out "${OUTDIR}/${tag}.pkl" 2>&1 \
      || echo "!!! ${tag} FAILED"
  ${PYTHON} - "${OUTDIR}/${tag}.pkl" <<'PY' || echo "!!! ${tag} SUMMARY FAILED"
import pickle, sys, numpy as np
d = pickle.load(open(sys.argv[1], "rb")); w = next(iter(d["results"].values()))
print("  >>> resid_hires_rms = %.4e   free=%s" % (w["resid_hires_rms"], d.get("free")))
for n, e in w["hires"]["params"].items():
    v = np.asarray(e["values"], float); p = np.asarray(e["prior"], float)
    print("      %-16s %s n=%3d val=%.6g pri=%.6g" % (n, "F" if e["free"] else "x", len(e["positions"]), v.mean(), p.mean()))
PY
}
git rev-parse HEAD

# --- Part R: representable (rm-ad16) truth, all frozen rows on anchor grid ---
run R_A1_exact_co2       ${RM} --free co2_ppm                --prior-fields exact
run R_A2_exact_co2p      ${RM} --free co2_ppm,p_surface_hpa   --prior-fields exact
run R_C1_wronglevel      ${RM} --free co2_ppm,p_surface_hpa   --prior-fields co2_plus1pct_psurf_minus1pct

# --- Part G: aerosol re-test on continuous truth + fine-texture oracle ---
run G_albedo_oracle_noaero  ${BASE} --free co2_ppm,albedo               --prior-fields exact --sub-bin-anomaly truth
run G_aero_frozen_oracle    ${BASE} --free co2_ppm,albedo --aerosol     --prior-fields exact --sub-bin-anomaly truth
run G_aero_free_oracle      ${BASE} --free co2_ppm,albedo,tau_aerosol,height_aerosol --aerosol --prior-fields exact --sub-bin-anomaly truth

echo; echo ALL DONE
