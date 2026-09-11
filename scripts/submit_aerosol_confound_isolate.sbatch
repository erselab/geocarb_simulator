#!/bin/bash
# Aerosol confound, part 2 (2026-09-10). Part 1 (aero_confound_diag)
# showed: --vary-albedo with a FROZEN albedo row already gives resid
# ~0.27 with NO aerosol at all (D1), and --surface-positions anchor does
# not help (D2). A direct field probe explains why: the fine-scale
# albedo texture is a 0.1km-sampled field with std 0.10, swinging
# 0.36..0.57 across the 27km window 189-197, while the 3 bin centers all
# land near 0.47 -- |dense - 3knot| rms 0.047 (~10% of the value). No
# retrieval grid short of ~0.1km resolves it.
#
# This isolates aerosol from that confound using --sub-bin-anomaly truth
# (gives a FREE albedo row the EXACT fine-texture oracle, riding on its
# bin values -- representability gap closed by construction):
#
#   E1  co2+albedo free, no oracle        -> does freeing G=3 albedo alone help?
#   E2  co2+albedo free + texture oracle   -> forward model clean? (expect ~1e-5)
#   E3  E2 + aerosol free (tau+height)     -> PURE aerosol test, albedo confound removed
#   E4  E2 + aerosol FROZEN at exact truth -> frozen-aerosol mismatch, isolated
#
# rows 189-197, FPA2, hires-only, analytic, --prior-fields exact, ad1.
#
#SBATCH --job-name=aero_confound_iso
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_out/aero_confound_iso_%j.out

set -uo pipefail
module load git/2.50.1
cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

OUTDIR=scratch_work/aero_confound_iso
mkdir -p "${OUTDIR}"
COMMON="--fpa 2 --row-min 189 --row-max 197 --hires-only --jacobian analytic \
        --prior-fields exact --anchor-workers 8 --vary-albedo"

run () {
  tag="$1"; shift
  echo; echo "==================== ${tag} ===================="
  echo "args: $*"
  ${PYTHON} scripts/gd_joint_block_retrieve.py ${COMMON} "$@" \
      --out "${OUTDIR}/${tag}.pkl" 2>&1 || echo "!!! ${tag} FAILED"
  ${PYTHON} - "${OUTDIR}/${tag}.pkl" <<'PY' || echo "!!! ${tag} SUMMARY FAILED"
import pickle, sys, numpy as np
d = pickle.load(open(sys.argv[1], "rb"))
w = next(iter(d["results"].values()))
print("  >>> resid_hires_rms = %.4e   free=%s" % (w["resid_hires_rms"], d.get("free")))
for n, p in w["hires"]["params"].items():
    print("      %-16s hires mean=%+.5g" % (n, np.mean(np.asarray(p))))
PY
}

git rev-parse HEAD

run E1_co2albedo_free              --free co2_ppm,albedo
run E2_co2albedo_free_oracle       --free co2_ppm,albedo --sub-bin-anomaly truth
run E3_pure_aerosol_oracle         --free co2_ppm,albedo,tau_aerosol,height_aerosol --aerosol --sub-bin-anomaly truth
run E4_frozen_aerosol_oracle       --free co2_ppm,albedo --aerosol --sub-bin-anomaly truth

echo; echo "ALL DONE"
