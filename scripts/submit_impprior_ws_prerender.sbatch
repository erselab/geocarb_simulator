#!/bin/bash
# Pre-render the NO-AEROSOL dense TRUTH image the whole-slit imperfect-
# prior sweeps c1-c4 need (2026-09-10), so 58 array tasks don't race on a
# cache MISS. Runs task 0 (rows 0-8, small); also saves the raw
# (1024,1024) truth array for the --plot SNR panel.
# The WITH-aerosol truth (config c5) is split into its own job:
#   submit_impprior_ws_prerender_aero.sbatch  (aerosol render is ~10x slower)
#
#SBATCH --job-name=impprior_ws_prerender
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_out/impprior_ws_prerender_%j.out

set -euo pipefail
module load git/2.50.1
cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/geocarb_simulator/scripts:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

git rev-parse HEAD
COMMON="--fpa 2 --prior-fields structural --vary-albedo --anchor-density 16 \
        --surface-positions anchor --frozen-atmosphere-positions anchor \
        --g-ratio 0.5 --hires-only --jacobian analytic --anchor-workers 16 \
        --task-id 0 --n-tasks 58"

echo "=== prerender: no-aerosol truth (via c1 task 0) ==="
${PYTHON} scripts/gd_joint_block_retrieve.py ${COMMON} --free co2_ppm

echo "=== save raw truth array for plotting ==="
export TAGS="noaero"
${PYTHON} - <<'PY'
import numpy as np, pickle, geosat_geometry as gg, gert
from geocarb_gert import along_slit_scene as als, sample_geometries
from gd_joint_block_retrieve import band_basics
import gd_per_row_retrieve as gpr
from geocarb_gert.instrument import GEOCARB_BANDS
fpa=2
block=gg.geocarb_demo(verbose=False)["blocks"][0]
_,_,geo=sample_geometries(block,n=1,seed=0)[0]
absco=gert.ABSCOTable.load_all("/scratch/scrowel3_lab/gert/input/absco/absco.h5")
solar=gert.SolarSpectrum.load("/scratch/scrowel3_lab/gert/input/solar/solar.h5")
atm=als.atmosphere_at(0.0); gpr._G.update(dict(atm=atm,absco=absco,geo=geo,solar=solar))
snr=gpr.DEFAULT_SNR_BY_FPA[fpa]
import os
_want=set(os.environ.get("TAGS","noaero aero").split())
for wa,tag in [(False,"noaero"),(True,"aero")]:
    if tag not in _want: continue
    b=gpr._band_setup_cached(fpa,atm,absco,geo,solar,snr,5600,None,False,False,None,False,0,False,
        vary_albedo=True,use_cache=True,with_aerosol=wa)
    p=f"scratch_work/impprior_truth_{tag}_A.pkl"
    pickle.dump(np.asarray(b["A"],float),open(p,"wb"))
    print("saved",p, np.asarray(b["A"]).shape)
PY
echo "PRERENDER DONE"
