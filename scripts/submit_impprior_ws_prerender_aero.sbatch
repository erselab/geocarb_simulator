#!/bin/bash
# Pre-render the WITH-AEROSOL dense TRUTH image the whole-slit imperfect-
# prior sweep c5 needs (2026-09-10), split out of
# submit_impprior_ws_prerender.sbatch because the aerosol render path is
# far slower and timed out job 1886845 at 3h. Renders + caches the truth
# and saves the raw (1024,1024) array for the --plot SNR panel. No solve
# (the c5 array does the real retrieval).
#
#SBATCH --job-name=impprior_ws_prerender_aero
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_out/impprior_ws_prerender_aero_%j.out

set -euo pipefail
module load git/2.50.1
cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/geocarb_simulator/scripts:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

git rev-parse HEAD

echo "=== prerender: aerosol truth render + cache + raw-array save (both tags) ==="
${PYTHON} - <<'PY'
import numpy as np, pickle, geosat_geometry as gg, gert
from geocarb_gert import along_slit_scene as als, sample_geometries
import gd_per_row_retrieve as gpr
fpa=2
block=gg.geocarb_demo(verbose=False)["blocks"][0]
_,_,geo=sample_geometries(block,n=1,seed=0)[0]
absco=gert.ABSCOTable.load_all("/scratch/scrowel3_lab/gert/input/absco/absco.h5")
solar=gert.SolarSpectrum.load("/scratch/scrowel3_lab/gert/input/solar/solar.h5")
atm=als.atmosphere_at(0.0); gpr._G.update(dict(atm=atm,absco=absco,geo=geo,solar=solar))
snr=gpr.DEFAULT_SNR_BY_FPA[fpa]
for wa,tag in [(False,"noaero"),(True,"aero")]:
    b=gpr._band_setup_cached(fpa,atm,absco,geo,solar,snr,5600,None,False,False,None,False,0,False,
        vary_albedo=True,use_cache=True,with_aerosol=wa)
    p=f"scratch_work/impprior_truth_{tag}_A.pkl"
    pickle.dump(np.asarray(b["A"],float),open(p,"wb"))
    print("saved",p, np.asarray(b["A"]).shape)
PY
echo "PRERENDER AERO DONE"
