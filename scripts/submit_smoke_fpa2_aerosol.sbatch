#!/bin/bash
# Aerosol-convergence smoke test (2026-09-09): re-run of PROJECT_STATUS
# Sec.11's own single-window aerosol smoke test now that Sec.12
# consolidated the three hand-duplicated "build a spectrum from state
# parameters" functions into geocarb_gert.spectrum -- the exact
# root-cause refactor Sec.11's closing flag called for. Sec.11 left this
# config at resid_hires_rms ~0.101 (should be ~1e-5 for a fully
# representable config) with "at least one more issue remains, not yet
# found"; this checks whether the consolidation closed that gap.
#
# Single low-keystone window (rows 189-197), the same fast-subset window
# Sec.10's t_offset_k smoke used (which reached ~1e-5). Fully
# representable: --prior-fields exact freezes every nuisance field at
# local truth, so there is no representability gap by construction --
# any residual is a solver/forward-model problem, not a scene problem.
#
#SBATCH --job-name=gd_smoke_fpa2_aerosol
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=slurm_out/smoke_fpa2_aerosol_%j.out

set -euo pipefail
module load git/2.50.1

cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${PYTHON}" scripts/gd_joint_block_retrieve.py \
    --fpa 2 --free co2_ppm,p_surface_hpa,tau_aerosol,height_aerosol \
    --vary-albedo --aerosol --prior-fields exact \
    --row-min 189 --row-max 197 \
    --hires-only --jacobian analytic --anchor-workers "${SLURM_CPUS_PER_TASK}" \
    --out scratch_work/smoke_aerosol_fpa2.pkl
