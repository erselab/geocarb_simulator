#!/bin/bash
# One-off diagnostic (2026-09-13): reproduce the "p_levels must be strictly
# increasing (TOA -> surface)" crash that hit 7/58 windows of the c5
# (aerosol) impprior_ws_g1ad4ov2 sweep, with exceptions re-raised (not
# swallowed by _worker) so we get a full traceback instead of just the
# summary message merge_parts stored. Row 79-91 (task-id 9) is one of the
# 7 failures. Single window, in-process (no Pool) -- ~46min at 1 CPU per
# the original run's own t_hires.
#SBATCH --job-name=debug_c5_crash
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_out/debug_c5_crash_%j.out

set -uo pipefail
module load git/2.50.1
cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

${PYTHON} - <<'PYEOF'
import sys, traceback
sys.argv = [
    "gd_joint_block_retrieve.py",
    "--fpa", "2", "--prior-fields", "structural", "--vary-albedo",
    "--anchor-density", "4", "--surface-positions", "anchor", "--frozen-atmosphere-positions", "anchor",
    "--g-ratio", "1", "--overlap", "2", "--hires-only", "--jacobian", "analytic", "--gn-verbose",
    "--anchor-workers", "4",
    "--task-id", "9", "--n-tasks", "58",
    "--free", "co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo,tau_aerosol,height_aerosol", "--aerosol",
]
import scripts.gd_joint_block_retrieve as m

_orig_solve = m._solve_window
def _reraise(tile):
    row_lo, row_hi = tile
    return _orig_solve(row_lo, row_hi)
m._worker = _reraise

try:
    m.main()
except Exception:
    traceback.print_exc()
    sys.exit(1)
PYEOF
