#!/bin/bash
#SBATCH --partition=atmos
#SBATCH --account=ltmurray_lab
#SBATCH --job-name=sb_geomcfg_sweep
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=3:00:00
#SBATCH --output=slurm_out/sb_geomcfg_%x_%A_%a.out
# Single-band (--fpa) whole-slit sweep using a geometry config (gd_export_geometry_config.py) for
# --task-id/--n-tasks tiling instead of build_window_tiles -- so this run's bins land at the SAME
# eta positions as the reference (multiband) run the config was exported from, for a bin-for-bin
# single-vs-multiband comparison (2026-09-23, user: "Make the plots first and then kick off the
# single band runs"). N_TASKS must equal the config's own tile count (33 for the FPA0+FPA2
# reference tiling). SOLVER left unset so gd_joint_block_retrieve.py's own aerosol-conditional
# default applies (single_scatter here, no aerosol -- 2026-09-23 policy, commit 82440d1).
#   FPA=2 FREE=co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo PRIOR=realistic \
#   GEOM=geometry_config_noaero_realistic.json N_TASKS=33 \
#   sbatch --export=ALL --array=0-32%12 scripts/submit_singleband_geomcfg_sweep.sbatch
set -euo pipefail
cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
: "${FPA:?set FPA=...}" "${FREE:?set FREE=...}" "${GEOM:?set GEOM=...}" "${N_TASKS:?set N_TASKS=...}"
SHM_NAME="/dev/shm/geocarb_L_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
trap 'rm -f "${SHM_NAME}"*' EXIT
find /dev/shm -maxdepth 1 -user "${USER}" -name 'geocarb_L_*' -mmin +30 2>/dev/null \
    | while read -r f; do fuser "$f" >/dev/null 2>&1 || rm -f "$f"; done
/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python -u scripts/gd_joint_block_retrieve.py \
    --fpa "${FPA}" --prior-fields "${PRIOR:-structural}" --vary-albedo \
    --anchor-density 4 --surface-positions anchor --frozen-atmosphere-positions anchor \
    --g-ratio 1 --hires-only --jacobian analytic --gn-verbose \
    --anchor-workers "${SLURM_CPUS_PER_TASK}" \
    --geometry-config "${GEOM}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" --n-tasks "${N_TASKS}" \
    --free "${FREE}"
