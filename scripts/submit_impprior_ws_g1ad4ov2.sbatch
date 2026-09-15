#!/bin/bash
# Whole-slit imperfect-prior (structural) sweep, c1-c5, at g_ratio=1/
# anchor_density=4/overlap=2 (2026-09-13, user) -- a coarser, cheaper
# resolution than submit_impprior_ws.sbatch's own production g_ratio=0.5/
# anchor_density=16, run with this session's two other validated fixes
# already the default: truth generated via the retrieval's own forward
# model (Sec.20, no --legacy-truth), and every row NOT in --free held
# exactly at the true value (Sec.19's per-row fields mixing). --overlap 2
# lets a post-hoc analysis drop each window's own boundary bins (the ones
# shared with a neighbor) and score only the interior ones -- see
# docs/PROJECT_STATUS.md's own overlap/core-bin diagnostic (2026-09-13):
# excluding boundary bins cut the worst-case co2 error from ~0.24 ppm
# (overlap=0) to ~0.03 ppm at this same g_ratio=1/anchor_density=4.
#
# Select the free-parameter set with:  sbatch --export=ALL,CONF=c1 ...
#   c1  co2_ppm
#   c2  co2_ppm,p_surface_hpa
#   c3  co2_ppm,p_surface_hpa,albedo
#   c4  co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo   ("everything", no aerosol)
#   c5  c4 + tau_aerosol,height_aerosol  --aerosol                ("everything" + aerosol retrieved)
#   c5b c4 + tau_aerosol (NOT height_aerosol)  --aerosol           (diagnostic, 2026-09-15 --
#       see below)
#
# c5b (2026-09-15, user: isolate whether height_aerosol's non-smooth
# Jacobian specifically drives c5's slow convergence, vs. just "2 more
# free rows in general"): c5's own GN logs showed a stark, near-uniform
# split from c4 -- c4 converges in 3-5 iterations on EVERY one of 58
# windows, c5 needed up to 14 and 35/58 (60%) hit the max_iter=15 cap,
# while the per-iteration wall-clock cost only rose ~1.2-1.3x (fully
# explained by height_aerosol's own already-documented extra 2 RT
# calls/anchor, docs/PROJECT_STATUS.md Sec.11) -- i.e. the slowdown is
# almost entirely an ITERATION-COUNT problem, not a per-iteration-cost
# one. height_aerosol's Jacobian is a genuine RT-level finite
# difference of a discontinuous function (`tau_abv`'s hard
# `above_mask = atm.p_layers < height_aerosol`, gert's forward_model.py
# -- see jacobians.py's own height_aerosol_dI_dparam docstring); a
# locally-computed derivative of a discontinuous function is a poor
# local model for Gauss-Newton's quadratic approximation, producing
# exactly the "lam pinned, J flat for 10+ iterations" plateau signature
# seen directly in c5's own logs (e.g. task 56/57). c5b keeps aerosol
# physics and tau_aerosol (validated smooth, Sec.11) free, but FREEZES
# height_aerosol at truth (Sec.19's generic frozen-row-holds-truth
# logic) -- if iteration counts drop back near c4's 3-5 range, that
# confirms height_aerosol's non-smoothness specifically (not just
# aerosol/dimensionality in general) as the dominant driver.
#
#SBATCH --job-name=impprior_ws_g1ad4ov2
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --array=0-57%20
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
# 2026-09-13 (user: "bump up the default time to cover the longest runs
# so we can avoid rerunning"): 2h was too tight -- the c4/c5 resubmission
# at this same g_ratio=1/anchor_density=4/overlap=2 config hit TIMEOUT on
# 3/58 (c4) and 13/58 (c5, aerosol) windows at 2h, mostly the widest
# ones. 8h is a generous margin above that, not a measured ceiling --
# raise further if a future run still times out.
#SBATCH --output=slurm_out/impprior_ws_g1ad4ov2_%x_%A_%a.out

set -euo pipefail
module load git/2.50.1
cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

: "${CONF:?set CONF=c1|c2|c3|c4|c5|c5b via --export=ALL,CONF=...}"
case "${CONF}" in
  c1) FREE="co2_ppm"; AERO="" ;;
  c2) FREE="co2_ppm,p_surface_hpa"; AERO="" ;;
  c3) FREE="co2_ppm,p_surface_hpa,albedo"; AERO="" ;;
  c4) FREE="co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo"; AERO="" ;;
  c5) FREE="co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo,tau_aerosol,height_aerosol"; AERO="--aerosol" ;;
  c5b) FREE="co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo,tau_aerosol"; AERO="--aerosol" ;;
  *) echo "unknown CONF=${CONF}"; exit 1 ;;
esac

echo "CONF=${CONF}  FREE=${FREE}  ${AERO}  task ${SLURM_ARRAY_TASK_ID}/58"
git rev-parse HEAD

# Same shm safety net as submit_impprior_ws.sbatch -- see that script's
# own comment for the full reasoning (SIGKILL bypasses LinearizePool's
# own cleanup; this covers both "this task's own segment" and "any
# stale segment of ours left on this node").
SHM_NAME="/dev/shm/geocarb_L_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
trap 'rm -f "${SHM_NAME}"' EXIT
find /dev/shm -maxdepth 1 -user "${USER}" -name 'geocarb_L_*' -mmin +30 2>/dev/null \
    | while read -r f; do fuser "$f" >/dev/null 2>&1 || rm -f "$f"; done

${PYTHON} scripts/gd_joint_block_retrieve.py \
    --fpa 2 --prior-fields structural --vary-albedo \
    --anchor-density 4 --surface-positions anchor --frozen-atmosphere-positions anchor \
    --g-ratio 1 --overlap 2 --hires-only --jacobian analytic --gn-verbose \
    --anchor-workers "${SLURM_CPUS_PER_TASK}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" --n-tasks 58 \
    --free "${FREE}" ${AERO}
