#!/bin/bash
# Whole-slit imperfect-prior (structural) sweep, one SLURM array task per
# window (2026-09-10, user). Dense (continuous) truth, --vary-albedo, all
# frozen rows on the ad16 anchor grid so the scene is representable
# (docs/PROJECT_STATUS.md Sec.15), g_ratio=0.5 (ALGORITHM_ROADMAP
# production default).
#
# Select the free-parameter set with:  sbatch --export=ALL,CONF=c1 ...
#   c1  co2_ppm
#   c2  co2_ppm,p_surface_hpa
#   c3  co2_ppm,p_surface_hpa,albedo
#   c4  co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo   ("everything", no aerosol)
#   c5  c4 + tau_aerosol,height_aerosol  --aerosol                ("everything" + aerosol retrieved)
#
#SBATCH --job-name=impprior_ws
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --array=0-57%12
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_out/impprior_ws_%x_%A_%a.out
# 2026-09-11: --anchor-workers used to be a no-op (every --task-id task ran
# inside a daemon Pool worker, silently forcing anchor-level parallelism to
# 1 -- see gd_joint_block_retrieve.py's fix and docs/PROJECT_STATUS.md
# Sec.16). Now that it actually works, --cpus-per-task=8/--mem=16G (sized
# for the old accidentally-serial behavior) OOM'd on every resubmitted
# c3/c4/c5 task within ~1 minute -- linearize() creates a fresh
# multiprocessing.Pool per free-parameter-column batch on EVERY GN
# iteration (up to 15), and memory climbs with each iteration rather than
# holding at one fixed footprint (measured: a moderate c3 window still
# climbing past ~45GB attributable after 30 min, no plateau). Cut to 4
# workers (still a real speedup over the old serialized baseline).
# Validated at 64G/4-workers on a moderate c3 window (19 rows, 3 free
# rows): completed in 2886s, peak ~41.5GB attributable -- but c4 (5 free
# rows) / c5 (7, +aerosol) and wider windows (up to 45 rows vs this
# window's 19) scale further on both axes, so --mem is set to 128G here
# for headroom rather than the validated-but-tight 64G. Root fix (avoid
# re-forking a growing Pool every GN iteration, and stop passing large
# per-column Jacobian fields through pool.map's pickling instead of the
# fork-inherited-global pattern anchor_spectra_and_derivs already uses)
# is still open -- this is an operational mitigation, not that fix.

set -euo pipefail
module load git/2.50.1
cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

: "${CONF:?set CONF=c1|c2|c3|c4|c5 via --export=ALL,CONF=...}"
case "${CONF}" in
  c1) FREE="co2_ppm"; AERO="" ;;
  c2) FREE="co2_ppm,p_surface_hpa"; AERO="" ;;
  c3) FREE="co2_ppm,p_surface_hpa,albedo"; AERO="" ;;
  c4) FREE="co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo"; AERO="" ;;
  c5) FREE="co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo,tau_aerosol,height_aerosol"; AERO="--aerosol" ;;
  *) echo "unknown CONF=${CONF}"; exit 1 ;;
esac

echo "CONF=${CONF}  FREE=${FREE}  ${AERO}  task ${SLURM_ARRAY_TASK_ID}/58"
git rev-parse HEAD

# 2026-09-12 (user's own resubmitted sweep: several c5 tasks died with
# "No space left on device"/SIGBUS on /dev/shm, traced to LinearizePool's
# shared-memory buffer -- see jacobians.py's own comment at its creation
# site): a cgroup OOM-kill is a SIGKILL, which can take out
# multiprocessing.resource_tracker's watchdog in the same sweep, so its
# built-in orphan-cleanup can't be relied on either -- a leaked segment
# (tens to a couple hundred GB) then sits on that node forever, starving
# every later task that lands there. Two layers, since neither alone
# covers every way this task can die:
#   1. This task's OWN segment, by the deterministic name jacobians.py
#      constructs from these same two SLURM ids -- removed on ANY exit
#      from this script (success, `set -e` after the python line's own
#      death, or an explicit early `exit`), as long as THIS shell is still
#      alive to run it (true for a plain OOM-kill of the python child,
#      confirmed directly: the "Killed" message comes from this shell
#      catching the child's death, so the trap below gets to run after).
#   2. ANY other stale segment of ours on THIS node, so a case where even
#      this shell doesn't survive (the whole cgroup killed together) still
#      gets cleaned up -- by the NEXT task lucky enough to land here,
#      rather than never. `-mmin +30`: this sweep is a best-effort, so it
#      only ever removes something clearly abandoned (older than any
#      single L() batch takes to produce), and `fuser` (not just the
#      window) confirms nothing still has it open before removing it.
SHM_NAME="/dev/shm/geocarb_L_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
trap 'rm -f "${SHM_NAME}"' EXIT
find /dev/shm -maxdepth 1 -user "${USER}" -name 'geocarb_L_*' -mmin +30 2>/dev/null \
    | while read -r f; do fuser "$f" >/dev/null 2>&1 || rm -f "$f"; done

${PYTHON} scripts/gd_joint_block_retrieve.py \
    --fpa 2 --prior-fields structural --vary-albedo \
    --anchor-density 16 --surface-positions anchor --frozen-atmosphere-positions anchor \
    --g-ratio 0.5 --hires-only --jacobian analytic --gn-verbose \
    --anchor-workers "${SLURM_CPUS_PER_TASK}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" --n-tasks 58 \
    --free "${FREE}" ${AERO}
