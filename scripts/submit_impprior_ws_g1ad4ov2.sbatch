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
#   c5  c4 + amplitude_aerosol,height_aerosol,thickness_aerosol  --aerosol
#       ("everything" + aerosol retrieved)
#   c5b c4 + amplitude_aerosol,thickness_aerosol (NOT height_aerosol)  --aerosol
#       (diagnostic, 2026-09-15 -- see below)
#   c5x same free set as c5, but --solver xrtm (Phase 6 of the XRTM
#       integration plan, 2026-09-15 -- see below)
#
# c5x: real A/B test of the XRTM solver
# (geocarb_gert/spectrum.py::_build_solver, method='two_stream') against
# c5's own SingleScatterSolver-based result -- same free rows, same
# resolution, only the RT solver differs. Confirms whether XRTM's
# genuinely smooth analytic height_aerosol/thickness_aerosol Jacobians
# (validated via scripts/gd_xrtm_aerosol_jacobian_validate.py) actually
# help GN convergence speed, or whether (as c5b's own diagnostic
# suggests) the bottleneck lies elsewhere. method='two_stream' (not
# eig_add) after a direct benchmark found it ~32x cheaper per anchor
# (3.27s vs eig_add(n_streams=2)'s 105.28s at this band's own ~12353
# hi-res wavenumber points) -- eig_add even at its own minimum
# n_streams=2 is still the full discrete-ordinates eigenvalue machinery,
# not a cheap "2-stream" mode; two_stream is a separate, purpose-built
# implementation (the same one gert.rt_solver.LSISolver uses as its own
# "fast" solver).
#
# 2026-09-15 (user: "introduce the Gaussian parameters into the state
# vector instead of a tau_aerosol", matching how they run `gert` in other
# applications): `tau_aerosol` retired as a directly-retrievable row.
# `amplitude_aerosol`/`height_aerosol`/`thickness_aerosol` (the Gaussian
# profile's peak density, center, and width) take its place; column AOD
# is now `amplitude_aerosol * thickness_aerosol * sqrt(2*pi)`, a derived
# diagnostic rather than something GN solves for directly -- see
# `along_slit_scene.amplitude_aerosol`'s own docstring. The c5/c5b runs
# already merged under the OLD `tau_aerosol` row name (this file's own
# git history / docs/PROJECT_STATUS.md Sec.11-22) stay valid as a record
# of what was true then; only a FRESH submission under this updated CONF
# picks up the new parameterization.
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
#SBATCH --mem=180G
#SBATCH --time=8:00:00
# 2026-09-13 (user: "bump up the default time to cover the longest runs
# so we can avoid rerunning"): 2h was too tight -- the c4/c5 resubmission
# at this same g_ratio=1/anchor_density=4/overlap=2 config hit TIMEOUT on
# 3/58 (c4) and 13/58 (c5, aerosol) windows at 2h, mostly the widest
# ones. 8h is a generous margin above that, not a measured ceiling --
# raise further if a future run still times out.
#
# 2026-09-16 (user, after the c5x memory-cluster investigation): --mem
# raised 64G -> 180G -- tasks 36-56 (a contiguous, real memory-heavy
# stretch of this scene) measured MaxRSS 78-107GB against the old 64G
# request, causing 10 outright cgroup OOM-kills (PoolHangError -- a
# SIGKILL'd worker leaves no exception to report, see that class's own
# docstring) and pushing 11 more into TIMEOUT (surviving under memory
# pressure but thrashing too slowly to converge in 8h) -- 21/58 windows,
# one contiguous failure block. `atmos` nodes have ~493GB RAM/64 CPUs
# (sinfo) against this script's own --cpus-per-task=4, so 180G leaves
# real margin above the worst observed case while still packing 2
# tasks/node. --time briefly dropped to 6h in the same change, then put
# back to 8h (user: "I thought the wall clock was set to 3 hours" --
# it was already 8h, not 6h or 3h; kept at 8h, unchanged from 2026-09-13's
# own value above).
#SBATCH --output=slurm_out/impprior_ws_g1ad4ov2_%x_%A_%a.out

set -euo pipefail
module load git/2.50.1
cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

: "${CONF:?set CONF=c1|c2|c3|c4|c5|c5b|c5x|c5y|o5x via --export=ALL,CONF=...}"
SOLVER="single_scatter"
FPA="2"
case "${CONF}" in
  c1) FREE="co2_ppm"; AERO="" ;;
  c2) FREE="co2_ppm,p_surface_hpa"; AERO="" ;;
  c3) FREE="co2_ppm,p_surface_hpa,albedo"; AERO="" ;;
  c4) FREE="co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo"; AERO="" ;;
  c5) FREE="co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo,amplitude_aerosol,height_aerosol,thickness_aerosol"; AERO="--aerosol" ;;
  c5b) FREE="co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo,amplitude_aerosol,thickness_aerosol"; AERO="--aerosol" ;;
  c5x) FREE="co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo,amplitude_aerosol,height_aerosol,thickness_aerosol"; AERO="--aerosol"; SOLVER="xrtm" ;;
  # 2026-09-17: c5/c5x's amplitude_aerosol+thickness_aerosol combination is
  # near-exactly degenerate (they enter gert's forward model ONLY through
  # their product, tau_aerosol = amplitude*thickness*sqrt(2*pi) -- posterior
  # covariance condition numbers 1e17-1e25, independent of solver -- see
  # docs/PROJECT_STATUS.md Sec.27). c5y drops thickness_aerosol (fixed at
  # its own along_slit_scene.py default) and keeps height_aerosol free --
  # NOT part of that exact algebraic degeneracy, and requires XRTM
  # specifically (single_scatter's own height_aerosol Jacobian is
  # genuinely broken, K_ssa_lay==0 -- Sec.30) -- this is the validated,
  # well-conditioned design from Sec.30's own head-to-head/confirmation
  # tests, now run at full production scale for the first time.
  c5y) FREE="co2_ppm,p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo,amplitude_aerosol,height_aerosol"; AERO="--aerosol"; SOLVER="xrtm" ;;
  # 2026-09-18 (user): first production config on the O2-A band (fpa=0,
  # molecules=[o2,h2o] per input/geocarb_instrument.yml -- co2_ppm/ch4_ppb/
  # co_ppb are meaningless here, dropped from FREE entirely, not just left
  # frozen). All three aerosol shape parameters free (including thickness_
  # aerosol, deliberately reintroducing the amplitude*thickness degeneracy
  # from Sec.27/c5-c5x to see whether it behaves differently in this band)
  # -- unlike c5y, which dropped thickness_aerosol for the FPA2 production
  # design. XRTM required for height_aerosol per Sec.30's own settled
  # policy.
  o5x) FREE="p_surface_hpa,h2o_surface_vmr,t_offset_k,albedo,amplitude_aerosol,height_aerosol,thickness_aerosol"; AERO="--aerosol"; SOLVER="xrtm"; FPA="0" ;;
  *) echo "unknown CONF=${CONF}"; exit 1 ;;
esac

echo "CONF=${CONF}  FREE=${FREE}  ${AERO}  SOLVER=${SOLVER}  task ${SLURM_ARRAY_TASK_ID}/58"
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
    --fpa "${FPA}" --prior-fields structural --vary-albedo \
    --anchor-density 4 --surface-positions anchor --frozen-atmosphere-positions anchor \
    --g-ratio 1 --overlap 2 --hires-only --jacobian analytic --gn-verbose \
    --solver "${SOLVER}" \
    --anchor-workers "${SLURM_CPUS_PER_TASK}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" --n-tasks 58 \
    --free "${FREE}" ${AERO}
