#!/bin/bash
# Whole-slit sub_bin_anomaly sweep (docs/PROJECT_STATUS.md Sec.10,
# 2026-08-28): does the additive high-resolution-anchor-sampling mechanism
# (geocarb_gert.joint_state.ParamSpec.sub_bin_anomaly) actually help across
# the WHOLE slit, not just the single boundary window (rows 418-438) it was
# validated on so far?
#
# Three setups (--sub-bin-anomaly):
#   none    baseline -- no sub_bin_anomaly at all, plain linear interpolation
#           of albedo's own retrieved bin values (today's mechanism).
#   truth   albedo's sub_bin_anomaly.g_fn = als.albedo_for_label(...,
#           include_fine=True), the EXACT true fine-scale texture.
#   prior   albedo's sub_bin_anomaly.g_fn = als.albedo_for_label_prior_fine,
#           an INDEPENDENT noise realization with the same statistics
#           (ALBEDO_COV, ALBEDO_CORR_KM) but not the true phase -- what a
#           real, imperfect external product (MODIS-like) would supply.
#
# Crossed with anchor_density in {1, 4, 16} (user request, 2026-08-28) -- 3
# setups x 3 anchor densities = 9 independent configs, one SLURM array task
# per config, same "one task = one full 58-window sweep, internal
# multiprocessing Pool across windows" pattern as
# submit_retrieval_prior_albedo_v1.sbatch.
#
# Every config uses:
#   --free co2_ppm,p_surface_hpa,albedo --vary-albedo   (albedo free & real)
#   --prior-fields structural   (prior knows the patch layout, NOT the fine
#           texture -- isolates sub_bin_anomaly's OWN contribution, since
#           the only place fine-scale information can enter the retrieval
#           at all is through g_fn, not through a truth-sampled prior --
#           same reasoning as the single-window Sec.10 investigation)
#   --hires-only   (sub_bin_anomaly is exactly a no-op on the coarse solve
#           -- ParamSpec.sub_bin_anomaly's own docstring: value(bin_center_k)
#           == bin_value_k always -- so a coarse solve here would just be
#           redundant compute, same as submit_retrieval_prior_albedo_v1's
#           own convention)
#   --jacobian analytic --prior-form exponential --overlap 2 --n-windows 58
#           (standard conventions, matching every other prior sweep)
#
# Output under NINE tags (one per setup x anchor_density combo):
#   results/realistic_prior/config_matrix/sub_bin_anomaly_v1/results/
#     co2p_albedo-58-g<G_RATIO>[-ad<AD>]-pf-structural[-anomaly-<setup>]_analytic.pkl
#
# Per-task cost: the single-window smoke test (this feature's own
# validation) took ~1200s for a G=21 hires solve at anchor_density=16,
# g_ratio=1 -- SLOWER than production's default g_ratio (below), but the
# real per-task number for a full 58-window sweep at THIS script's
# settings is UNKNOWN, same caveat submit_retrieval_prior_albedo_v1.sbatch
# itself carries. --time below is a generous safety margin, not a real
# estimate -- if any task times out, that's the signal to measure a real
# config and come back with a tighter number.
#
# CHECK BEFORE SUBMITTING: --account is confirmed (ltmurray_lab); --partition
# is a guess (atmos/atmos2 look right on this cluster, `sinfo` for the full
# list) -- adjust to whatever allocation you actually want to use.
#
#SBATCH --job-name=gd_sub_bin_anomaly_v1
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --array=0-8
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_out/sub_bin_anomaly_v1_%A_%a.out

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

# -- axes, in nested order (setup slowest, anchor_density fastest);
# array-index -> config mapping only needs to be a bijection over all 9.
SETUPS=(none truth prior)
ADENSS=(1 4 16)
N_AD=${#ADENSS[@]}

IDX="${SLURM_ARRAY_TASK_ID}"
SETUP_IDX=$(( IDX / N_AD ))
AD_IDX=$(( IDX % N_AD ))
SETUP="${SETUPS[$SETUP_IDX]}"
AD="${ADENSS[$AD_IDX]}"

G_RATIO=3   # retrieval_defaults.yml's own default, explicit for reproducibility
FREE_SET=co2_ppm,p_surface_hpa,albedo

TAG=sub_bin_anomaly_v1
CID="co2p_albedo-58-g${G_RATIO}"
if [[ "${AD}" != "1" ]]; then
    CID="${CID}-ad${AD}"
fi
CID="${CID}-pf-structural"
if [[ "${SETUP}" != "none" ]]; then
    CID="${CID}-anomaly-${SETUP}"
fi

OUT_DIR="results/realistic_prior/config_matrix/${TAG}/results"
mkdir -p "${OUT_DIR}"

echo "task ${IDX}: tag=${TAG} setup=${SETUP} anchor_density=${AD} cid=${CID}"

"${PYTHON}" scripts/gd_joint_block_whole_slit_sweep.py \
    --g-ratio "${G_RATIO}" --free "${FREE_SET}" \
    --n-windows 58 --anchor-density "${AD}" \
    --hires-only --jacobian analytic \
    --prior-fields structural --sub-bin-anomaly "${SETUP}" \
    --prior-form exponential --overlap 2 --vary-albedo \
    --n-workers "${SLURM_CPUS_PER_TASK}" \
    --out "${OUT_DIR}/${CID}_analytic.pkl"

# No chained postprocess step yet (unlike submit_retrieval_prior_albedo_v1.sh)
# -- sub_bin_anomaly isn't wired into gd_joint_block_matrix.py's config_id()/
# scoring machinery. Inspect results directly: each output pkl's
# payload["results"][(row_lo,row_hi)]["hires"] is a StateSpec.snapshot() dict
# (geocarb_gert.along_slit_state.stack_windows_along_slit(payload["results"]
# .values(), "albedo", solve="hires") stitches the whole slit together, same
# convention notebooks/pkl_results_explorer.ipynb already uses for a single
# pkl) -- or load all nine and diff native-resolution error the same way
# scripts/gd_information_density_bins_demo.py's own _native_error_summary
# does, just aggregated over every window instead of one.
