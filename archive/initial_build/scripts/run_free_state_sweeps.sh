#!/bin/bash
# Whole-slit joint-block sweeps with CO2 and surface pressure both retrieved,
# state-space interpolation on, for gratio 1 and gratio 3 -- then every
# residual and state-vector plot for each.
#
# Purpose: characterise SYSTEMATIC PATTERNS IN THE SPECTRAL RESIDUALS. The CO2
# bias in these runs is expected to be poor and is not the point: CO2 and
# surface pressure are near-perfectly degenerate in this band alone
# (corr(dCO2/CO2, dp/p) = -1.000, slope -2.02, measured 2026-08-17), so a free
# pressure row will absorb CO2 signal. That is a problem for the O2 A-band
# (FPA0) to solve, not this run.
#
# Each sweep runs BOTH the coarse and hi-res solves and saves, per window, the
# entire state vector (free AND frozen rows, with positions, priors, sigmas and
# correlation lengths) plus the complete residual field -- the standing
# save-everything rule.
#
# Priors are the physical-eta exponential form with per-row correlation lengths
# from geocarb_gert.joint_state.DEFAULT_CORR_LENGTH_ETA (co2 ~10 km hot-spot
# scale, p_surface ~140 km topography scale). Pass --prior-form tikhonov to
# reproduce the pre-2026-08-17 bin-index prior instead.
#
# Run:  bash scripts/run_free_state_sweeps.sh
#   or: bash scripts/run_free_state_sweeps.sh 8          # 8 workers
#   or: SKIP_EXISTING=0 bash scripts/run_free_state_sweeps.sh   # force re-run
#
# Env:
#   N_WORKERS      worker processes (default 12, or $1)
#   FREE           rows to retrieve (default co2_ppm,p_surface_hpa)
#   GRATIOS        space-separated g-ratios (default "1.0 3.0")
#   SKIP_EXISTING  1 (default) skips a sweep whose pickle already exists
#
# Outputs, per g-ratio:
#   results/gd_joint_block_whole_slit_fpa2_gratio<G>_stateinterp_free-co2-p.pkl
#   plots/joint_block/<stem>_state.png                     state vector, all rows
#   plots/joint_block/<stem>_residual_full_{coarse,hires}_pct.png
#   plots/joint_block/<stem>_residual_detector_*_pct.png   region detail
#   plots/joint_block/<stem>_residual_fitting_*_pct.png    within-bin vs wavenumber
#   plots/joint_block/<stem>_residual_vs_scene.png         residual vs scene structure

set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

N_WORKERS="${1:-${N_WORKERS:-12}}"
FREE="${FREE:-co2_ppm,p_surface_hpa}"
GRATIOS="${GRATIOS:-1.0 3.0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# gert is found automatically (../../gert then ../gert, validated by input/),
# so GERT_ROOT is only needed for a non-standard layout.
GERT_DIR="$(cd "$REPO/.." && pwd)/gert"
[ -d "$GERT_DIR/input" ] || GERT_DIR="$(cd "$REPO/../.." && pwd)/gert"
export PYTHONPATH="$REPO:$GERT_DIR"
PYTHON="${PYTHON:-python3}"

# free-row tag must match the sweep's own suffix logic (first token of each name)
TAG="free-$(echo "$FREE" | tr ',' '\n' | cut -d_ -f1 | paste -sd- -)"

echo "repo:      $REPO"
echo "gert:      $GERT_DIR"
echo "free rows: $FREE   (suffix _$TAG)"
echo "workers:   $N_WORKERS"
echo

fail=0
declare -a PICKLES=()

for GR in $GRATIOS; do
    STEM="gd_joint_block_whole_slit_fpa2_gratio${GR%.0}_stateinterp_${TAG}"
    PKL="results/${STEM}.pkl"
    PICKLES+=("$PKL")
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$PKL" ]; then
        echo "=== gratio $GR: $PKL exists, skipping sweep (SKIP_EXISTING=0 to force) ==="
        continue
    fi
    echo "=== gratio $GR: sweep (coarse + hi-res, --free $FREE) ==="
    "$PYTHON" scripts/gd_joint_block_whole_slit_sweep.py \
        --g-ratio "$GR" --state-interp --free "$FREE" --n-workers "$N_WORKERS" \
        || { echo "  FAILED: sweep gratio $GR"; fail=1; }
    echo
done

for PKL in "${PICKLES[@]}"; do
    if [ ! -f "$PKL" ]; then
        echo "=== $PKL missing -- skipping its plots ==="
        fail=1
        continue
    fi
    echo "=== plots for $(basename "$PKL") ==="
    # state vector along the slit: every row, free and frozen
    "$PYTHON" scripts/gd_joint_block_state_plot.py "$PKL" \
        || { echo "  FAILED: state plot"; fail=1; }
    # residuals: full detector + per-region detector/fitting space, % of continuum
    "$PYTHON" scripts/gd_joint_block_residual_plot.py "$PKL" --solve both --n-windows 4 \
        || { echo "  FAILED: residual plot"; fail=1; }
    # residual structure against the scene that drives it
    "$PYTHON" scripts/gd_joint_block_residual_vs_scene.py "$PKL" \
        || { echo "  FAILED: residual-vs-scene"; fail=1; }
    echo
done

echo "-----------------------------------------------------------"
if [ "$fail" -ne 0 ]; then
    echo "Done, WITH FAILURES -- see FAILED lines above."
else
    echo "Done. Sweeps and plots complete."
fi
echo
echo "Residual plots default to % of continuum, which needs the cached band"
echo "image. If they fell back to raw radiance with a warning, run:"
echo "  PYTHONPATH=\"$REPO:$GERT_DIR\" $PYTHON scripts/gd_cache_band_image.py --fpa 2"
echo "and re-run the residual plots."
exit "$fail"
