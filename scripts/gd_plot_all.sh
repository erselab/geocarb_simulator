#!/bin/bash
# Generate every diagnostic plot for every FPA configuration that has been
# run through gd_test.py: for each of the 8 standard configs (single bands
# 0-3, plus joint 0+1, 0+2, 0+3, 0+1+2), calls all 4 plotting scripts.
# Each plotting script loops internally over its own scene x noise
# combinations (gd_plot.py and gd_plot_residual_spectra.py: 3 scenes x 2
# noise = 6 each; gd_plot_grid.py: same, 6; gd_plot_scene_overview.py: 3
# scenes only, no noise axis) and silently skips any case whose
# results/gd_joint_<fpas_tag>...pkl doesn't exist yet, so this is safe to
# run on a partial results/ directory -- it just plots whatever is there.
#
# Run:  bash scripts/gd_plot_all.sh
#   or: bash scripts/gd_plot_all.sh "0 1,2"     # only these configs
#   or: bash scripts/gd_plot_all.sh "0 1,2" native,undistorted
#       # drop rectified from gd_plot.py/gd_plot_grid.py/
#       # gd_plot_residual_spectra.py's panels (axes autoscale to whatever's
#       # left, no separate rescale step needed); gd_plot_scene_overview.py
#       # has no pipeline comparison, so this arg doesn't apply to it and is
#       # silently ignored there. Output filenames get a trailing
#       # _<pipelines> tag whenever this isn't the full default set, so
#       # filtered and unfiltered runs never collide.
# Output: plots/gd_joint_<fpas_tag>*.png (+ combined *_all.pdf for
# gd_plot.py/gd_plot_residual_spectra.py/gd_plot_grid.py)

set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON="/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python"
export PYTHONPATH="$(pwd):/scratch/scrowel3_lab/gert"

CONFIGS=(${1:-0 1 2 3 0,1 0,2 0,3 0,1,2})
PIPELINES="${2:-native,rectified,undistorted}"

PIPE_SCRIPTS=(
    scripts/gd_plot.py
    scripts/gd_plot_residual_spectra.py
    scripts/gd_plot_grid.py
)

fail=0
for fpas in "${CONFIGS[@]}"; do
    for script in "${PIPE_SCRIPTS[@]}"; do
        echo "=== $script --fpas $fpas --pipelines $PIPELINES ==="
        "$PYTHON" "$script" --fpas "$fpas" --pipelines "$PIPELINES" || { echo "  FAILED: $script --fpas $fpas"; fail=1; }
    done
    echo "=== scripts/gd_plot_scene_overview.py --fpas $fpas ==="
    "$PYTHON" scripts/gd_plot_scene_overview.py --fpas "$fpas" || { echo "  FAILED: scripts/gd_plot_scene_overview.py --fpas $fpas"; fail=1; }
done

echo
if [ "$fail" -ne 0 ]; then
    echo "Done, with at least one failure -- see FAILED lines above."
else
    echo "Done: plots for ${#CONFIGS[@]} FPA configs written to plots/."
fi
exit "$fail"
