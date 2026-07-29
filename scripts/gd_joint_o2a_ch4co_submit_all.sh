#!/bin/bash
# Submit the joint O2-A (FPA0, ~760 nm) + CH4/CO (FPA3, ~2300 nm) battery --
# retrieves H2O + surface pressure (via FPA0) jointly with CH4 + CO + H2O
# (via FPA3), the same no-aerosol methodology as the FPA0+FPA2 battery
# (KEYSTONE_SMILE_BIAS_PLAN.md Sec. 11g/11h), just a different band pair.
#
# This is a thin wrapper, not a separate implementation: gd_joint_band_test.py
# and gd_joint_band_test_submit_all.sh are already fully general via FPAS
# (see their own headers) -- this script only fixes the pair and gives it a
# name that's easy to find/run. Same 3 scenes x 2 noise settings x
# all-three-pipelines battery, same output-filename convention
# (results/gd_joint_fpa0_fpa3[_uniform|_barcode][_noise].pkl).
#
# Run:  bash scripts/gd_joint_o2a_ch4co_submit_all.sh
# Analyze with:
#   .../python scripts/gd_joint_band_plot.py --fpas 0,3
#   .../python scripts/gd_joint_band_plot_residual_spectra.py --fpas 0,3

set -euo pipefail
cd "$(dirname "$0")"

FPAS=0,3 bash gd_joint_band_test_submit_all.sh
