#!/bin/bash
#SBATCH --job-name=gd_band_stress
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --array=1
#SBATCH --output=%x_fpa%a_%j.out
#SBATCH --error=%x_fpa%a_%j.err
#
# Composition/pressure stress test (scripts/gd_band_stress_test.py), one
# SLURM array task per FPA. Submit with:
#
#   sbatch scripts/gd_band_stress_test.slurm
#
# --array=1 : only FPA1/CO2_weak is actually clear to run right now.
#   FPA0/O2_A technically runs today but on a razor-thin ABSCO margin
#   (1.99 cm-1 at margin_cm1=10 -- breaks entirely at margin=15) -- add
#   ,0 back once its O2 block has been padded (recommended target
#   ~12745-13300, up from 12745-13245) in the same regeneration pass as
#   FPA3's fix, rather than running it now on the thin margin and
#   re-running after padding anyway.
#   FPA3/CH4_CO is hard-blocked until the ch4/h2o/co ABSCO blocks are
#   extended to ~4400 cm-1 -- it will fail with a clear "Wavenumber ...
#   not found" ValueError from gert.absco until then. Add ,3 once done.
#   FPA2/CO2_strong is deliberately not included here: it already has
#   its own dense-sweep results (gd_dense_sweep.py) from a uniform scene.
#
# --cpus-per-task: gd_band_stress_test.py auto-detects available cores
# (geocarb_gert.gd_render.available_cpus(), via os.sched_getaffinity) and
# uses all of them by default -- raise this to whatever the target node
# actually has (checked 2026-07-24: "standard" partition nodes have
# 56-96 cores; bhdrb6x/bhdrb8x nodes are the 96-core ones) for a
# proportional speedup. Memory stays low regardless of core count -- the
# ~2.2GB ABSCO table is shared read-only across all worker processes via
# fork copy-on-write, not duplicated per worker.
#
# Runtime: FPA1's forward-model calls are cheap (~0.04s each, vs FPA2's
# ~0.09-0.4s), so the dominant cost is still the 4096 retrievals (1024
# rows x native/rectified x order 0/2). At 32 cores this should be well
# under the 4-hour limit above; increase --time if using --row-step 1
# on a smaller core count, or decrease it once you've seen one array
# task's actual runtime in its .out file.

set -euo pipefail

REPO_ROOT="/scratch/scrowel3_lab/geocarb_simulator"
GERT_ROOT="/scratch/scrowel3_lab/gert"
PYTHON="/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python"

cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${GERT_ROOT}"

FPA="${SLURM_ARRAY_TASK_ID}"
echo "Host: $(hostname)   FPA: ${FPA}   CPUs: ${SLURM_CPUS_PER_TASK}"

"$PYTHON" scripts/gd_band_stress_test.py --fpa "${FPA}"
