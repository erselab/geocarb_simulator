#!/bin/bash
# Ablation #3 (2026-09-04, user): "isolate frozen-row contamination" --
# rerun the structural-prior/Mode-2-truth experiment
# (co2p_albedo-58-g1-ad16-pf-exact-MODE2TRUTH_structural_analytic.pkl) with
# the three FROZEN atmosphere rows (ch4_ppb, co_ppb, h2o_surface_vmr) fixed
# at exact truth instead of the structural prior. Free rows
# (co2_ppm, p_surface_hpa, albedo) keep their imperfect (structural) prior
# unchanged. See scratch_work/retrieval_vs_mode2_truth_frozen_exact_array.py
# for the full rationale and exactly what differs from the structural run.
#
# Two-level parallelism (validated this session): one SLURM array task per
# whole-slit window (--task-id/--n-tasks, top-level processes, not daemons)
# PLUS anchor-level parallelism within each window's own linearize/
# build_forward_state calls (ANCHOR_WORKERS, read by the driver script).
#
#SBATCH --job-name=gd_frozen_exact_ablation
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --array=0-57
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_out/frozen_exact_ablation_%A_%a.out

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

export SLURM_ARRAY_TASK_COUNT=58
export ANCHOR_WORKERS="${SLURM_CPUS_PER_TASK}"

"${PYTHON}" scratch_work/retrieval_vs_mode2_truth_frozen_exact_array.py
