#!/bin/bash
#SBATCH --partition=atmos
#SBATCH --account=ltmurray_lab
#SBATCH --job-name=mb_probe
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_out/mb_probe_%x_%j.out
# Cost/memory probe for ONE multi-band window (phase 5 of docs/MULTIBAND_PLAN.md): the solve plus a
# background cgroup sampler (memory.peak / memory.current every 120 s) written to
# slurm_out/mb_probe_mem_<jobid>.csv, since the cgroup disappears when the job ends.
#   ARGS="--fpas 0,2 --tile 0 --anchor-mode cover --g-ratio 1 --free ..." sbatch --export=ALL scripts/submit_multiband_probe.sbatch
set -euo pipefail
cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
SHM_NAME="/dev/shm/geocarb_L_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
CG="/sys/fs/cgroup/system.slice/slurmstepd.scope/job_${SLURM_JOB_ID}"
MEMCSV="slurm_out/mb_probe_mem_${SLURM_JOB_ID}.csv"
echo "epoch,peak_bytes,current_bytes" > "${MEMCSV}"
( while true; do echo "$(date +%s),$(cat ${CG}/memory.peak 2>/dev/null || echo NA),$(cat ${CG}/memory.current 2>/dev/null || echo NA)" >> "${MEMCSV}"; sleep 120; done ) &
SAMPLER=$!
trap 'echo "$(date +%s),$(cat ${CG}/memory.peak 2>/dev/null || echo NA),$(cat ${CG}/memory.current 2>/dev/null || echo NA)" >> "${MEMCSV}"; kill ${SAMPLER} 2>/dev/null || true; rm -f "${SHM_NAME}"*' EXIT
/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python -u scripts/gd_multiband_window.py ${ARGS} --anchor-workers "${SLURM_CPUS_PER_TASK}"
