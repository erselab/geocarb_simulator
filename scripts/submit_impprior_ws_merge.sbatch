#!/bin/bash
# Merge a completed impprior_ws array + make the full state-vector /
# residual / reduced-chi2 along-slit plot.  sbatch --export=ALL,CONF=c1 ...
#
#SBATCH --job-name=impprior_ws_merge
#SBATCH --account=ltmurray_lab
#SBATCH --partition=atmos
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --output=slurm_out/impprior_ws_merge_%x_%j.out

set -euo pipefail
module load git/2.50.1
cd /scratch/scrowel3_lab/geocarb_simulator
export PYTHONPATH="/scratch/scrowel3_lab/geocarb_simulator:/scratch/scrowel3_lab/gert"
PYTHON=/gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python

: "${CONF:?set CONF=c1|c2|c3|c4|c5}"
ROOT=results/realistic_prior
COMMON="gratio0.5_adens16"
TAIL="analytic_prior-structural_valb_spos-anchor_fapos-anchor"
case "${CONF}" in
  c1) DIR="${ROOT}/gd_joint_block_whole_slit_fpa2_${COMMON}_${TAIL}_parts"; TRUTH=scratch_work/impprior_truth_noaero_A.pkl ;;
  c2) DIR="${ROOT}/gd_joint_block_whole_slit_fpa2_${COMMON}_free-co2-p_${TAIL}_parts"; TRUTH=scratch_work/impprior_truth_noaero_A.pkl ;;
  c3) DIR="${ROOT}/gd_joint_block_whole_slit_fpa2_${COMMON}_free-co2-p-albedo_${TAIL}_parts"; TRUTH=scratch_work/impprior_truth_noaero_A.pkl ;;
  c4) DIR="${ROOT}/gd_joint_block_whole_slit_fpa2_${COMMON}_free-co2-p-h2o-t-albedo_${TAIL}_parts"; TRUTH=scratch_work/impprior_truth_noaero_A.pkl ;;
  c5) DIR="${ROOT}/gd_joint_block_whole_slit_fpa2_${COMMON}_free-co2-p-h2o-t-albedo-tau-height_${TAIL}_aero_parts"; TRUTH=scratch_work/impprior_truth_aero_A.pkl ;;
  *) echo "unknown CONF"; exit 1 ;;
esac

echo "merging ${DIR}"
${PYTHON} scripts/gd_joint_block_retrieve.py --merge "${DIR}"
MERGED="${DIR%_parts}.pkl"
echo "plotting ${MERGED}"
${PYTHON} scripts/gd_joint_block_retrieve.py --plot "${MERGED}" --truth-image "${TRUTH}"
echo "MERGE+PLOT DONE for ${CONF}"
