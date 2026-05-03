#!/bin/bash
#SBATCH --job-name "dual_alpha_sweep"
#SBATCH -A IscrC_TVU
#SBATCH --time 4:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --mem=128G
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-11
#SBATCH --error=logs/slurm/%A_%a.err
#SBATCH --output=logs/slurm/%A_%a.out

export http_proxy='http://login05:3142/'
export https_proxy='http://login05:3142/'

set -euo pipefail

cd /leonardo_scratch/fast/IscrC_TVU/agargiul/model-merging

source .venv/bin/activate

ALPHAS=(0.1 0.3 0.5 0.7 0.9 1.0 1.1 1.3 1.5 1.7 1.9 2.1)

ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}

echo "Running alpha=${ALPHA}"

HYDRA_FULL_ERROR=1 uv run scripts/evaluate_multitask_merging.py \
  merger=dual \
  merger.dualize_pretrained=true \
  +merger.optimal_alphas.ViT-B-32.8=${ALPHA}
