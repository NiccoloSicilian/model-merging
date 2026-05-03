#!/bin/bash
#SBATCH --job-name "test_dual_pretrained"
#SBATCH -A IscrC_TVU
#SBATCH --time 0:30:00
#SBATCH --qos boost_qos_dbg
#SBATCH -p boost_usr_prod
#SBATCH --mem=128G
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --error=logs/slurm/%j.err
#SBATCH --output=logs/slurm/%j.out

export http_proxy='http://login05:3142/'
export https_proxy='http://login05:3142/'

set -euo pipefail

cd /leonardo_scratch/fast/IscrC_TVU/agargiul/model-merging

source .venv/bin/activate

HYDRA_FULL_ERROR=1 uv run scripts/evaluate_multitask_merging.py merger=dual merger.dualize_pretrained=false

