#!/bin/bash
# Template for running any script of this repository on a single A100 node with SLURM.
# Adjust partition, environment and command to your cluster. Submit from the repository root:
#   sbatch scripts/slurm/example_job.sbatch

#SBATCH --job-name=novelty-reviewer
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
##SBATCH --partition=<your-partition>

set -euo pipefail
echo "Running on $(hostname)"

# source activate novelty-reviewer   # your environment

python scripts/evaluation/generate_predictions.py \
    --paper-texts data/eval_paper_texts.parquet \
    --model AbeerMostafa/Novelty_Reviewer --prompt novelty_reviewer
