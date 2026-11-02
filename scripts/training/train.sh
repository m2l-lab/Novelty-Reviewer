#!/usr/bin/env bash
# Full fine-tuning of Llama-3.1-8B-Instruct into the Novelty Reviewer model with axolotl, DeepSpeed ZeRO-3 and
# Liger kernels. The paper used 32 A100-80GB GPUs (8 nodes x 4); adapt num_machines / num_processes in
# scripts/training/configs/accelerate_32xA100.yaml to your setup. Run from the repository root.
#
#   pip install "axolotl[flash-attn,deepspeed]"   # see requirements/full-environment.txt for exact versions
#   bash scripts/training/train.sh
set -euo pipefail

CONFIG=${1:-scripts/training/configs/llama31_8b_full_finetune.yaml}

accelerate launch --config_file scripts/training/configs/accelerate_32xA100.yaml -m axolotl.cli.train "$CONFIG"
