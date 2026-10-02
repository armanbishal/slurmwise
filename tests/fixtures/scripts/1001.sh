#!/bin/bash
#SBATCH --job-name=train_llama_sft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/arman/sft/logs/%j.out
#SBATCH --error=/home/arman/sft/logs/%j.err

module load cuda/12.4
source ~/.venv/bin/activate
python train_sft.py --config configs/llama_sft.yaml --num_workers 8
