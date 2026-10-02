#!/bin/bash
#SBATCH --job-name=grpo_granite_2b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

module load cuda/12.4
echo "Activating env /home/arman/.venv"
python scripts/train_grpo.py --config configs/shaped.yaml
