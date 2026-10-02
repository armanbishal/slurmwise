#!/bin/bash
#SBATCH --job-name=sweep_lr_3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00

source ~/.venv/bin/activate
python sweep.py --lr 3e-5 --resume auto
