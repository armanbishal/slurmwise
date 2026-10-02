#!/bin/bash
#SBATCH --job-name=energy_bench_full
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

source ~/.venv/bin/activate
python bench.py --problems all --models all --repeats 5
