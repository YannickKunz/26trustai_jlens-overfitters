#!/bin/bash
#------------------------
#SBATCH --account=gratis
#------------------------
#SBATCH --job-name="J-Lens fit GSM8K Questions + Answers"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=20:00:00

python3 src/main.py