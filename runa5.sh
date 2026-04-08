#!/bin/bash
#SBATCH --job-name=experiments_notebook
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --partition=mtech
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn07

# Create logs directory if it doesn't exist
mkdir -p logs

# Print some node info
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Initialize Conda
source ~/.bashrc
conda activate dl_env
pip install -r requirements.txt 

# Execute the notebook
python Que2.py
echo "Job finished"