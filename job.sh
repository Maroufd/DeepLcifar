#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --job-name=deep_learning
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32g
##------------------------ End job description ------------------------
module purge && module load Python/3.9.5-GCCcore-10.3.0
source ~/.virtualenvs/deep_learning/bin/activate
srun python code_cnn.py
