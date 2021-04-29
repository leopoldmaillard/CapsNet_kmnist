#!/bin/bash

# Created from: slurm submission script, serial job
# support@criann.fr

# Max time the script will run (here 30 mins)
#SBATCH --time 00:30:00

# RAM to use (Mo)
#SBATCH --mem 10000

# Number of cpu core to use
#SBATCH --cpus-per-task=10

# Enable the mailing for the start of the experiments
#SBATCH --mail-type ALL
#SBATCH --mail-user leopold.maillard@insa-rouen.fr

# Which partition to use
#SBATCH --partition insa

# Number of gpu(s) to use
#SBATCH --gres gpu:1

# Number of nodes to use
#SBATCH --nodes 1

# Log files (%J is a variable for the job id)
#SBATCH --output %J.out
#SBATCH --error %J.err

#Loading the module
module load conda3/1907

# Activate conda to access env with keras
# conda activate DL-py3gpu

# Creating a directory to save the training weights
mkdir callbacks

# Define the repository where the trained weights will be stored
# This variable is used in the script mnist.py
# export LOCAL_WORK_DIR=checkpoints

# Start the calculation
srun python3 capsnet_kmnist.py
