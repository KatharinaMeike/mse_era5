#!/bin/bash

#SBATCH --job-name=p37_march
#SBATCH --account=pi-tas1
#SBATCH --partition=caslake  # accessible partitions listed by the sinfo command
#SBATCH --ntasks-per-node=1  # number of tasks per node
#SBATCH --cpus-per-task=1    # number of CPU cores per task
#SBATCH --mem=128G


# match the no. of threads with the no. of CPU cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  

# Load the require module(s)
module load python/miniforge-25.3.0
module load uv

source /project2/tas1/katharinah/envs/mse/bin/activate

# Load the require module(s)
python3 p37_mse_budget.py
