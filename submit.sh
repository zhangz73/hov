#!/bin/bash
#SBATCH --job-name=empire_gpu
#SBATCH --partition=cpu
#SBATCH --account=cornell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00     # Time limit hrs:min:sec
#SBATCH --output=job_%j.out

# Optional: make Python aware of CPU count
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run your command
srun python3 opt_toll_noseg.py

