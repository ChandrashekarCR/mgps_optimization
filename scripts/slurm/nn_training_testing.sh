#! /bin/bash
#SBATCH --job-name=neural-network-training-testing
#SBATCH --partition=gpua40i
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --nodes=1                  # 1 node
#SBATCH --ntasks-per-node=4        # 4 tasks per node
#SBATCH --cpus-per-task=1          # 1 CPU per task
#SBATCH --mem=16G                  # 16GB memory total
#SBATCH --time=01:00:00           # 1 hour runtime
#SBATCH --output=%x_%j.out        # Output file
#SBATCH --error=%x_%j.err         # Error file
#SBATCH --qos=normal

# Load modules
module --force purge
module load Anaconda3/2024.02-1

# Activate environment
conda init
conda activate binp37_env

# Your commands here
echo "Running on GPU node: $(hostname)"
echo "Starting job $SLURM_JOB_ID"


# Change to working directory
cd /home/chandru/binp37

python scripts/metasub/main.py --continent -d results/metasub/metasub_tax_training_testing.csv -b 32 -lr 0.001 -n 1 -e 200 -c True 