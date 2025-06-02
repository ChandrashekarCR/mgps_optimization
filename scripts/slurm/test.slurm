#! /bin/bash
#SBATCH --job-name=recursive_feature_selection  # Job name
#SBATCH --partition=lu48                        # Parittion name
#SBATCH --account=lu2025-2-11                          # Address
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                    # Tasks per node
#SBATCH --cpus-per-task=24                       # CPUs per task
#SBATCH --mem=64G                               # Memory per node
#SBATCH --time=00:20:00                         # Time limit
#SBATCH --output=%x_%j.output                   # Output file (jobname_jobid.out)
#SBATCH --error=%x_%j.error                     # Error file (jobname_jobid.err)
#SBATCH --qos=test                            # Using normal priority

#Load required modules
module load Anaconda3/2024.02-1 
conda activate binp37_env

#Calculate the ray parameters
echo "Starting job $SLURM_JOB_ID"

# Change to the working directory (update as needed)
cd /home/chandru/binp37
python scripts/rfe_feature_selection.py -i results/metasub/processed_metasub.csv -o results/metasub/temp.csv -s 42 -p city 
echo "Job completed successfully"