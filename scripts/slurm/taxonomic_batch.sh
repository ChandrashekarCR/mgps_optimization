#! /bin/bash
#SBATCH --job-name=taxonomic_information        # Job name
#SBATCH --partition=lu48                        # Parittion name
#SBATCH --account=lu2025-2-11                   # Address
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                     # Tasks per node
#SBATCH --cpus-per-task=12                      # CPUs per task
#SBATCH --mem=16G                               # Memory per node
#SBATCH --time=04:00:00                         # Time limit
#SBATCH --output=%x_%j.output                   # Output file (jobname_jobid.out)
#SBATCH --error=%x_%j.error                     # Error file (jobname_jobid.err)
#SBATCH --qos=normal                            # Using normal priority


#Load required modules
module --force purge
module load Anaconda3/2024.06-1

# Activate environment
conda activate binp37_env 

#Calculate the ray parameters
echo "Starting job $SLURM_JOB_ID"

# Change to the working directory (update as needed)
cd /home/chandru/binp37
python scripts/metasub/data_preprocess/get_taxonomic_info.py -d results/metasub/processed_metasub.csv -o results/metasub/taxonomic_info.csv \
                    -ao results/metasub/tax_metasub_data.csv
