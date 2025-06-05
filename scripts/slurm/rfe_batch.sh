#!/bin/bash
#SBATCH --job-name=recursive_feature_selection
#SBATCH --partition=lu48
#SBATCH --account=lu2025-2-11
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16  # Reduced from 24
#SBATCH --mem=120G          # Increased memory
#SBATCH --time=02:00:00     # Increased time limit
#SBATCH --output=%x_%j.output
#SBATCH --error=%x_%j.error
#SBATCH --qos=normal

# Load modules
module --force purge
module load Anaconda3/2024.02-1

# Activate environment
conda activate binp37_env

# Set environment variables
export RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID
mkdir -p $RAY_TMPDIR

# Prevent Ray from using too much shared memory
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

echo "Starting job $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Available memory: $(free -h | grep '^Mem:' | awk '{print $7}')"

# Change to working directory
cd /home/chandru/binp37

# Run the script with error handling
python scripts/rfe_feature_selection.py \
    -i results/metasub/tax_metasub_data.csv \
    -o results/metasub/metasub_tax_train_test.csv \
    -s 42 \
    -p city \
    --num_cpus 12 || {
    echo "Python script failed with exit code $?"
    echo "Ray processes:"
    ps aux | grep ray | head -10
    echo "Memory usage:"
    free -h
    exit 1
}

echo "Job completed successfully"

# Cleanup
echo "Cleaning up Ray temporary files..."
rm -rf $RAY_TMPDIR
pkill -f ray:: 2>/dev/null || true