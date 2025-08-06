#!/bin/bash
#SBATCH --job-name=recursive_feature_selection
#SBATCH --partition=lu48
#SBATCH --account=lu2025-2-11
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24      # Use available cores on lu48
#SBATCH --mem=160G              # Sufficient memory allocation
#SBATCH --time=12:00:00         # Increased time limit for larger datasets
#SBATCH --output=rfe_job_%j.output
#SBATCH --error=rfe_job_%j.error
#SBATCH --qos=normal
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@example.com  # Replace with your email

# Load modules - use generic module name that exists on the system
module --force purge
module load Anaconda3    # Removed specific version that doesn't exist

# Initialize conda for bash shell
source $(conda info --base)/etc/profile.d/conda.sh

# Activate environment
conda activate binp37_ray_env

# Set environment variables - changed to user-writable directory
export RAY_TMPDIR=/home/chandru/binp37/tmp/ray_$SLURM_JOB_ID
mkdir -p $RAY_TMPDIR

# Enhanced Ray configuration
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_DISABLE_MEMORY_MONITOR=1
# Configure dashboard (optional)
export RAY_DASHBOARD_PORT=8265
export RAY_DASHBOARD_HOST=0.0.0.0
# Set plasma store memory limit explicitly to avoid auto-detection issues
export RAY_OBJECT_STORE_MEMORY=40000000000  # ~40GB for object store
# Use plasma object store in user's home directory
export RAY_plasma_store_socket_name=/home/chandru/binp37/tmp/plasma_store_$SLURM_JOB_ID

# Print job information
echo "================ Job Information ================"
echo "Starting job $SLURM_JOB_ID at $(date)"
echo "Running on node: $HOSTNAME"
echo "Allocated CPU cores: $SLURM_CPUS_ON_NODE"
echo "Allocated memory: $SLURM_MEM_PER_NODE MB"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Available memory: $(free -h | grep '^Mem:' | awk '{print $7}')"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Ray temp directory: $RAY_TMPDIR"
echo "==============================================="

# Create checkpoint directory for potential job recovery
export CHECKPOINT_DIR=/home/chandru/binp37/checkpoints/rfe_job_${SLURM_JOB_ID}
mkdir -p $CHECKPOINT_DIR
echo "Checkpoint directory: $CHECKPOINT_DIR"

# Change to working directory
cd /home/chandru/binp37

# Save start time
start_time=$(date +%s)

# Test Ray initialization separately to provide better error messages
echo "Testing Ray initialization..."
python -c "
import ray
try:
    ray.init(ignore_reinit_error=True, 
             num_cpus=$((SLURM_CPUS_ON_NODE - 2)),
             _temp_dir='$RAY_TMPDIR')
    print('Ray initialized successfully')
    ray.shutdown()
except Exception as e:
    print(f'Ray initialization failed: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "ERROR: Ray initialization test failed. Exiting."
    exit 1
fi

# Run the script with comprehensive error handling
# REMOVED the --checkpoint_dir parameter that isn't supported
echo "Starting feature selection script at $(date)"
python scripts/feature_engineering/rfe_feature_selection.py \
    -i results/metasub/processed_metasub.csv \
    -o results/metasub/metasub_training_testing_data_new.csv \
    -s 42 \
    -p city \
    --num_cpus $((SLURM_CPUS_ON_NODE - 2)) || {
    rc=$?
    end_time=$(date +%s)
    runtime=$((end_time - start_time))
    
    echo "================ ERROR REPORT ================"
    echo "Python script failed with exit code $rc"
    echo "Job failed after running for $runtime seconds ($(date -d@$runtime -u +%H:%M:%S))"
    echo "Memory usage at failure:"
    free -h
    echo "CPU usage:"
    top -b -n 1 | head -20
    echo "Ray processes:"
    ps aux | grep ray | head -20
    
    # Check if ray command exists before trying to use it
    if command -v ray &> /dev/null && pgrep -f "ray" > /dev/null; then
        echo "Ray object store status (if available):"
        ray memory 2>/dev/null || echo "Ray memory command failed"
    else
        echo "Ray is not running, skipping ray memory check"
    fi
    
    echo "Python path:"
    which python
    echo "Python environment:"
    pip list | grep ray
    
    echo "Last 50 lines of ray logs (if available):"
    find $RAY_TMPDIR -name "*.log" -type f -exec ls -la {} \; 2>/dev/null || echo "No Ray log files found"
    find $RAY_TMPDIR -name "*.log" -type f -exec tail -n 20 {} \; 2>/dev/null || echo "No Ray log content available"
    echo "==============================================="
    
    # Save error data for debugging
    mkdir -p $CHECKPOINT_DIR/error_data
    top -b -n 1 > $CHECKPOINT_DIR/error_data/top.log
    free -h > $CHECKPOINT_DIR/error_data/memory.log
    ps aux | grep ray > $CHECKPOINT_DIR/error_data/ray_processes.log
    pip list > $CHECKPOINT_DIR/error_data/pip_packages.log
    cp $SLURM_JOB_NAME.$SLURM_JOB_ID.error $CHECKPOINT_DIR/error_data/ 2>/dev/null || true
    echo "Error data saved to $CHECKPOINT_DIR/error_data"
    
    exit $rc
}

# Calculate runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))

echo "================ JOB SUMMARY ================"
echo "Job completed successfully at $(date)"
echo "Total runtime: $runtime seconds ($(date -d@$runtime -u +%H:%M:%S))"
echo "Results saved to: results/metasub/metasub_training_testing_data_new.csv"
echo "==============================================="

# Cleanup
echo "Cleaning up Ray temporary files..."
rm -rf $RAY_TMPDIR
pkill -f "ray::" 2>/dev/null || true
echo "Cleanup completed"