#!/bin/bash

#SBATCH --account=scraplab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=5:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=connectivity_1stlevel
#SBATCH --mail-type=END
#SBATCH --mail-user=gabriel.j.fajardo.gr@dartmouth.edu
#SBATCH --output=slurm/%x/slurm_%x_%a_%j.out
#SBATCH --array=1-13

# get subject ID from array job
subs=(01 02 03 04 05 06 07 09 10 12 13 14 15)
SUB_LABEL=${subs[(${SLURM_ARRAY_TASK_ID} - 1)]}
echo "array id: " ${SLURM_ARRAY_TASK_ID}, "subject id: " ${SUB_LABEL}

# create a folder for slurm outputs if it already doesn't exist
mkdir -p "slurm/$SLURM_JOB_NAME"

# initialize bash shell
#conda init bash
source /optnfs/common/miniconda3/etc/profile.d/conda.sh

# activate the conda environment
conda activate pyfmri

python -u connectivity.py $SUB_LABEL