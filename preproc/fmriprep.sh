#!/bin/bash

#SBATCH --account=scraplab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=fmriprep_1_2
#SBATCH --mail-type=END
#SBATCH --mail-user=gabriel.j.fajardo.gr@dartmouth.edu
#SBATCH --output=slurm/%x/slurm_%x_%a_%j.out
#SBATCH --array=1-2

# get subject ID from array job
participants=(01 02)
PARTICIPANT_LABEL=${participants[(${SLURM_ARRAY_TASK_ID} - 1)]}
echo "array id: " ${SLURM_ARRAY_TASK_ID}, "subject id: " ${PARTICIPANT_LABEL}

# create a folder for slurm outputs if it already doesn't exist
mkdir -p "slurm/$SLURM_JOB_NAME"

# fMRIPrep parameters
BIDS_DIR=/dartfs/rc/lab/D/DBIC/DBIC/psych160/data/stopsignal
OUTPUT_DIR=/dartfs/rc/lab/S/Scraplab/gabe_fajardo/exam/data/fmriprep
WORK_DIR=/dartfs/rc/lab/S/Scraplab/gabe_fajardo/exam/preproc/fmriprep_work
FMRIPREP_RESOURCES_PATH=/dartfs/rc/lab/D/DBIC/DBIC/psych160/resources/fmriprep

# run fMRIPrep using singularity container
singularity run \
    --cleanenv \
    -B ${FMRIPREP_RESOURCES_PATH}:/resources \
    -B ${BIDS_DIR}:/data \
    -B ${WORK_DIR}:/work \
    -B ${OUTPUT_DIR}:/output \
    ${FMRIPREP_RESOURCES_PATH}/fmriprep-25.2.5.sif /data /output \
    participant --participant_label $PARTICIPANT_LABEL \
    -w /work \
    --nprocs 8 \
    --write-graph \
    --fs-license-file /resources/license.txt \
    --ignore slicetiming \
    --fs-no-reconall