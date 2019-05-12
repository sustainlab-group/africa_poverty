#!/bin/bash

##########################
# This file is a template. Fill in the blanks with python.
#
# To stop this job, get the jobid using
#   squeue -u <username>
# then cancel the job with
#   scancel <jobid>
##########################

# choose the machine
#SBATCH --partition=atlas --qos=normal

# set the machine parameters
#SBATCH --nodes=1 --cpus-per-task=10 --mem={SLURM_MEM} --gres=gpu:1

# set the job name
#SBATCH --job-name={SLURM_JOB_NAME}

# set maximum time for job to run
# indefinite job: --time=0
# days/hours: --time=days-hours
#SBATCH --time=1-0

# set the output log name
#SBATCH --output={SLURM_OUTPUT_LOG}

# print out Slurm Environment Variables
echo "
Slurm Environment Variables:
- SLURM_JOBID=$SLURM_JOBID
- SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST
- SLURM_NNODES=$SLURM_NNODES
- SLURMTMPDIR=$SLURMTMPDIR
- SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR
"

# slurm doesn't source .bashrc automatically
source ~/.bashrc

project_dir="/atlas/u/chrisyeh/africa_poverty/"
echo "Setting directory to: $project_dir"
cd $project_dir

# list out some useful information
echo "
Basic system information:
- Date: $(date)
- Hostname: $(hostname)
- User: $USER
- pwd: $(pwd)
"

conda activate py37

{content}

echo "All jobs launched!"
echo "Waiting for child processes to finish..."
wait
echo "Done!"
