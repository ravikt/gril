#!/bin/bash
#SBATCH --get-user-env=L             #Replicate login environment
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --job-name="agil_imitation"
#SBATCH --mem=56000M
#SBATCH --output=agil_log.txt
#SBATCH --mail-user=ravikt@tamu.edu
#SBATCH --mail-type=All
##SBATCH --test-only
#
echo "SLURM_JOB_ID="$SLURM_JOB_ID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

cd $SLURM_SUBMIT_DIR
echo "working directory = "$SLURM_SUBMIT_DIR


module load Python/3.6.6-intel-2018b
source ../vscl/col/bin/activate
module list
python agil.py -d ../airsim/test_data/moving_truck14.npz


#

#echo "Launch test with srun"
#NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
#export OMP_NUM_THREADS=1
#echo NPROCS=$NPROCS


#
echo "All Done!"
