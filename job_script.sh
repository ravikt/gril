#!/bin/bash
#SBATCH --get-user-env=L             #Replicate login environment
<<<<<<< HEAD
#SBATCH --time=4:00:00
=======
#SBATCH --time=2:00:00
>>>>>>> main
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
<<<<<<< HEAD
python agil.py -d ../airsim/test_data/moving_truck14.npz
=======
python agil.py -i ../airsim/test_data/truck_mountains3.npz -l ../airsim/test_data/act_tm3.npz
>>>>>>> main


#

#echo "Launch test with srun"
#NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
#export OMP_NUM_THREADS=1
#echo NPROCS=$NPROCS


#
echo "All Done!"
