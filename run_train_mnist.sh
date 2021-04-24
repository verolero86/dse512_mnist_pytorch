#!/bin/bash -l
#PBS -N train_mnist_dse512
#PBS -A ACF-UTK0150
#PBS -o logs/p1_$PBS_JOBNAME.o$PBS_JOBID
#PBS -j oe
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:10:00
#PBS -l qos=condo

# Perform some simple commands
set -x 

##
## FOR YOUR JOBS, REPLACE BELOW WITH YOUR CODE
##

# NOTE: first run the following in your environment (named dse512 here)
# conda install pytorch=1.7.1=cpu_py38h6a09485_0
module load anaconda3
source $ANACONDA3_SH
#conda activate dse512
conda activate /lustre/haven/proj/UTK0150/envs/pytorch181

cd $PBS_O_WORKDIR

export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
export MASTER_PORT=$(( 12345 + $PBS_NP ))

env

mpirun -n $PBS_NP python train_mnist.py
