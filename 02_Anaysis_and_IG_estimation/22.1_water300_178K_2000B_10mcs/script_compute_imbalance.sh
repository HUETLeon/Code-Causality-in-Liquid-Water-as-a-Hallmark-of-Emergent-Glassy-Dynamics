#!/bin/bash
#SBATCH -J compute_imbalance
#SBATCH --nodes=1
#SBATCH --array=1-50
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
##SBATCH --gres=gpu:4
##SBATCH --mem=512GB
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=qos_prio	
#SBATCH --err=%x_%A_%a.err
#SBATCH --out=%x_%A_%a.out
#SBATCH --account=ICT25_RECOVERY_0

# SOURCING

source ~/.bashrc

module load anaconda3/2023.09-0

export OMP_NUM_THREADS=2

# SLURM_ARRAY_TASK_ID

J_NUM=$SLURM_ARRAY_TASK_ID

# move to SCRATCH

#echo "MOVING TO SCRATCH"

#Submit_Dir=${SLURM_SUBMIT_DIR}
#Calc_Dir="$SCRATCH/MDAnalyis/job_$SLURM_JOB_ID"

#mkdir -p $Calc_Dir

#cd $Calc_Dir

#ln -s $Submit_Dir/* $Calc_Dir/

# Load MDAnalysis env

date "+%H:%M:%S   %d/%m/%y"

#export SLURM_CPU_BIND=none

# Run the program

echo "RUN...."

which python # get_density_histo.py 1> get_density_histo.out 2> get_density_histo.err

python py_compute_imbalance.py --seed ${J_NUM} 

echo "END"
date "+%H:%M:%S   %d/%m/%y"

# RECOVER DATA
#echo "MOVING DATA"
#cp -v ./*  $Submit_Dir/

#echo "DATA MOVED"
