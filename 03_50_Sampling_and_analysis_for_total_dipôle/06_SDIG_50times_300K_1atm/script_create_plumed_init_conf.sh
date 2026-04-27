#!/bin/bash
#SBATCH -J create_plumed
#SBATCH --nodes=1
#SBATCH --array=0-49
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=1
##SBATCH --gres=gpu:4
##SBATCH --mem=512GB
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal	
#SBATCH --err=%x_%A_%a.err
#SBATCH --out=%x_%A_%a.out
#SBATCH --account=ICT25_RECOVERY_0

# SOURCING

source ~/.bashrc

module load anaconda3/2023.09-0

export OMP_NUM_THREADS=112

# SLURM_ARRAY_TASK_ID

J_NUM=$SLURM_ARRAY_TASK_ID

# move to SCRATCH

mkdir -p Run_$J_NUM

cd Run_$J_NUM

cp ../py_create_plumed_init_conf.py .
cp ../*dat .
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

python py_create_plumed_init_conf.py --seed $((${J_NUM} + 1))

echo "END"
date "+%H:%M:%S   %d/%m/%y"

# RECOVER DATA
#echo "MOVING DATA"
#cp -v ./*  $Submit_Dir/

#echo "DATA MOVED"
