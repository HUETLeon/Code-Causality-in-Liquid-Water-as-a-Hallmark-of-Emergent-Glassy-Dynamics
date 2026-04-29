#!/bin/bash
#SBATCH -J GROMACS
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
##SBATCH --mem=512GB
#SBATCH --partition=boost_usr_prod
##SBATCH --qos=boost_qos_dbg
#SBATCH --err=%x_%j.err
#SBATCH --out=%x_%j.out
#SBATCH --account=ICT25_RECOVERY
#SBATCH --exclusive
#SBATCH --mem=0

source /home/ictp/ictp464830/.bashrc

# SLURM_ARRAY_TASK_ID

#J_NUM=$SLURM_ARRAY_TASK_ID

# move to SCRATCH

echo "MOVING TO SCRATCH"

#Submit_Dir="$SLURM_SUBMIT_DIR"
#Calc_Dir="$SCRATCH/GROMACS/job_$SLURM_JOB_ID"

#mkdir -p $Calc_Dir

#cd $Calc_Dir

#ln -s $Submit_Dir/* $Calc_Dir/

# Load GROMACS

date "+%H:%M:%S   %d/%m/%y"
echo "Load GROMACS"
module purge
module load profile/chem-phys
module load gromacs/2022.3--openmpi--4.1.6--gcc--12.2.0-cuda-12.1

export SLURM_CPU_BIND=none

# Run the program

echo "RUN...."

export GMX_ENABLE_DIRECT_GPU_COMM=TRUE

gmx_mpi grompp -f md.mdp -c conf.gro -p topol.top -o conf.tpr

srun gmx_mpi mdrun -s conf.tpr -x 300w_300K_1atm.xtc -o 300w_300K_1atm.trr -cpo 300w_300K_1atm.cpt -cpi 300w_300K_1atm.cpt -c conf.gro -maxh 23.5 #-append

echo "END"

date "+%H:%M:%S   %d/%m/%y"

# RECOVER DATA
#echo "MOVING DATA"
#cp -v 300w_300K_1atm.xtc 300w_300K_1atm.trr *log 300w_300K_1atm.cpt *.gro  $Submit_Dir/
#ll

#echo "DATA MOVED"

#cd $Submit_Dir

#sbatch script_relaunch.sh

