#!/bin/bash
#SBATCH -J gromacs
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
##SBATCH --mem=256GB
#SBATCH --partition=acc_ehpc
#SBATCH --qos=acc_ehpc
#SBATCH --err=gromacs_%j.err
#SBATCH --out=gromacs_%j.out
#SBATCH --account=ehpc10

source /home/ictp/ictp464830/.bashrc

# SLURM_ARRAY_TASK_ID

#J_NUM=$SLURM_ARRAY_TASK_ID

# move to SCRATCH

echo "MOVING TO SCRATCH"

Submit_Dir="$SLURM_SUBMIT_DIR"
Calc_Dir="$SCRATCH/GROMACS/job_$SLURM_JOB_ID"

mkdir -p $Calc_Dir

cd $Calc_Dir

ln -s $Submit_Dir/* $Calc_Dir/

# Load GROMACS

date "+%H:%M:%S   %d/%m/%y"
echo "Load GROMACS"
module purge
module load openmpi/4.1.5-gcc fftw/3.3.10-gcc-ompi plumed/2.9.0-gcc-ompi
module load gromacs/2022.5_plumed-2.9.0-gcc-ompi

export SLURM_CPU_BIND=none

# Run the program

echo "RUN...."
gmx_mpi grompp -f md.mdp -c conf.gro -p topol.top -o conf.tpr
gmx_mpi mdrun -s conf.tpr -x 300w_300K_1atm.xtc -o 300w_300K_1atm.trr -cpo 300w_300K_1atm.cpt -c conf_end.gro 
echo "END"
date "+%H:%M:%S   %d/%m/%y"

# RECOVER DATA
echo "MOVING DATA"
cp -v 300w_300K_1atm.xtc 300w_300K_1atm.trr *log 300w_300K_1atm.cpt *.gro  $Submit_Dir/
ll

echo "DATA MOVED"
