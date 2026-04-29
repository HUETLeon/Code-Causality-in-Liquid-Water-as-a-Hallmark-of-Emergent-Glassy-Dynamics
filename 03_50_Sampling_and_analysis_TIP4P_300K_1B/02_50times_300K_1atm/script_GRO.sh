#!/bin/bash
#SBATCH -J GROMACS
#SBATCH --nodes=1
##SBATCH --array=0-49
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
##SBATCH --mem=512GB
#SBATCH --partition=boost_usr_prod
##SBATCH --qos=boost_qos_dbg
#SBATCH --err=%x_%j.err
#SBATCH --out=%x_%j.out
#SBATCH --account=ICT25_CMSP_0

#source /home/ictp/ictp464830/.bashrc

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
srun gmx_mpi mdrun -s conf.tpr -x traj.xtc -o traj.trr -cpo traj.cpt -cpi traj.cpt -c conf.gro -maxh 23.5 

echo "END"

date "+%H:%M:%S   %d/%m/%y"

