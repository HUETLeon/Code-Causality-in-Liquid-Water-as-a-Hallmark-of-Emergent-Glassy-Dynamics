#!/bin/bash
#SBATCH -J GPUMD
#SBATCH --nodes=1
#SBATCH --array=1-50
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
##SBATCH --mem=512GB
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --err=%x_%A_%a.err
#SBATCH --out=%x_%A_%a.out
#SBATCH --account=ICT25_CMSP_0

RESTART="YES"

source /leonardo/home/userexternal/lhuet000/.bashrc

# SLURM_ARRAY_TASK_ID

I=$SLURM_ARRAY_TASK_ID

# move to SCRATCH


cp nep.txt run.in run_$I/
cd run_$I
	mv restart.xyz model.xyz
	#head -n $((3500000*902)) dump.xyz > temporar_dump.xyz
	#mv temporar_dump.xyz dump.xyz
	#rm neighbor.out force.out EXIT

#mkdir -p $Calc_Dir

#cd $Calc_Dir

#ln -s $Submit_Dir/* $Calc_Dir/

# Load GROMACS

date "+%H:%M:%S   %d/%m/%y"
echo "Load CUDA"
module purge
nvidia-smi

module load nvhpc

nvcc --version

module list
#export SLURM_CPU_BIND=none

echo "Launch ..."

gpumd  >> gpumd.out


date "+%H:%M:%S   %d/%m/%y"


if [ $RESTART == "YES" ]
then
	echo "Checking is this is the last sub in the array" 
	if [ $I -eq "50" ]
	then
		echo "Checking the size of the output"
		L=$( wc -l neighbor.out | cut -d " " -f 1)
		if [ $L -lt "100000" ]
			echo "Resart validated wating for other to stop."
			sleep 3h
			echo "Restarting at:" 
			date "+%H:%M:%S   %d/%m/%y"
			cd ../
			# sbatch script_gpumd.sh
		fi
	fi
fi	

