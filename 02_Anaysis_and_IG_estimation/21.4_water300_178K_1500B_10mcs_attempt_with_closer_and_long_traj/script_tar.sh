#!/bin/bash
#SBATCH -J create_dataset
#SBATCH --nodes=1
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

tar -cf pickles_imb_E10_dci00.tar  pickles_imb_E10_dci200 pickles_imb_E10_dci300 pickles_imb_E10_dci400 pickles_imb_E10_dci500
