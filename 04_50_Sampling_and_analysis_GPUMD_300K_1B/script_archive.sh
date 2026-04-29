#!/bin/bash
#SBATCH -J disk-usage
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
##SBATCH --gres=gpu:1
##SBATCH --mem=512GB
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=qos_prio
#SBATCH --err=%x_%j.err
#SBATCH --out=%x_%j.out
#SBATCH --account=ICT25_RECOVERY_0

tar -czf dump.xyz.targz dump.xyz
