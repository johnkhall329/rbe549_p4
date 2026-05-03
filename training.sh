#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64g
#SBATCH -p academic
#SBATCH -t 30:00:00
#SBATCH --gres=gpu:1

RUN_NAME=$1

module load python/3.10.17
module load cuda/12.8.0/4fdo42o
module load ffmpeg

export LD_LIBRARY_PATH=$(dirname $(which ffmpeg))/../lib:$LD_LIBRARY_PATH
echo LD_LIBRARY_PATH


python3 -m venv pytorch_venv
source pytorch_venv/bin/activate
pip3 install -r Phase2/requirements.txt


python3 Phase2/Code/train.py --run_name $RUN_NAME 
