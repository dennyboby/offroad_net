#!/bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --job-name=rugddp
#SBATCH -t 20:00:00
#SBATCH --mem 40G
#SBATCH -p short

source /home/sskodate/py39_dl/bin/activate

echo "Starting to run src/dataset_preprocesssing.py"
python src/dataset_preprocesssing.py
