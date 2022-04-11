#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=mmseg
#SBATCH -t 20:00:00
#SBATCH -C A100
#SBATCH --mem 80G
#SBATCH --gres=gpu:1
#SBATCH -p whitehill


module load cuda11.1/toolkit/11.1.1
module load cudnn/8.1.1.33-11.2/3k5bbs63

source /home/sskodate/py_venvs/py37_a100/bin/activate

echo "Starting to run gen_labelled_cvat_xml"
python gen_labelled_cvat_xml.py

echo "Starting to run demo_modified"
python demo_modified.py
