#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=mmseg
#SBATCH -t 20:00:00
#SBATCH -C P100
#SBATCH --mem 80G
#SBATCH --gres=gpu:1
#SBATCH -p short


module load cuda10.2/toolkit/10.2.89
module load cudnn/8.1.1.33-11.2/3k5bbs63

source /home/dboby/py39_dl/bin/activate

echo "Starting to run src/offroad_path_detection.py on psp_res50"
python src/offroad_path_detection.py --yaml_path=src/run_configs/r002_psp_res50_rugd_full.yaml

echo "Starting to run src/offroad_path_detection.py on psp_res101"
python src/offroad_path_detection.py --yaml_path=src/run_configs/r004_psp_res101_rugd_full.yaml

echo "Starting to run src/offroad_path_detection.py on encnet_res101"
python src/offroad_path_detection.py --yaml_path=src/run_configs/r010_encnet_res101_rugd_full.yaml
