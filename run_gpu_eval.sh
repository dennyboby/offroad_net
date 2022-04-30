#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=mmseg
#SBATCH -t 30:00:00
#SBATCH -C P100
#SBATCH --mem 80G
#SBATCH --gres=gpu:1
#SBATCH -p whitehill


module load cuda11.1/toolkit/11.1.1
module load cudnn/8.1.1.33-11.2/3k5bbs63

source /home/sskodate/py39_dl/bin/activate


echo "Starting to run eval: src/offroad_path_detection.py --yaml_path=src/eval_configs/e001_psp_res50_rugd_full.yaml"
python src/offroad_path_detection.py --yaml_path=src/eval_configs/e001_psp_res50_rugd_full.yaml

echo "Starting to run eval: src/offroad_path_detection.py --yaml_path=src/eval_configs/e002_psp_res101_rugd_full.yaml"
python src/offroad_path_detection.py --yaml_path=src/eval_configs/e002_psp_res101_rugd_full.yaml

echo "Starting to run eval: src/offroad_path_detection.py --yaml_path=src/eval_configs/e003_encnet_res101_rugd_full.yaml"
python src/offroad_path_detection.py --yaml_path=src/eval_configs/e003_encnet_res101_rugd_full.yaml

echo "Starting to run eval: src/offroad_path_detection.py --yaml_path=src/eval_configs/e004_pspnet_res50_rugd_gravel.yaml"
python src/offroad_path_detection.py --yaml_path=src/eval_configs/e004_pspnet_res50_rugd_gravel.yaml

echo "Starting to run eval: src/offroad_path_detection.py --yaml_path=src/eval_configs/e005_psp_res101_rugd_gravel.yaml"
python src/offroad_path_detection.py --yaml_path=src/eval_configs/e005_psp_res101_rugd_gravel.yaml

echo "Starting to run eval: src/offroad_path_detection.py --yaml_path=src/eval_configs/e006_encnet_res101_rugd_gravel.yaml"
python src/offroad_path_detection.py --yaml_path=src/eval_configs/e006_encnet_res101_rugd_gravel.yaml
