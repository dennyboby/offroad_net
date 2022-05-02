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

# echo "Starting to run src/offroad_path_detection.py on psp_res50 sample data"
# python src/offroad_path_detection.py --yaml_path=src/run_configs/r003_psp_res101_rugd.yaml

echo "Starting to run src/offroad_path_detection.py on psp_res50 for all class RUGD full"
python src/offroad_path_detection.py --yaml_path=src/run_configs/rugd_full/psp_res50_rugd_full_10k.yaml

echo "Starting to run src/offroad_path_detection.py on psp_res101 for all class RUGD full"
python src/offroad_path_detection.py --yaml_path=src/run_configs/rugd_full/psp_res101_rugd_full_10k.yaml

echo "Starting to run src/offroad_path_detection.py on encnet_res101 for all class RUGD full"
python src/offroad_path_detection.py --yaml_path=src/run_configs/rugd_full/encnet_res101_rugd_full_10k.yaml

# echo "Starting to run src/offroad_path_detection.py on psp_res50 for gravel class RUGD full"
# python src/offroad_path_detection.py --yaml_path=src/run_configs/r011_pspnet_res50_rugd_gravel.yaml

# echo "Starting to run src/offroad_path_detection.py on psp_res101 for gravel class RUGD full"
# python src/offroad_path_detection.py --yaml_path=src/run_configs/r012_psp_res101_rugd_gravel.yaml

# echo "Starting to run src/offroad_path_detection.py on encnet_res101 for gravel class RUGD full"
# python src/offroad_path_detection.py --yaml_path=src/run_configs/r013_encnet_res101_rugd_gravel.yaml
