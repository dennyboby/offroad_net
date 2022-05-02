<div align="center">
  <img src="resources/offroadnet_logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>
</div>
<br />

[comment]: <> ([![PyPI - Python Version]&#40;https://img.shields.io/pypi/pyversions/mmsegmentation&#41;]&#40;https://pypi.org/project/mmsegmentation/&#41;)

[comment]: <> ([![PyPI]&#40;https://img.shields.io/pypi/v/mmsegmentation&#41;]&#40;https://pypi.org/project/mmsegmentation&#41;)

[comment]: <> ([![docs]&#40;https://img.shields.io/badge/docs-latest-blue&#41;]&#40;https://mmsegmentation.readthedocs.io/en/latest/&#41;)

[comment]: <> ([![badge]&#40;https://github.com/open-mmlab/mmsegmentation/workflows/build/badge.svg&#41;]&#40;https://github.com/open-mmlab/mmsegmentation/actions&#41;)

[comment]: <> ([![codecov]&#40;https://codecov.io/gh/open-mmlab/mmsegmentation/branch/master/graph/badge.svg&#41;]&#40;https://codecov.io/gh/open-mmlab/mmsegmentation&#41;)

[comment]: <> ([![license]&#40;https://img.shields.io/github/license/open-mmlab/mmsegmentation.svg&#41;]&#40;https://github.com/open-mmlab/mmsegmentation/blob/master/LICENSE&#41;)

[comment]: <> ([![issue resolution]&#40;https://isitmaintained.com/badge/resolution/open-mmlab/mmsegmentation.svg&#41;]&#40;https://github.com/open-mmlab/mmsegmentation/issues&#41;)

[comment]: <> ([![open issues]&#40;https://isitmaintained.com/badge/open/open-mmlab/mmsegmentation.svg&#41;]&#40;https://github.com/open-mmlab/mmsegmentation/issues&#41;)

Documentation: https://mmsegmentation.readthedocs.io/

[comment]: <> (English | [简体中文]&#40;README_zh-CN.md&#41;)

## Introduction

OffroadNet is based on the MMSegmentation library which is an open source semantic segmentation toolbox based on PyTorch and is a part of the OpenMMLab project.

The master branch works with **PyTorch 1.5+**.

The inference on the **creek** video sequence as part of the test split of RUGD Dataset.
![demo image](resources/inference_gifs/creek.gif)

The inference on the **trail-7** video sequence as part of the test split of RUGD Dataset.
![demo image](resources/inference_gifs/trail-7.gif)

The inference on the **park-1** video sequence as part of the test split of RUGD Dataset.
![demo image](resources/inference_gifs/park-1.gif)

The inference on the **trail-13** video sequence as part of the test split of RUGD Dataset.
![demo image](resources/inference_gifs/trail-13.gif)

### Running the scripts

- **Calculate Image Statistics**

```shell
cd offroad_net
source py39_dl/bin/activate

echo "Starting to run: python src/calc_image_stats.py "
python  src/calc_image_stats.py
```

- **Training multiple models in WPI Turing cluster**

```shell
module load cuda11.1/toolkit/11.1.1
module load cudnn/8.1.1.33-11.2/3k5bbs63

source py39_dl/bin/activate

echo "Starting to run src/offroad_path_detection.py on psp_res50 for all class RUGD full"
python src/offroad_path_detection.py --yaml_path=src/run_configs/rugd_full/psp_res50_rugd_full_10k.yaml

echo "Starting to run src/offroad_path_detection.py on encnet_res101 for all class RUGD full"
python src/offroad_path_detection.py --yaml_path=src/run_configs/rugd_full/encnet_res101_rugd_full_10k.yaml
```
- **Data pre-processing**

```shell
echo "Starting to run src/dataset_preprocesssing.py"
python src/dataset_preprocesssing.py
```
- **Sample yaml file**

```yaml
data_root: "RUGD/RUGD_full"
dataset: 'rugd'
do_format_data: false
img_dir: 'images'
ann_dir: 'labels'
split_dir: 'splits'
true_ann_dir: 'annotations'
work_dir: './work_dirs/rugd_full_10k/encnet_r101-d8'
config_path: 'configs/offroadnet/encnet_r101-d8_512x1024_80k_cityscapes.py'
pretrained_path: 'checkpoints/encnet_r101-d8_512x1024_80k_cityscapes_20200622_003555-1de64bec.pth'
train_args:
  max_iters: 10000
  log_interval: 1000
  eval_interval: 5000
  checkpoint_interval: 5000
img_scale : [688, 550]
crop : [256, 256]
mean : [102.9630, 102.9438, 102.3976]
std : [69.8548, 70.3098, 70.9376]
img_ratios : [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
ratio_range : [0.5, 2.0]
cat_max_ratio : 0.75
seg_pad_val: 255
pad_val : 0
flip_ratio : 0.5
suffix : '.png'
inf_img_dir : 'inference_images'
list_sub_dirs : ["test1"]

```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Results

<table>
    <tr>
        <td>Backbone</td>
        <td>DecodeHead</td>
        <td>AuxillaryHead</td>
        <td>Metrics</td>
        <td>A</td>
        <td>G</td>
        <td>R</td>
        <td>C</td>
        <td>O</td>
        <td>OG</td>
    </tr>
    <tr>
        <td>ResNetV1c 50</td>
        <td>PSPHead</td>
        <td>FCNHead</td>
        <td>mIoU</td>
        <td>58.66</td>
        <td>81.24</td>
        <td>67.43</td>
        <td>39.78</td>
        <td>37.14</td>
        <td>80.73</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td>mAcc</td>
        <td>93.72</td>
        <td>\textbf{90.31}</td>
        <td>\textbf{96.42}</td>
        <td>41.22</td>
        <td>43.24</td>
        <td>87.04</td>
    </tr>
    <tr>
        <td>ResNetV1c 101</td>
        <td>OffRoadHead</td>
        <td>FCNHead</td>
        <td>mIoU</td>
        <td>\textbf{59.64}</td>
        <td>\textbf{92.37}</td>
        <td>\textbf{85.24}</td>
        <td>\textbf{43.99}</td>
        <td>\textbf{42.85}</td>
        <td>\textbf{82.18}</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td>mAcc</td>
        <td>93.47</td>
        <td>88.66</td>
        <td>94.45</td>
        <td>\textbf{63.73}</td>
        <td>\textbf{48.87}</td>
        <td>88.73</td>
    </tr>
    <tr>
        <td>ResNetV1c 101</td>
        <td>EncHead</td>
        <td>FCNHead</td>
        <td>mIoU</td>
        <td>54.87</td>
        <td>78.78</td>
        <td>79.3</td>
        <td>34.39</td>
        <td>40.52</td>
        <td>80.09</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td>mAcc</td>
        <td>\textbf{94.0}</td>
        <td>87.18</td>
        <td>93.58</td>
        <td>40.51</td>
        <td>48.78</td>
        <td>\textbf{90.18}</td>
    </tr>
</table>

## Installation

Please refer to [get_started.md](docs/en/get_started.md#installation) for installation and [dataset_prepare.md](docs/en/dataset_prepare.md#prepare-datasets) for dataset preparation.


## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{offroadnet2022,
    title={{OffroadNet}: Based on MMSegmentation Toolbox},
    author={Abhay Chhagan Karade, Denny Boby, Sumukh Sreenivasarao Balakrishna, Shreedhar Kodate},
    howpublished = {\url{https://github.com/AbhayKarade/offroad_net}},
    year={2022}
}
```


## Creating gifs
```shell
ffmpeg -y -i file.mp4 -vf palettegen palette.png

ffmpeg -y -i file.mp4 -i palette.png -filter_complex paletteuse -r 10 -s 480x340 file.gif
```

