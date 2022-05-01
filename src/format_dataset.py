import os.path as osp
import numpy as np
from PIL import Image
import mmcv
import cv2
import matplotlib.pyplot as plt

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

import constants

#test
# This code just scans the directory and creates a map based on the class color maps
# This need not be run for RUGD as we already have what we need
# list_file_names = mmcv.scandir(osp.join(iccv_data_root, ann_dir), suffix='.regions.txt')
# for file in list_file_names:
#     seg_map = np.loadtxt(osp.join(iccv_data_root, ann_dir, file)).astype(np.uint8)
#     seg_img = Image.fromarray(seg_map).convert('P')
#     seg_img.putpalette(np.array(iccv_palette, dtype=np.uint8))
#     seg_img.save(osp.join(iccv_data_root, ann_dir, file.replace('.regions.txt',
#                                                                 '.png')))

def transform_seg_map_mode_p(data_root, ann_dir, true_ann_dir, palette):
    # Add code to change the seg map to the required format
    for file in mmcv.scandir(osp.join(data_root, true_ann_dir), suffix='.png'):
        seg_map = cv2.imread(osp.join(data_root, true_ann_dir, file))
        seg_map_new = np.zeros((seg_map.shape[0], seg_map.shape[1]), dtype=np.uint8)
        for i in range(seg_map.shape[0]):
            for j in range(seg_map.shape[1]):
                seg_map_new[i, j] = palette.index(list(seg_map[i, j, ::-1]))
        seg_img = Image.fromarray(seg_map_new).convert('P')
        seg_img.putpalette(np.array(palette, dtype=np.uint8))
        seg_img.save(osp.join(data_root, ann_dir, file))


def split_dataset(split_dir, data_root, ann_dir):
    # split train/val set randomly
    mmcv.mkdir_or_exist(osp.join(data_root, split_dir))

    train_percent = 0.8
    mmcv.mkdir_or_exist(osp.join(data_root, split_dir))
    filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
        osp.join(data_root, ann_dir), suffix='.png')]
    np.random.shuffle(filename_list)

    with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
        # select first 80% data as train set
        train_length = int(len(filename_list) * train_percent)
        f.writelines(line + '\n' for line in filename_list[:train_length])

    with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
        # select last 1/5 as train set
        f.writelines(line + '\n' for line in filename_list[train_length:])


"""
After downloading the data, we need to implement `load_annotations` function in 
the new dataset class `StanfordBackgroundDataset`.
"""


# @DATASETS.register_module()
# class StanfordBackgroundDataset(CustomDataset):
#     CLASSES = classes
#     PALETTE = palette
#
#     def __init__(self, split, **kwargs):
#         super().__init__(img_suffix='.jpg', seg_map_suffix='.png',
#                          split=split, **kwargs)
#         assert osp.exists(self.img_dir) and self.split is not None

@DATASETS.register_module()
class RUGDDataset(CustomDataset):
    CLASSES = constants.rugd_classes
    PALETTE = constants.rugd_palette

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png',
                         split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None


@DATASETS.register_module()
class OffRoadDataset(CustomDataset):
    CLASSES = constants.offroad_classes
    PALETTE = constants.offroad_palette

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png',
                         split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None


def get_dataset_type(dataset):
    dict_dataset_type = {
        'rugd': 'RUGDDataset',
        'offroad': 'OffRoadDataset'
    }
    return dict_dataset_type[dataset]


"""
Since the given config is used to train PSPNet on the cityscapes dataset, 
we need to modify it accordingly for our new dataset.  
"""


def update_data_config(cfg,
                       data_root,
                       img_dir,
                       ann_dir,
                       dataset_type='RUGDDataset',
                       **kwargs):
    """

    """
    cfg.dataset_type = dataset_type
    cfg.data_root = data_root

    cfg.data.samples_per_gpu = kwargs.get("samples_per_gpu", 8)
    cfg.data.workers_per_gpu = kwargs.get("workers_per_gpu", 8)

    cfg.img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

    # TODO: Study this, and change based on our dataset
    #  mean and std. dev of the data
    # cfg.img_norm_cfg = dict(
    #     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.data_root = cfg.data_root
    cfg.data.train.img_dir = img_dir
    cfg.data.train.ann_dir = ann_dir
    cfg.data.train.pipeline = cfg.train_pipeline
    cfg.data.train.split = 'splits/train.txt'

    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.data_root = cfg.data_root
    cfg.data.val.img_dir = img_dir
    cfg.data.val.ann_dir = ann_dir
    cfg.data.val.pipeline = cfg.test_pipeline
    cfg.data.val.split = 'splits/val.txt'

    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.img_dir = img_dir
    cfg.data.test.ann_dir = ann_dir
    cfg.data.test.pipeline = cfg.test_pipeline
    cfg.data.test.split = 'splits/val.txt'

    return cfg


def setup():
    pass


def format_data(data_root=constants.rugd_dir,
                img_dir='images',
                ann_dir='labels',
                split_dir='splits',
                true_ann_dir='annotations',
                classes=constants.rugd_classes,
                palette=constants.rugd_palette):
    """
    convert dataset annotation to semantic segmentation map
    """
    transform_seg_map_mode_p(data_root, ann_dir, true_ann_dir, palette)
    split_dataset(split_dir, data_root, ann_dir)


def main():
    setup()
    format_data()


if __name__ == '__main__':
    main()
