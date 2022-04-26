import torch, torchvision
import cv2
import os
import os.path as osp
import yaml
import argparse

import mmseg
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
from mmseg.apis import set_random_seed
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

import mmcv
from mmcv import Config
import matplotlib.pyplot as plt

import constants
import utils
import format_dataset as fd


def create_cfg(data_root,
               img_dir,
               ann_dir,
               work_dir='./work_dirs/rugd_sample',
               config_path='configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py',
               pretrained_path='checkpoints/pspnet_r101-d8_512x1024_40k_cityscapes_20200604_232751-467e7cf4.pth',
               train_args=None):
    if train_args is None:
        train_args = {}
    # Main config file - base file
    cfg = Config.fromfile(config_path)

    """
    Since the given config is used to train PSPNet on the cityscapes dataset, 
    we need to modify it accordingly for our new dataset.  
    """

    # Since we use only one GPU, BN is used instead of SyncBN
    cfg.norm_cfg = dict(type='BN', requires_grad=True)
    cfg.model.backbone.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
    # modify num classes of the model in decode/auxiliary head
    cfg.model.decode_head.num_classes = 8
    cfg.model.auxiliary_head.num_classes = 8

    cfg.crop_size = (256, 256)

    # original size of the iccv09data image
    # img_scale = (320, 240)

    # Changing to original size of the rugd image
    img_scale = (688, 550)

    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **cfg.img_norm_cfg),
        dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]

    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=img_scale,
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **cfg.img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

    cfg = fd.update_data_config(cfg, data_root, img_dir, ann_dir)

    # We can still use the pre-trained Mask RCNN model though we do not need to
    # use the mask branch
    cfg.load_from = pretrained_path

    # Set up working dir to save files and logs.
    cfg.work_dir = work_dir

    cfg.runner.max_iters = train_args.get("max_iters", 20)
    cfg.log_config.interval = train_args.get("log_interval", 10)
    cfg.evaluation.interval = train_args.get("eval_interval", 10)
    cfg.checkpoint_config.interval = train_args.get("checkpoint_interval", 10)

    # Set seed to facitate reproducing the result
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    # Let's have a look at the final config used for training
    print(f'Config:\n{cfg.pretty_text}')

    return cfg


def apply_inference(model,
                    cfg,
                    dir_data=constants.rugd_dir,
                    image_name="creek_00001.png",
                    work_dir="work",
                    infer_dir="inference"):
    img_path = os.path.join(dir_data, 'images', image_name)
    img = mmcv.imread(img_path)
    model.cfg = cfg
    print("Applying inference")
    result = inference_segmentor(model, img)
    plt.figure(figsize=(8, 6))

    save_path = os.path.join(work_dir, infer_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    palette = constants.rugd_palette
    show_result_pyplot(model, img, result, palette)

    plt.savefig(os.path.join(save_path, image_name))


def apply_inference_video(model,
                          cfg,
                          dir_data=constants.rugd_dir,
                          image_name="creek_00001.png",
                          work_dir="work",
                          infer_dir="inference"):
    img_path = os.path.join(dir_data, 'images', image_name)
    img = mmcv.imread(img_path)
    model.cfg = cfg
    print("Applying inference")
    result = inference_segmentor(model, img)
    plt.figure(figsize=(8, 6))

    save_path = os.path.join(work_dir, infer_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    palette = constants.rugd_palette
    show_result_pyplot(model, img, result, palette)

    plt.savefig(os.path.join(save_path, image_name))


def setup():
    print(f"\ntorch version: {torch.__version__}"
          f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}"
          f"\nmmseg version: {mmseg.__version__}")
    pass


def train_model(data_root=constants.rugd_dir,
                do_format_data=True,
                img_dir='images',
                ann_dir='labels',
                split_dir='splits',
                true_ann_dir='annotations',
                classes=constants.rugd_classes,
                palette=constants.rugd_palette,
                work_dir='./work_dirs/rugd_sample',
                config_path='configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py',
                pretrained_path='checkpoints/pspnet_r101-d8_512x1024_40k_cityscapes_20200604_232751-467e7cf4.pth',
                train_args=None):
    """
    convert dataset annotation to semantic segmentation map
    """
    if do_format_data:
        fd.format_data(data_root=data_root,
                       img_dir=img_dir,
                       ann_dir=ann_dir,
                       split_dir=split_dir,
                       true_ann_dir=true_ann_dir,
                       classes=classes,
                       palette=palette
                       )

    cfg = create_cfg(data_root,
                     img_dir,
                     ann_dir,
                     work_dir,
                     config_path,
                     pretrained_path,
                     train_args)
    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    print("Building model")
    model = build_segmentor(cfg.model,
                            train_cfg=cfg.get('train_cfg'),
                            test_cfg=cfg.get('test_cfg'))

    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    print("Training Segmentor")
    train_segmentor(model, datasets, cfg, distributed=False, validate=True,
                    meta=dict())

    return model, cfg


def load_yaml(yaml_path):
    dict_args = {}
    with open(yaml_path, 'r') as stream:
        dictionary = yaml.safe_load(stream)
        for key, val in dictionary.items():
            dict_args[key] = val

    return dict_args


def get_classes_palette(dataset):
    classes = None
    palette = None
    if dataset == 'rugd':
        classes = constants.rugd_classes
        palette = constants.rugd_palette

    return classes, palette


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path",
                        type=str,
                        default="run_configs/r001_psp_res50_rugd.yaml",
                        help="yaml path for the run")

    args = parser.parse_args()
    return args


def main():
    """
    work_dir = './work_dirs/rugd1'
    config_path = 'configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'

    """
    setup()
    args = parse_args()
    dict_args = load_yaml(args.yaml_path)
    classes, palette = get_classes_palette(dict_args['dataset'])
    print(utils.print_dict(dict_args, "dict_args to the training"))

    model, cfg = train_model(data_root=dict_args['data_root'],
                             do_format_data=dict_args['do_format_data'],
                             img_dir=dict_args['img_dir'],
                             ann_dir=dict_args['ann_dir'],
                             split_dir=dict_args['split_dir'],
                             true_ann_dir=dict_args['true_ann_dir'],
                             classes=classes,
                             palette=palette,
                             work_dir=dict_args['work_dir'],
                             config_path=dict_args['config_path'],
                             pretrained_path=dict_args['pretrained_path'],
                             train_args=dict_args['train_args'])

    print("Training completed. Inferring.")
    apply_inference(model,
                    cfg,
                    dir_data=dict_args['data_root'],
                    work_dir=dict_args['work_dir'],
                    image_name="creek_00001.png")


if __name__ == '__main__':
    main()