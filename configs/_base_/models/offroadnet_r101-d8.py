# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    # Args:
    #     depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
    #     in_channels (int): Number of input image channels. Default: 3.
    #     stem_channels (int): Number of stem channels. Default: 64.
    #     base_channels (int): Number of base channels of res layer. Default: 64.
    #     num_stages (int): Resnet stages, normally 4. Default: 4.
    #     strides (Sequence[int]): Strides of the first block of each stage.
    #         Default: (1, 2, 2, 2).
    #     dilations (Sequence[int]): Dilation of each stage.
    #         Default: (1, 1, 1, 1).
    #     out_indices (Sequence[int]): Output from which stages.
    #         Default: (0, 1, 2, 3).
    #     style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
    #         layer is the 3x3 conv layer, otherwise the stride-two layer is
    #         the first 1x1 conv layer. Default: 'pytorch'.
    #     deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
    #         Default: False.
    #     avg_down (bool): Use AvgPool instead of stride conv when
    #         downsampling in the bottleneck. Default: False.
    #     frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
    #         -1 means not freezing any parameters. Default: -1.
    #     conv_cfg (dict | None): Dictionary to construct and config conv layer.
    #         When conv_cfg is None, cfg will be set to dict(type='Conv2d').
    #         Default: None.
    #     norm_cfg (dict): Dictionary to construct and config norm layer.
    #         Default: dict(type='BN', requires_grad=True).
    #     norm_eval (bool): Whether to set norm layers to eval mode, namely,
    #         freeze running stats (mean and var). Note: Effect on Batch Norm
    #         and its variants only. Default: False.
    #     dcn (dict | None): Dictionary to construct and config DCN conv layer.
    #         When dcn is not None, conv_cfg must be None. Default: None.
    #     stage_with_dcn (Sequence[bool]): Whether to set DCN conv for each
    #         stage. The length of stage_with_dcn is equal to num_stages.
    #         Default: (False, False, False, False).
    #     plugins (list[dict]): List of plugins for stages, each dict contains:

    #         - cfg (dict, required): Cfg dict to build plugin.

    #         - position (str, required): Position inside block to insert plugin,
    #         options: 'after_conv1', 'after_conv2', 'after_conv3'.

    #         - stages (tuple[bool], optional): Stages to apply plugin, length
    #         should be same as 'num_stages'.
    #         Default: None.
    #     multi_grid (Sequence[int]|None): Multi grid dilation rates of last
    #         stage. Default: None.
    #     contract_dilation (bool): Whether contract first dilation of each layer
    #         Default: False.
    #     with_cp (bool): Use checkpoint or not. Using checkpoint will save some
    #         memory while slowing down the training speed. Default: False.
    #     zero_init_residual (bool): Whether to use zero init for last norm layer
    #         in resblocks to let them behave as identity. Default: True.
    #     pretrained (str, optional): model pretrained path. Default: None.
    #     init_cfg (dict or list[dict], optional): Initialization config dict.
    #         Default: None.
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='ResNetV1c', # Different models present in mmseg/models/backbones/
        depth=101),
        # num_stages=4,
        # out_indices=(0, 1, 2, 3),
        # dilations=(1, 1, 2, 4),
        # strides=(1, 2, 1, 1),
        # norm_cfg=norm_cfg,
        # norm_eval=False,
        # style='pytorch',
        # contract_dilation=True),
    decode_head=dict(
        type='OffRoadHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
