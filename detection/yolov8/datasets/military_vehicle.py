'''
Author: huangqj23 huangquanjin24@gmail.com
Date: 2024-10-27 09:22:27
LastEditors: huangqj23 huangquanjin24@gmail.com
LastEditTime: 2024-10-27 13:59:26
FilePath: /mmyolo/projects/yolov8/datasets/military_vehicle.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE

'''

from mmdet.datasets.transforms import Albu, RandomFlip, PackDetInputs
from mmyolo.datasets.transforms import YOLOv5HSVRandomAug, YOLOv5RandomAffine, YOLOv5KeepRatioResize, Mosaic, \
    LetterResize, LoadAnnotations
from mmyolo.datasets import yolov5_collate
from mmcv.transforms import LoadImageFromFile
from mmengine.dataset.sampler import DefaultSampler
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
from mmengine.runner.loops import EpochBasedTrainLoop, TestLoop, ValLoop
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.hooks.ema_hook import EMAHook
from mmyolo.models.layers import ExpMomentumEMA
from mmyolo.engine.optimizers import YOLOv5OptimizerConstructor
from mmyolo.engine.hooks import YOLOv5ParamSchedulerHook
from mmdet.engine.hooks import PipelineSwitchHook
from mmdet.evaluation.metrics import CocoMetric
from torch.optim.sgd import SGD
from military_vehicle_dataset import MilitaryVehicleDataset


backend_args = None
# ========================Frequently modified parameters======================
# -----data related-----
data_root = '/root/MVRSD_dataset/'  # Root path of data
# Path of train annotation file
train_ann_file = 'coco/train_military.json'
train_data_prefix = 'images/train/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'coco/val_military.json'
val_data_prefix = 'images/val/'  # Prefix of val image path

# -----train val related-----
max_epochs = 500  # Maximum training epochs
# Disable mosaic augmentation for final 10 epochs (stage 2)
close_mosaic_epochs = 10


# ========================Possible modified parameters========================
# -----data related-----
img_scale = (640, 640)  # width, height
# Dataset type, this will be used to define the dataset
dataset_type = MilitaryVehicleDataset

# Config of batch shapes. Only on val.
# We tested YOLOv8-m will get 0.02 higher than not using it.
batch_shapes_cfg = None
# You can turn on `batch_shapes_cfg` by uncommenting the following lines.
# batch_shapes_cfg = dict(
#     type='BatchShapePolicy',
#     batch_size=val_batch_size_per_gpu,
#     img_size=img_scale[0],
#     # The image scale of padding should be divided by pad_size_divisor
#     size_divisor=32,
#     # Additional paddings for pixel scale
#     extra_pad_ratio=0.5)

# -----model related-----
num_det_layers = 3  # The number of model output scales

# -----train val related-----
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
# YOLOv5RandomAffine aspect ratio of width and height thres to filter bboxes
max_aspect_ratio = 100

# TODO: Automatically scale loss_weight based on number of detection layers
# Single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]

pre_transform = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True)
]

last_transform = [
    # dict(
    #     type=Albu,
    #     transforms=albu_train_transforms,
    #     bbox_params=dict(
    #         type='BboxParams',
    #         format='pascal_voc',
    #         label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
    #     keymap={
    #         'img': 'image',
    #         'gt_bboxes': 'bboxes'
    #     }),
    dict(type=YOLOv5HSVRandomAug),
    dict(type=RandomFlip, prob=0.5),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_pipeline = [
    *pre_transform,
    dict(
        type=Mosaic,
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type=YOLOv5RandomAffine,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    *last_transform
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type=YOLOv5KeepRatioResize, scale=img_scale),
    dict(
        type=LetterResize,
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type=YOLOv5RandomAffine,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        border_val=(114, 114, 114)), *last_transform
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True, # must be False when num_workers = 0
    pin_memory=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=yolov5_collate),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))

test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=YOLOv5KeepRatioResize, scale=img_scale),
    dict(
        type=LetterResize,
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type=LoadAnnotations, with_bbox=True, _scope_='mmdet'),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg))

test_dataloader = val_dataloader

param_scheduler = None
optim_wrapper = dict(
    type=OptimWrapper,
    clip_grad=dict(max_norm=10.0),
    optimizer=dict(
        type=SGD,
        lr=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
        batch_size_per_gpu=16),
    constructor=YOLOv5OptimizerConstructor)

default_hooks = dict(
    param_scheduler=dict(
        type=YOLOv5ParamSchedulerHook,
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=max_epochs),
    checkpoint=dict(
        type=CheckpointHook,
        interval=10,
        save_best='auto',
        max_keep_ckpts=2))

custom_hooks = [
    dict(
        type=EMAHook,
        ema_type=ExpMomentumEMA,
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type=PipelineSwitchHook,
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2)
]

val_evaluator = dict(
    type=CocoMetric,
    proposal_nums=(100, 1, 10),
    ann_file=data_root + val_ann_file,
    metric='bbox')
test_evaluator = val_evaluator

train_cfg = dict(
    type=EpochBasedTrainLoop,
    max_epochs=max_epochs,
    val_interval=10,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                        1)])

val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)


