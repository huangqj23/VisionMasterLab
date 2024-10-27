'''
Author: huangqj23 huangquanjin24@gmail.com
Date: 2024-10-27 09:26:18
LastEditors: huangqj23 huangquanjin24@gmail.com
LastEditTime: 2024-10-27 13:42:17
FilePath: /mmyolo/projects/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from mmengine.config import read_base
with read_base():
    from .default_runtime import *
    # from .det_p5_tta import *
    from .datasets.military_vehicle import *
    
from mmyolo.models.detectors import YOLODetector
from mmyolo.models.data_preprocessors import YOLOv5DetDataPreprocessor
from mmyolo.models.backbones import YOLOv8CSPDarknet
from mmyolo.models.necks import YOLOv8PAFPN
from mmyolo.models.dense_heads import YOLOv8Head, YOLOv8HeadModule
from mmyolo.models.task_modules.coders import DistancePointBBoxCoder
from mmyolo.models.task_modules.assigners import BatchTaskAlignedAssigner
from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
from mmdet.models.losses import CrossEntropyLoss, DistributionFocalLoss


num_classes = 5
# ===============================Unmodified in most cases====================
model = dict(
    type=YOLODetector,
    data_preprocessor=dict(
        type=YOLOv5DetDataPreprocessor,
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type=YOLOv8CSPDarknet,
        arch='P5',
        last_stage_out_channels=1024,
        deepen_factor=0.33,
        widen_factor=0.5,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type=YOLOv8PAFPN,
        deepen_factor=0.33,
        widen_factor=0.5,
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type=YOLOv8Head,
        head_module=dict(
            type=YOLOv8HeadModule,
            num_classes=num_classes,
            in_channels=[256, 512, 1024],
            widen_factor=0.5,
            reg_max=16,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=[8, 16, 32]),
        prior_generator=dict(
            type=MlvlPointGenerator, offset=0.5, strides=[8, 16, 32]),
        bbox_coder=dict(type=DistancePointBBoxCoder),
        # scaled based on number of detection layers
        loss_cls=dict(
            type=CrossEntropyLoss,
            use_sigmoid=True,
            reduction='none',
            loss_weight=0.5),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=7.5,
            return_iou=False),
        loss_dfl=dict(
            type=DistributionFocalLoss, # Since the dfloss is implemented differently in the official and mmdet, we're going to divide loss_weight by 4.
            reduction='mean',
            loss_weight=1.5 / 4)),
    train_cfg=dict(
        assigner=dict(
            type=BatchTaskAlignedAssigner,
            num_classes=num_classes,
            use_ciou=True,
            topk=10,
            alpha=0.5,
            beta=6.0,
            eps=1e-9)),
    test_cfg=dict(
        # The config of multi-label for multi-class prediction.
        multi_label=True,
        # The number of boxes before NMS
        nms_pre=30000,
        score_thr=0.001,  # Threshold to filter out boxes.
        nms=dict(type='nms', iou_threshold=0.7),  # NMS type and threshold
        max_per_img=300))  # Max number of detections of each image)


