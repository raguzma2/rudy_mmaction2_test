ann_file_test = 'C:\\Users\\rodol\\Desktop\\basic_salsa_step_data\\Computer_vision_model\\val_video.txt'
ann_file_train = 'C:\\Users\\rodol\\Desktop\\basic_salsa_step_data\\Computer_vision_model\\train_video.txt'
ann_file_val = 'C:\\Users\\rodol\\Desktop\\basic_salsa_step_data\\Computer_vision_model\\val_video.txt'
auto_scale_lr = dict(base_batch_size=64, enable=False)
data_root = 'C:\\Users\\rodol\\Desktop\\basic_salsa_step_data\\Computer_vision_model\\train'
data_root_val = 'C:\\Users\\rodol\\Desktop\\basic_salsa_step_data\\Computer_vision_model\\val'
dataset_type = 'VideoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=5, max_keep_ckpts=5, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(io_backend='disk')
launcher = 'none'
load_from = 'C:\\\\Users\\\\rodol\\\\Desktop\\\\model_training_test\\\\epoch_30.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        conv1_kernel=(
            5,
            7,
            7,
        ),
        conv1_stride_t=2,
        conv_cfg=dict(type='Conv3d'),
        depth=50,
        inflate=(
            (
                1,
                1,
                1,
            ),
            (
                1,
                0,
                1,
                0,
            ),
            (
                1,
                0,
                1,
                0,
                1,
                0,
            ),
            (
                0,
                1,
                0,
            ),
        ),
        norm_eval=False,
        pool1_stride_t=2,
        pretrained='torchvision://resnet50',
        pretrained2d=True,
        type='ResNet3d',
        zero_init_residual=False),
    cls_head=dict(
        average_clips='prob',
        dropout_ratio=0.5,
        in_channels=2048,
        init_std=0.01,
        num_classes=2,
        spatial_type='avg',
        type='I3DHead'),
    data_preprocessor=dict(
        format_shape='NCTHW',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='ActionDataPreprocessor'),
    type='Recognizer3D')
optim_wrapper = dict(
    clip_grad=dict(max_norm=40, norm_type=2),
    optimizer=dict(lr=0.005, momentum=0.9, type='SGD', weight_decay=0.0001))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=30,
        gamma=0.1,
        milestones=[
            20,
            40,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        'C:\\Users\\rodol\\Desktop\\basic_salsa_step_data\\Computer_vision_model\\val_video.txt',
        data_prefix=dict(
            video=
            'C:\\Users\\rodol\\Desktop\\basic_salsa_step_data\\Computer_vision_model\\val'
        ),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=32,
                frame_interval=2,
                num_clips=10,
                test_mode=True,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=256, type='ThreeCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='AccMetric')
test_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(
        clip_len=32,
        frame_interval=2,
        num_clips=10,
        test_mode=True,
        type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=256, type='ThreeCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
train_cfg = dict(
    max_epochs=30, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file=
        'C:\\Users\\rodol\\Desktop\\basic_salsa_step_data\\Computer_vision_model\\train_video.txt',
        data_prefix=dict(
            video=
            'C:\\Users\\rodol\\Desktop\\basic_salsa_step_data\\Computer_vision_model\\train'
        ),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=32,
                frame_interval=2,
                num_clips=1,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(
                input_size=224,
                max_wh_scale_gap=0,
                random_crop=False,
                scales=(
                    1,
                    0.8,
                ),
                type='MultiScaleCrop'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(flip_ratio=0.5, type='Flip'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=32, frame_interval=2, num_clips=1, type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(
        input_size=224,
        max_wh_scale_gap=0,
        random_crop=False,
        scales=(
            1,
            0.8,
        ),
        type='MultiScaleCrop'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file=
        'C:\\Users\\rodol\\Desktop\\basic_salsa_step_data\\Computer_vision_model\\val_video.txt',
        data_prefix=dict(
            video=
            'C:\\Users\\rodol\\Desktop\\basic_salsa_step_data\\Computer_vision_model\\val'
        ),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=32,
                frame_interval=2,
                num_clips=1,
                test_mode=True,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='AccMetric')
val_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True,
        type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs\\i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb'
