model = dict(
    type='RecognizerZelda',
    backbone=dict(type='BackBoneZelda'),
    cls_head=dict(
        type='ClsHeadZelda',
        num_classes=2,
        in_channels=128,
        average_clips='prob'),
    data_preprocessor = dict(
        type='DataPreprocessorZelda',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]))