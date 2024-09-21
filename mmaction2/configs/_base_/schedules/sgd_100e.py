train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=30,  # Lowered to account for tiny dataset
    val_begin=1,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=30, # Changed from 50 to 100 to match the number of epochs
        by_epoch=True,
        milestones=[20, 40], # Changed milestones to be within epoch range
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(type='SGD',
                   lr=0.005, # Changed from 0.01 to 0.005 because smaller learning rate is needed
                   momentum=0.9,
                   weight_decay=0.0001
                   ),
    clip_grad=dict(max_norm=40, norm_type=2))
