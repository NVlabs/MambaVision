_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/ade20k_640x640.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (640, 640)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MM_mamba_vision',
        out_indices=(0, 1, 2, 3),
        pretrained="/ckpts/mambavision_L3_21k_700m_512.pth.tar",
        depths = (3, 3, 20, 10),
        num_heads = (4, 8, 16, 32),
        window_size = (8, 8, 64, 32),
        dim = 256,
        in_dim = 64,
        mlp_ratio = 4,
        drop_path_rate = 0.8,
        norm_layer="ln2d",
        layer_scale = 1e-5,
        ),
    decode_head=dict(in_channels=[256, 512, 1024, 2048], num_classes=150),
    auxiliary_head=dict(in_channels=1024, num_classes=150))

optim_wrapper = dict(
    _delete_=True,
    # type='AmpOptimWrapper', don't use amp for mambavision_L3_21k_700m_512 as it can cause instability
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00008, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=5000)

# This model is trained on 2 nodes, 16 GPUs, 1 image per GPU
train_dataloader = dict(batch_size=1)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
