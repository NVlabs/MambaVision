_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MM_mamba_vision',
        out_indices=(0, 1, 2, 3),
        pretrained="/ckpts/mambavision_tiny_1k.pth.tar",
        depths = (1, 3, 8, 4),
        num_heads = (2, 4, 8, 16),
        window_size = (8, 8, 64, 32),
        dim = 80,
        in_dim = 32,
        mlp_ratio = 4,
        drop_path_rate = 0.3,
        norm_layer="ln2d",
        layer_scale=None,
        ),
    decode_head=dict(in_channels=[80, 160, 320, 640], num_classes=150),
    auxiliary_head=dict(in_channels=320, num_classes=150))

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00005, betas=(0.9, 0.999), weight_decay=0.01),
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

# This model is trained on 2 nodes, 16 GPUs, 1 image per GPU
train_dataloader = dict(batch_size=1)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
