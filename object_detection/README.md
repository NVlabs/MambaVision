# Object detection with MambaVision

## Detection Results + Models 

<table>
  <tr>
    <th>Backbone</th>
    <th>Detector</th>
    <th>Lr Schd</th>
    <th>box mAP</th>
    <th>mask mAP</th>
    <th>#Params(M)</th>
    <th>FLOPs(G)</th>
    <th>Config</th>
    <th>Log</th>
    <th>Model Ckpt</th>
  </tr>

<tr>
    <td><a href="https://huggingface.co/nvidia/MambaVision-T-1K">MambaVision-T-1K</a></td>
    <td>Cascade Mask R-CNN</td>
    <td>3x</td>
    <td>51.1</td>
    <td>44.3</td>
    <td>86</td>
    <td>740</td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/object_detection/configs/mamba_vision/cascade_mask_rcnn_mamba_vision_tiny_3x_coco.py">config</a></td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/object_detection/tools/work_dirs/cascade_mask_rcnn_mamba_vision_tiny_3x_coco/20250607_142007/20250607_142007.log">log</a></td>
    <td><a href="https://huggingface.co/nvidia/cascade_mask_rcnn_mamba_vision_tiny_3x_coco/resolve/main/cascade_mask_rcnn_mamba_vision_tiny_3x_coco.pth">model</a></td>
</tr>

<tr>
    <td><a href="https://huggingface.co/nvidia/MambaVision-S-1K">MambaVision-S-1K</a></td>
    <td>Cascade Mask R-CNN</td>
    <td>3x</td>
    <td>52.3</td>
    <td>45.2</td>
    <td>108</td>
    <td>828</td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/object_detection/configs/mamba_vision/cascade_mask_rcnn_mamba_vision_small_3x_coco.py">config</a></td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/object_detection/tools/work_dirs/cascade_mask_rcnn_mamba_vision_small_3x_coco/20250607_144612/20250607_144612.log">log</a></td>
    <td><a href="https://huggingface.co/nvidia/cascade_mask_rcnn_mamba_vision_tiny_3x_coco/resolve/main/cascade_mask_rcnn_mamba_vision_tiny_3x_coco.pth">model</a></td>
</tr>

<tr>
    <td><a href="https://huggingface.co/nvidia/MambaVision-B-1K">MambaVision-B-1K</a></td>
    <td>Cascade Mask R-CNN</td>
    <td>3x</td>
    <td>52.8</td>
    <td>45.7</td>
    <td>145</td>
    <td>964</td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/object_detection/configs/mamba_vision/cascade_mask_rcnn_mamba_vision_base_3x_coco.py">config</a></td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/object_detection/tools/work_dirs/cascade_mask_rcnn_mamba_vision_base_3x_coco/20250607_145939/20250607_145939.log">log</a></td>
    <td><a href="https://huggingface.co/nvidia/cascade_mask_rcnn_mamba_vision_base_3x_coco/resolve/main/cascade_mask_rcnn_mamba_vision_base_3x_coco.pth">model</a></td>
</tr>


</table>

## Installation

Our object detection code is built upon on top of the popular [MMDetection](https://github.com/open-mmlab/mmdetection) framework. 

### 1. Verify MambaVision Backbone Support

Before proceeding, ensure your environment is configured to run MambaVision pre-trained backbones. For detailed prerequisites and setup instructions, see the MambaVision [installation guide](https://github.com/NVlabs/MambaVision/tree/main#Installation).

### 2. Install Dependencies

MambaVision builds on top of MMDetection and relies on the following packages:

```bash
pip install \
  mmengine==0.10.1 \
  mmcv==2.1.0 \
  opencv-python-headless \
  mmdet==3.3.0 \
  mmsegmentation==1.2.2 \
  mmpretrain==1.2.0
```

> Tip: You can also pin these into a `requirements.txt` for reproducibility.

### 3. Verify Your Environment

Ensure your system meets the recommended version requirements:

| Component   | Version     |
| ----------- | ----------- |
| PyTorch     | 2.4.1+cu124 |
| CUDA        | 12.4        |
| OpenCV      | 4.10.0      |
| MMCV        | 2.1.0       |
| MMDetection | 3.3.0       |
| MMEngine    | 0.10.1      |

### 4. Further Reading

For complete setup instructions and troubleshooting, see the MMDetection [installation guide](https://mmdetection.readthedocs.io/en/latest/get_started.html).


## Training

Start your experiments with **multi‑GPU training** using our Slurm script on 8 GPUs. First, browse the available configuration files for different MambaVision models [here](https://github.com/NVlabs/MambaVision/tree/main/object_detection/configs/mamba_vision)

Once you’ve chosen a config (e.g., `cascade_mask_rcnn_mamba_vision_tiny_3x_coco.py`), launch training with:

```bash
# multi‑GPU training (8 GPUs)
srun --gres=gpu:8 python tools/train.py configs/mamba_vision/<CONFIG_FILE>.py
```

You can also use our slurm [train script](https://github.com/NVlabs/MambaVision/blob/main/object_detection/tools/cascade_mask_rcnn_mamba_vision_base_3x.sh).


If you’d rather run on a single GPU—for quick tests or debugging—use:

```bash
# single‑GPU training
env CUDA_VISIBLE_DEVICES=0 python tools/train.py ${CONFIG}
```


## Evaluation

For evaluation, we recommend using our slurm [test script](https://github.com/NVlabs/MambaVision/blob/main/object_detection/tools/test.sh) for inference with 8 GPUs. 


We provide both multi‑GPU and single‑GPU inference options:

### Multi‑GPU Inference

Run our Slurm [test script](https://github.com/NVlabs/MambaVision/blob/main/object_detection/tools/test.sh) on 8 GPUs for high‑throughput evaluation:

```bash
# multi‑GPU inference (8 GPUs)
bash tools/test.sh
```

### Single‑GPU Inference

For quick evaluation or debugging on a single GPU, use the standard test tool:

```bash
# single‑GPU inference
env CUDA_VISIBLE_DEVICES=0 \
  python tools/test.py \
    configs/mamba_vision/<CONFIG_FILE>.py \
    <CHECKPOINT_FILE>.pth \
    --eval bbox segm
```


## Data Preparation

Follow these steps to prepare the COCO dataset for object detection and instance segmentation:

1. **Download the dataset**

```bash
cd <path-to-mmdetection-root>
mkdir -p data/coco && cd data/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```

2. **Verify directory structure**

```
data/coco/
├── annotations/
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   └── panoptic_train2017.json  # optional for panoptic segmentation
├── train2017/
└── val2017/
```

3. **Configure your MMDetection config**

In your config file, set the `data_root` and annotation/img paths:

```python
data_root = 'data/coco/'

data = dict(
    train=dict(
        img_prefix=data_root + 'train2017/',
        ann_file=data_root + 'annotations/instances_train2017.json'),
    val=dict(
        img_prefix=data_root + 'val2017/',
        ann_file=data_root + 'annotations/instances_val2017.json'),
    test=dict(
        img_prefix=data_root + 'val2017/',
        ann_file=data_root + 'annotations/instances_val2017.json')
)
```


## Citation

If you find MambaVision to be useful for your work, please consider citing our paper: 

```
@inproceedings{hatamizadeh2025mambavision,
  title={Mambavision: A hybrid mamba-transformer vision backbone},
  author={Hatamizadeh, Ali and Kautz, Jan},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={25261--25270},
  year={2025}
}
```

## Licenses

Copyright © 2025, NVIDIA Corporation. All rights reserved.

This work is made available under the NVIDIA Source Code License-NC. Click [here](LICENSE) to view a copy of this license.

The pre-trained models are shared under [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

For license information regarding the MMDetection repository, please refer to its [repository](https://github.com/open-mmlab/mmdetection).

For COCO dataset, please refer to its [terms of use](https://cocodataset.org/#termsofuse).


## Acknowledgement
The detection repository is built on top of the [MMDetection](https://github.com/open-mmlab/mmdetection) and inspired by [VMamba](https://github.com/MzeroMiko/VMamba/tree/main/detection). 
