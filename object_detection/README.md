# Object detection with MambaVision

Our object detection code is built upon on top of the popular [MMDetection](https://github.com/open-mmlab/mmdetection) framework. 


## Installation 

Assuming an environment that already works for MambaVision, you need to install the following packages for downstream tasks such as object detection and semantic segmentation:

```
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless mmdet==3.3.0 mmsegmentation==1.2.2 mmpretrain==1.2.0
```

The following represents a compatible environment for running the model:

```
Pytorch: 2.4.1+cu124

CUDA: 12.4

OpenCV: 4.10.0

MMCV: 2.1.0

MMdet: 3.3.0

MMEngine: 0.10.1
```

For futher information, please see the MMDetection [installation guide](https://mmdetection.readthedocs.io/en/latest/get_started.html).

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
    <th>HF</th>
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
    <td><a href="https://huggingface.co/nvidia/cascade_mask_rcnn_mamba_vision_tiny_3x_coco">HF</a></td>
    <td><a href="https://huggingface.co/nvidia/cascade_mask_rcnn_mamba_vision_tiny_3x_coco/resolve/main/cascade_mask_rcnn_mamba_vision_tiny_3x_coco.pth">config</a></td>
    <td><a href="https://huggingface.co/nvidia/cascade_mask_rcnn_mamba_vision_tiny_3x_coco">log</a></td>
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
    <td><a href="https://huggingface.co/nvidia/cascade_mask_rcnn_mamba_vision_small_3x_coco">HF</a></td>
    <td><a href="https://huggingface.co/nvidia/cascade_mask_rcnn_mamba_vision_tiny_3x_coco/resolve/main/cascade_mask_rcnn_mamba_vision_tiny_3x_coco.pth">config</a></td>
    <td><a href="https://huggingface.co/nvidia/cascade_mask_rcnn_mamba_vision_tiny_3x_coco">log</a></td>
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
    <td><a href="https://huggingface.co/nvidia/cascade_mask_rcnn_mamba_vision_base_3x_coco">HF</a></td>
    <td><a href="https://huggingface.co/nvidia/cascade_mask_rcnn_mamba_vision_tiny_3x_coco/resolve/main/cascade_mask_rcnn_mamba_vision_tiny_3x_coco.pth">config</a></td>
    <td><a href="https://huggingface.co/nvidia/cascade_mask_rcnn_mamba_vision_tiny_3x_coco">log</a></td>
    <td><a href="https://huggingface.co/nvidia/cascade_mask_rcnn_mamba_vision_base_3x_coco/resolve/main/cascade_mask_rcnn_mamba_vision_base_3x_coco.pth">model</a></td>
</tr>


</table>


## Training

For training, we recommend using our slurm [train script](https://github.com/open-mmlab/mmdetection) which uses 8 GPUs.


## Evaluation

For evaluation, we recommend using our slurm [test script](https://github.com/open-mmlab/mmdetection) for inference with 8 GPUs. 

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