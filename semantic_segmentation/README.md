# Semantic Segmentation with MambaVision

##  Segmentation Results + Models 

<table>
  <tr>
    <th>Backbone</th>
    <th>Method</th>
    <th>Lr Schd</th>
    <th>mIoU</th>
    <th>#Params(M)</th>
    <th>FLOPs(G)</th>
    <th>Config</th>
    <th>Log</th>
    <th>Model Ckpt</th>
  </tr>

<tr>
    <td><a href="https://huggingface.co/nvidia/MambaVision-T-1K">MambaVision-T-1K</a></td>
    <td>UPerNet</td>
    <td>160K</td>
    <td>46.0</td>
    <td>55</td>
    <td>945</td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/semantic_segmentation/configs/mamba_vision/mamba_vision_160k_ade20k-512x512_tiny.py">config</a></td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/semantic_segmentation/tools/logs/mamba_vision_160k_ade20k-512x512_tiny.log">log</a></td>
    <td><a href="https://huggingface.co/nvidia/mamba_vision_160k_ade20k-512x512_tiny/resolve/main/mamba_vision_160k_ade20k-512x512_tiny.pth">model</a></td>
</tr>


<tr>
    <td><a href="https://huggingface.co/nvidia/MambaVision-S-1K">MambaVision-S-1K</a></td>
    <td>UPerNet</td>
    <td>160K</td>
    <td>48.2</td>
    <td>84</td>
    <td>1135</td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/semantic_segmentation/configs/mamba_vision/mamba_vision_160k_ade20k-512x512_small.py">config</a></td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/semantic_segmentation/tools/logs/mamba_vision_160k_ade20k-512x512_small.log">log</a></td>
    <td><a href="https://huggingface.co/nvidia/mamba_vision_160k_ade20k-512x512_small/resolve/main/mamba_vision_160k_ade20k-512x512_small.pth">model</a></td>
</tr>

<tr>
    <td><a href="https://huggingface.co/nvidia/MambaVision-B-1K">MambaVision-B-1K</a></td>
    <td>UPerNet</td>
    <td>160K</td>
    <td>49.1</td>
    <td>126</td>
    <td>1342</td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/semantic_segmentation/configs/mamba_vision/mamba_vision_160k_ade20k-512x512_base.py">config</a></td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/semantic_segmentation/tools/logs/mamba_vision_160k_ade20k-512x512_base.log">log</a></td>
    <td><a href="https://huggingface.co/nvidia/mamba_vision_160k_ade20k-512x512_base/resolve/main/mamba_vision_160k_ade20k-512x512_base.pth">model</a></td>
</tr>


<tr>
    <td><a href="https://huggingface.co/nvidia/MambaVision-L3-512-21K">MambaVision-L3-512-21K</a></td>
    <td>UPerNet</td>
    <td>160K</td>
    <td>53.2</td>
    <td>780</td>
    <td>3670</td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/semantic_segmentation/configs/mamba_vision/mamba_vision_160k_ade20k-640x640_l3_21k.py">config</a></td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/semantic_segmentation/tools/logs/mamba_vision_160k_ade20k-640x640_l3_21k.log">log</a></td>
    <td><a href="https://huggingface.co/nvidia/mamba_vision_160k_ade20k-640x640_l3_21k/resolve/main/mamba_vision_160k_ade20k-640x640_l3_21k.pth">model</a></td>
</tr>


</table>

## Installation

Our semantic segmentation code is built upon on top of the popular [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) framework. 

### 1. Verify MambaVision Backbone Support

Before proceeding, ensure your environment is configured to run MambaVision pre-trained backbones. For detailed prerequisites and setup instructions, see the MambaVision [installation guide](https://github.com/NVlabs/MambaVision/tree/main#Installation).

### 2. Install Dependencies

MambaVision builds on top of MMSegmentation and relies on the following packages:

```bash
pip install \
  mmengine==0.10.1 \
  mmcv==2.1.0 \
  opencv-python-headless \
  mmsegmentation==1.2.2 \
  mmdet==3.3.0 \
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
| MMSegmentation | 1.2.2       |
| MMEngine    | 0.10.1      |

### 4. Further Reading

For complete setup instructions and troubleshooting, see the MMSegmentation [installation guide](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation).


## Training

Start your experiments with **multi‑GPU training** using our Slurm script on 8 GPUs. First, browse the available configuration files for different MambaVision models [here](https://github.com/NVlabs/MambaVision/tree/main/semantic_segmentation/configs/mamba_vision)

Once you’ve chosen a config (e.g., `mamba_vision_160k_ade20k-512x512_small.py`), launch training with:

```bash
# multi‑GPU training (8 GPUs)
srun --gres=gpu:8 python tools/train.py configs/mamba_vision/<CONFIG_FILE>.py
```

But note that, as also mentioned in the config files, except for the small model, all other models have been trained with 2 nodese using 16 GPUs for our paper experiments. For this purpose, you can use our slurm-based [train script](https://github.com/NVlabs/MambaVision/blob/main/semantic_segmentation/tools/mamba_vision_160k_ade20k-512x512_tiny.sh) to run the experiments.


If you’d rather run on a single GPU—for quick tests or debugging—use:

```bash
# single‑GPU training
env CUDA_VISIBLE_DEVICES=0 python tools/train.py ${CONFIG}
```


## Evaluation

For evaluation, we recommend using our slurm [test script](https://github.com/NVlabs/MambaVision/blob/main/semantic_segmentation/tools/test.sh) for inference with 8 GPUs. 


We provide both multi‑GPU and single‑GPU inference options:

### Multi‑GPU Inference

Run our Slurm [test script](https://github.com/NVlabs/MambaVision/blob/main/semantic_segmentation/tools/test.sh) on 8 GPUs for high‑throughput evaluation:

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
```


## Data Preparation

Follow these steps to prepare the ADE20K dataset for pure semantic segmentation tasks:

1. **Download the dataset**

```bash
cd <path-to-mambavision_seg-root>
mkdir -p data/ade20k && cd data/ade20k
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ADEChallengeData2016.zip
```

2. **Verify directory structure**

```
data/ade20k/
├── images/
│   ├── training/
│   └── validation/
└── annotations/
    ├── training/
    └── validation/
```

3. **Configure your MMSEGMENTATION config**

In your semantic segmentation config (e.g., `/configs/mamba_vision/segmentation`), set:

```python
data_root = 'data/ade20k/'

data = dict(
    train=dict(
        type='ADE20KDataset',
        data_root=data_root,
        img_dir='images/training',
        ann_dir='annotations/training'),
    val=dict(
        type='ADE20KDataset',
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation'),
    test=dict(
        type='ADE20KDataset',
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation')
)
```

4. **Further details**

For more options (e.g., panoptic settings, custom pipelines), refer to the MMSEGmentation [dataset preparation guide](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md).



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

For license information regarding the MMSegmentation repository, please refer to its [repository](https://github.com/open-mmlab/mmsegmentation).

For ADE20OK dataset, please refer to its [terms of use](https://ade20k.csail.mit.edu/terms/).


## Acknowledgement
The segmentation repository is built on top of the [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and inspired by [VMamba](https://github.com/MzeroMiko/VMamba/tree/main/segmentation). 
