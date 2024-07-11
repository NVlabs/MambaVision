# MambaVision: A Hybrid Mamba-Transformer Vision Backbone

Official PyTorch implementation of **MambaVision: A Hybrid Mamba-Transformer Vision Backbone**.

[![Star on GitHub](https://img.shields.io/github/stars/NVlabs/MambaVision.svg?style=social)](https://github.com/NVlabs/MambaVision/stargazers)

[Ali Hatamizadeh](https://research.nvidia.com/person/ali-hatamizadeh),
[Jan Kautz](https://jankautz.com/), 

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

--- 

MambaVision demonstrates a strong performance by achieving a new SOTA Pareto-front in
terms of Top-1 accuracy and throughput. 

<p align="center">
<img src="https://github.com/NVlabs/MambaVision/assets/26806394/79dcf841-3966-4b77-883d-76cd5e1d4320" width=62% height=62% 
class="center">
</p>

We introduce a novel mixer block by creating a symmetric path without SSM to enhance the modeling of global context: 


<p align="center">
<img src="https://github.com/NVlabs/MambaVision/assets/26806394/295c0984-071e-4c84-b2c8-9059e2794182" width=32% height=32% 
class="center">
</p>



MambaVision has a hierarchial architecture that employs both self-attention and mixer blocks:

![teaser](./mambavision/assets/arch.png)


## 💥 News 💥
- **[07.11.2024]** [Mambavision pip package](https://pypi.org/project/mambavision/) is released !

- **[07.10.2024]** We have released the code and model checkpoints for Mambavision !

## Quick Start

### Classification

We can import pre-trained MambaVision models with **1 line of code**:

```bash
pip install mambavision
```

A pretrained MambaVision model with default hyper-parameters can be created as in:

```python
>>> from mambavision import create_model

# Define mamba_vision_T model with 224 x 224 resolution

>>> model = create_model('mamba_vision_T', 
                          pretrained=True,
                          model_path="/tmp/mambavision_tiny_1k.pth.tar")
```

`model_path` is used to set the directory to download the model.

We can also simply test the model by passing a dummy input image. The output is the logits:

```python
>>> import torch

>>> image = torch.rand(1, 3, 224, 224)
>>> output = model(image) # torch.Size([1, 1000])
```

## Results + Pretrained Models

### ImageNet-1K
**MambaVision ImageNet-1K Pretrained Models**

<table>
  <tr>
    <th>Name</th>
    <th>Acc@1(%)</th>
    <th>Acc@5(%)</th>
    <th>Throughput(Img/Sec)</th>
    <th>Resolution</th>
    <th>#Params(M)</th>
    <th>FLOPs(G)</th>
    <th>Download</th>
  </tr>

<tr>
    <td>MambaVision-T</td>
    <td>82.3</td>
    <td>96.2</td>
    <td>6298</td>
    <td>224x224</td>
    <td>31.8</td>
    <td>4.4</td>
    <td><a href="https://drive.google.com/file/d/1zE8czwSTG5ogcsb93A95o_F3rlYf8R1G/view?usp=sharing">model</a></td>
</tr>

<tr>
    <td>MambaVision-T2</td>
    <td>82.7</td>
    <td>96.3</td>
    <td>5990</td>
    <td>224x224</td>
    <td>35.1</td>
    <td>5.1</td>
    <td><a href="https://drive.google.com/file/d/1KNJVRRBUSqOq7ZxqH1mDth4wWL5f1SFq/view?usp=sharing">model</a></td>
</tr>

<tr>
    <td>MambaVision-S</td>
    <td>83.3</td>
    <td>96.5</td>
    <td>4700</td>
    <td>224x224</td>
    <td>50.1</td>
    <td>7.5</td>
    <td><a href="https://drive.google.com/file/d/1XoSctKJgRI6OMmYmdKOoTzvnoOtfqI64/view?usp=sharing">model</a></td>
</tr>

<tr>
    <td>MambaVision-B</td>
    <td>84.2</td>
    <td>96.9</td>
    <td>3670</td>
    <td>224x224</td>
    <td>97.7</td>
    <td>15.0</td>
    <td><a href="https://drive.google.com/file/d/1wR2UeFzSmNjrC3jqJgp4IOGvYhlO9QYw/view?usp=sharing">model</a></td>
</tr>

<tr>
    <td>MambaVision-L</td>
    <td>85.0</td>
    <td>97.1</td>
    <td>2190</td>
    <td>224x224</td>
    <td>227.9</td>
    <td>34.9</td>
    <td><a href="https://drive.google.com/file/d/1YfA9K_ZbZcoLCif-ltLWCvj2pQCvb4bJ/view?usp=sharing">model</a></td>
</tr>

<tr>
    <td>MambaVision-L2</td>
    <td>85.3</td>
    <td>97.2</td>
    <td>1021</td>
    <td>224x224</td>
    <td>241.5</td>
    <td>37.5</td>
    <td><a href="https://drive.google.com/file/d/1fw1bo_oNtIImScW38DFJIPKnRt6GrlS9/view?usp=sharing">model</a></td>
</tr>

</table>

## Installation

We provide a [docker file](./Dockerfile). In addition, assuming that a recent [PyTorch](https://pytorch.org/get-started/locally/) package is installed, the dependencies can be installed by running:

```bash
pip install -r requirements.txt
```

## Evaluation

The MambaVision models can be evaluated on ImageNet-1K validation set using the following: 

```
python validate.py \
--model <model-name>
--checkpoint <checkpoint-path>
--data_dir <imagenet-path>
--batch-size <batch-size-per-gpu
``` 

Here `--model` is the MambaVision variant (e.g. `mambavision_tiny_1k`), `--checkpoint` is the path to pretrained model weights, `--data_dir` is the path to ImageNet-1K validation set and `--batch-size` is the number of batch size. We also provide a sample script [here](./mambavision/validate.sh). 


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=NVlabs/MambaVision&type=Date)](https://star-history.com/#NVlabs/MambaVision&Date)


## Licenses

Copyright © 2024, NVIDIA Corporation. All rights reserved.

This work is made available under the NVIDIA Source Code License-NC. Click [here](LICENSE) to view a copy of this license.

For license information regarding the timm repository, please refer to its [repository](https://github.com/rwightman/pytorch-image-models).

For license information regarding the ImageNet dataset, please see the [ImageNet official website](https://www.image-net.org/). 

## Acknowledgement
This repository is built on top of the [timm](https://github.com/huggingface/pytorch-image-models) repository. We thank [Ross Wrightman](https://rwightman.com/) for creating and maintaining this high-quality library.  
