# Use PyTorch with CUDA and cuDNN
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

RUN pip install --no-cache-dir tensorboardX causal-conv1d==1.6.0 mamba-ssm==2.2.5 timm==1.0.9 einops transformers

# Set working directory and copy application code
WORKDIR /app
COPY . /app

