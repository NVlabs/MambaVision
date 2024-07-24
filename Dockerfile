# Use PyTorch with CUDA and cuDNN
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

RUN pip install --no-cache-dir tensorboardX causal-conv1d==1.0.2 mamba-ssm==1.0.1 timm==0.9.0 einops transformers

# Set working directory and copy application code
WORKDIR /app
COPY . /app

