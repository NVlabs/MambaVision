# Use PyTorch with CUDA and cuDNN
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

RUN pip install --no-cache-dir tensorboardX causal-conv1d==1.4.0 mamba-ssm==2.2.2 timm==1.0.9 einops transformers

# Set working directory and copy application code
WORKDIR /app
COPY . /app

