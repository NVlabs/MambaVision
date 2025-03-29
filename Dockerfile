# Use PyTorch with CUDA and cuDNN
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

RUN pip install --no-cache-dir tensorboardX==2.6.2.2 mamba-ssm==2.2.4 timm==1.0.9 einops==0.8.1 transformers==4.50.0 requests==2.32.3 Pillow==11.1.0

# Set working directory and copy application code
WORKDIR /app
COPY . /app

