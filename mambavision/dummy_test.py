import torch
from timm.models import create_model, load_checkpoint
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', metavar='NAME', default='mamba_vision_T', help='model architecture (default: mamba_vision_T)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--use_pip', action='store_true', default=False, help='to use pip package')
args = parser.parse_args()

# Define mamba_vision_T model with 224 x 224 resolution

if args.use_pip:
      from mambavision import create_model
      model = create_model(args.model, pretrained=True, model_path="/tmp/mambavision_tiny_1k.pth.tar")
else:
      from models.mamba_vision import *
      model = create_model(args.model) 
      if args.checkpoint:
        load_checkpoint(model, args.checkpoint, None)
        
print('{} model succesfully created !'.format(args.model))

image = torch.rand(1, 3, 754, 234).cuda() # place image on cuda

model = model.cuda() # place model on cuda

output = model(image) # output logit size is [1, 1000]

print('Inference succesfully completed on dummy input !')

