
#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from models.mamba_vision import *
import argparse
import time
import numpy as np
from ptflops import get_model_complexity_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model name",
                        default="mamba_vision_T",type=str)
    parser.add_argument("--resolution", help="model resolution",type=int,
                        default=224)
    parser.add_argument("--bs", help="batch size",type=int,
                        default=128)
    parser.add_argument("--channel_last", help="run trt mode",
                        action="store_true")
    args = parser.parse_args()

    if args.model == "mamba_vision_T":
        model = mamba_vision_T()
    elif args.model == "mamba_vision_T2":
        model = mamba_vision_T2()
    elif args.model == "mamba_vision_S":
        model = mamba_vision_S()
    elif args.model == "mamba_vision_B":
        model = mamba_vision_B()
    elif args.model == "mamba_vision_L":
        model = mamba_vision_L()
    elif args.model == "mamba_vision_L2":
        model = mamba_vision_L2()

    input_data = torch.randn((bs, 3, resolution, resolution), device='cuda').cuda()

    # we recommend using channel_last
    if args.channel_last:
        input_data = input_data.to(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)

    model.cuda()
    model.eval()
    macs, params = get_model_complexity_info(model, tuple([3, resolution, resolution]),
                                             as_strings=False, print_per_layer_stat=False, verbose=False)

    print(f"Model stats: macs: {macs}, and params: {params}")

    # warm up
    runs=10
    with torch.cuda.amp.autocast():
        for ii in range(runs):
            with torch.no_grad():
                output = model(input_data)

    timer = []
    start_time = time.time()
    runs=500
    with torch.cuda.amp.autocast(True):

        for ii in range(runs):
            start_time_loc = time.time()
            with torch.no_grad():
                output = model(input_data)

            timer.append(time.time()-start_time_loc)
        torch.cuda.synchronize()
    end_time = time.time()
    print(f"Throughput {bs * 1.0 / ((end_time - start_time) / runs)}")
    print(f"Throughput Med {int(bs * 1.0 / ((np.median(timer))))}")
