# -*- coding: utf-8 -*-
# Author: xingyun

import os
import sys
import time
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

import torch
import timm
from torchvision.models import resnet50

def occupy_gpu():
    model = timm.create_model('resnet50', pretrained=True).cuda()
    x = torch.randn(10, 3, 224, 224)
    while True:
        model(x)

def run_gpu():
    # Old weights with accuracy 76.130%, resnet50-0676ba61.pth
    # model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = resnet50()
    # New weights with accuracy 80.858%, download resnet50-11ad3fa6.pth to /root/.cache/torch/hubs/checkpoints/
    # model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.cuda()
    model.eval()
    # gpu_util = gpu_info()[-1]
    while True:
        with torch.no_grad():
            dummy_input = torch.rand(128, 3, 224, 224).cuda()
            model(dummy_input)
            # gpu_power, gpu_memory, gpu_util = gpu_info()
            # print(f'gpu_util: {gpu_util}, gpu_power: {gpu_power}, gpu_memory: {gpu_memory}')

def gpu_info():
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    gpu_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip())
    gpu_power = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
    gpu_util = int(gpu_status[3].strip().split('%')[0])
    return gpu_power, gpu_memory, gpu_util

def narrow_setup(interval=2):
    gpu_power, gpu_memory, gpu_util = gpu_info()
    i = 0
    # set waiting condition
    # while gpu_memory > 1000 or gpu_power > 20:
    GPU_UTIL_THRESHOLD = -1

    while gpu_util > GPU_UTIL_THRESHOLD:  
        gpu_power, gpu_memory, gpu_util = gpu_info()
        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        gpu_power_str = f'gpu power: {gpu_power} W |'
        gpu_memory_str = f'gpu memory: {gpu_memory} MiB |'
        gpu_util_str = f'gpu util: {gpu_util} % |'
        sys.stdout.write('\r' + gpu_memory_str + ' ' + gpu_power_str + ' ' + gpu_util_str + ' ' + symbol)
        sys.stdout.flush()
        time.sleep(interval)
        i += 1
    print('run cmd')
 

if __name__ == "__main__":
    # narrow_setup()
    logging.info('run gpu')
    # run_gpu()
    run_gpu()