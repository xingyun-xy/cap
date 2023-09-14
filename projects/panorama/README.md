# Configs of models used in Panorama

## 环境准备

首先请参考[Installation Guide](../../docs/source/quick_start/installation.md)完成CAP基础开发环境的安装


来安装Panorama相关模型训练所特别需要的专用依赖项。这里`cuda_version`可以是111或者102，取决于环境中torch对应的cuda版本。

## Getting Started
export PYTHONPATH=`pwd`:${PYTHONPATH}

### Train
python3 tools/train.py --stage with_bn --config projects/panorama/configs/resize/multitask.py -ids 4,5,6,7

### Pred
python3 tools/predict.py --stage with_bn --config projects/panorama/configs/resize/pred_multitask.py -ids 0,1

### eval
python3 tools/predict.py --stage with_bn --config projects/panorama/configs/resize/eval_multitask.py -ids 6,7

### Pred by BEV
python3 tools/predict.py --stage with_bn --config projects/panorama/configs/resize/pred_singletask_bev.py -ids 0
### Train BEV(matrixVT)
python3 tools/train.py --stage with_bn --config projects/panorama/configs/resize/multitask.py -ids 0 (if you wanna train bev only under multitask.py, please go to common.py and modify task,only keep bev_depth!) 

### Pytorch2onnx
python3 tools/pytorch2onnx.py --stage with_bn --config projects/panorama/configs/resize/pred_multitask.py -ids 7

### Calculate ops
python3 tools/calops.py --stage with_bn --config projects/panorama/configs/resize/pred_multitask.py -ids 7

### Tensorboard
tensorboard --logdir tmp_output/panorama/logs/ --host=127.0.0.1 --port=6006

### Work_dir
./CAP
../data/train 软连接到数据训练目录
../data/train
../data/train/vehicle_detection
../data/train/lane_parsing

## Important notice for pred_multitask.py
### READ THIS PART CAREFULLY BEFORE YOU RUN THE CODE!
Dataloader choice part will be modified later in a better way. 
At current stage, please choose your dataloader carefully and respectively!
For Nuscences Json format, you can either use data_loader_bev or data_loader_bev_cooperate_pilot (two choices, double the happiness)
For Changan image BEV visualization only, please use data_loader_changanbev
If your visualization task does not contain BEV, you can still use data_loader from pred_mulitask.py page.
Also, if you want to visualize changan bev image, please go to det_multitask.py and go to
def visulize() then read the temporary comments scrupulously! 