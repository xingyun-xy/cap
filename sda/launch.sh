#!/bin/bash
source ~/.bashrc
### check follow ####
ROOT_DIR="/root/cap-xy"
PRE_MODEL="/tmp/pre-model/model-2478-version-5"
# load data
# root_dir
echo "###### load data #####"
cd $ROOT_DIR
mkdir -p data/nuscenes
rm data/nuscenes/samples data/nuscenes/sweeps data/nuscenes/v1.0-trainval data/nuscenes/nuscenes_C_infos_train.pkl data/nuscenes/nuscenes_C_infos_val.pkl
ln -s /tmp/dataset/dataset-803-version-1/trainval/samples data/nuscenes/samples
ln -s /tmp/dataset/dataset-803-version-1/trainval/sweeps data/nuscenes/sweeps
ln -s /tmp/dataset/dataset-803-version-1/trainval/v1.0-trainval data/nuscenes/v1.0-trainval
ln -s /tmp/dataset/dataset-2766-version-1 data/nuscenes/maps
ln -s /tmp/dataset/dataset-2765-version-1/nuscenes_C_infos_train.pkl data/nuscenes/nuscenes_C_infos_train.pkl
ln -s /tmp/dataset/dataset-2765-version-1/nuscenes_C_infos_val.pkl data/nuscenes/nuscenes_C_infos_val.pkl

echo "##### load ckpt ######"
# load ckpt
ln -s $PRE_MODEL ckpts/

echo "###### pip install ######"
pip install -e .

echo "###### copy bev_pool_v2_ext.cpython-37m-x86_64-linux-gnu.so ######"
rm mmdet3d/ops/bev_pool_v2/bev_pool_v2_ext.cpython-37m-x86_64-linux-gnu.so
cp sda/bev_pool_v2_ext.cpython-37m-x86_64-linux-gnu.so mmdet3d/ops/bev_pool_v2/
chmod +x tools/dist_train.sh

# backup code
cp -r /root/cap-xy /tmp/model/
current_git_branch_latest_id=`git rev-parse HEAD`
touch /tmp/model/git_commit.log
echo current git branch latest commit: $current_git_branch_latest_id
echo current git branch latest commit: $current_git_branch_latest_id >> /tmp/model/git_commit.log