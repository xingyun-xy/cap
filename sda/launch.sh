#!/bin/bash
source ~/.bashrc
### check follow ####
ROOT_DIR="/root/cap-xy"
PRE_MODEL="/tmp/pre-model/model-2478-version-5"
# load data
# root_dir
echo "###### load data #####"
cd $ROOT_DIR
mkdir -p data/cabev
ln -s /tmp/dataset/dataset-3180-version-1 data/cabev/dataset-3180-version-1
ln -s /tmp/dataset/ataset-3180-version-5 data/cabev/dataset-3180-version-5
ln -s /tmp/dataset/dataset-3218-version-1 data/cabev/dataset-3218-version-1
ln -s /tmp/dataset/dataset-3237-version-1 data/cabev/dataset-3237-version-1
ln -s /tmp/dataset/dataset-3363-version-1 data/cabev/dataset-3363-version-1
ln -s /tmp/dataset/dataset-3377-version-1 data/cabev/dataset-3377-version-1
ln -s /tmp/dataset/dataset-3480-version-1 data/cabev/dataset-3480-version-1
ln -s /tmp/dataset/dataset-3537-version-1 data/cabev/dataset-3537-version-1
ln -s /tmp/dataset/dataset-3551-version-1 data/cabev/dataset-3551-version-1
ln -s /tmp/dataset/dataset-3557-version-1 data/cabev/dataset-3557-version-1
ln -s /tmp/dataset/dataset-3557-version-2 data/cabev/dataset-3557-version-2
ln -s /tmp/dataset/dataset-3557-version-3 data/cabev/dataset-3557-version-3
ln -s /tmp/dataset/dataset-3557-version-4 data/cabev/dataset-3557-version-4
ln -s /tmp/dataset/dataset-3557-version-5 data/cabev/dataset-3557-version-5
ln -s /tmp/dataset/dataset-3698-version-1 data/cabev/dataset-3698-version-1
ln -s /tmp/dataset/dataset-3804-version-1 data/cabev/dataset-3804-version-1
ln -s /tmp/dataset/dataset-3805-version-1 data/cabev/dataset-3805-version-1
ln -s /tmp/dataset/dataset-3810-version-1 data/cabev/dataset-3810-version-1
ln -s /tmp/dataset/dataset-3812-version-1 data/cabev/dataset-3812-version-1

echo "##### load ckpt ######"
# load ckpt
ln -s $PRE_MODEL ckpts

echo "###### pip install ######"
# pip install -e .

# backup code
# cp -r /root/cap-xy /tmp/model/
# current_git_branch_latest_id=`git rev-parse HEAD`
# touch /tmp/model/git_commit.log
# echo current git branch latest commit: $current_git_branch_latest_id
# echo current git branch latest commit: $current_git_branch_latest_id >> /tmp/model/git_commit.log