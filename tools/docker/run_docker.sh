export DATA_DIR=/root/users/data_hdd/users_data
export CODE_DIR=/media/home/users/work
docker run --name xx_cap  --ipc=host --gpus all -it -v ${DATA_DIR}:/data -v ${CODE_DIR}:/code -p 8888:22 cap-trt843:v1.0
