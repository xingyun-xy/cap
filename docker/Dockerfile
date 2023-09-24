ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
LABEL maintainer="xingyun" email="202303117@any3.com"
USER root

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

ENV DEBIAN_FRONTEND=noninteractive

# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# (Optional, use Mirror to speed up downloads)
RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install the required packages
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 vim tree && \
    tmux itop wget expect itop openssh-client openssh-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --chown=root:root ../../config/.bashrc /root/
COPY --chown=root:root requirements.txt /root/requirements.txt

# Install Detectron2 by develop
RUN git clone https://github.com/facebookresearch/detectron2.git /root/detectron2 && \
    cd /root/detectron2 && \
    pip install -e .

# Install MMCV, MMDetection and MMSegmentation
# RUN pip install --no-cache-dir openmim && \
#     mim install --no-cache-dir "mmcv-full==1.7.1" "mmcv==1.6.1" "mmdet==2.25.1" && \
#     pip install --no-cache-dir -r requirements.txt

# Configure SSH
RUN mkdir -p /var/run/sshd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i '/^session\s\+required\s\+pam_loginuid.so/s/^/#/' /etc/pam.d/sshd \
    && sed -i 's/UsePAM yes/UsePAM no/g' /etc/ssh/sshd_config \
    && mkdir -p /root/.ssh
RUN ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa 
COPY --chown=root:root authorized_keys /root/.ssh

# code-server
# 创建 code-server 目录，禁止 codeserver 放在其他路径，否则任务无法启动
RUN mkdir -p /home/service && mkdir -p /home/service/workspace
# 修改点 2：拷贝 vscode 代码，根据需要修改源目录，禁止修改目标目录
COPY --chown=root:root code-server-4.0.1-linux-amd64.tar.gz /home/service/workspace
RUN cd /home/service/workspace && tar -zxvf code-server-4.0.1-linux-amd64.tar.gz && rm -f /home/service/workspace/code-server-4.0.1-linux-amd64.tar.gz

# .vscode-server
COPY --chown=root:root .vscode-server.tar /root
RUN cd /root && tar -xvf .vscode-server.tar && rm -f /root/.vscode-server.tar

RUN pip config set global.index-url http://202303117:Xy19940130@tspdemo.changan.com.cn/nexus/repository/pypi-public && \
	pip config set global.extra_index_url http://tspdemo.changan.com.cn/nexus/repository/pypi-public/simple && \
	pip config set install.trusted-host tspdemo.changan.com.cn && \
    git config --global http.sslverify false && \
    git config --global https.sslverify false && \
	git config --global user.email "202303117@any3.com" && \
	git config --global user.name "xingyun"

WORKDIR /root
