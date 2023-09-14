# ARG VERSION="21.10"
ARG VERSION="20.12"

FROM nvcr.io/nvidia/pytorch:${VERSION}-py3

ENV FORCE_CUDA="1" \
    NOTEBOOK_APP_TOKEN="123456" \
    NOTEBOOK_DIR="/workspace"

# set up environment:
ENV DEBIAN_FRONTEND noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# ENV HOME=/root SHELL=/bin/bash
# 开启终端色彩
ENV TERM=xterm-256color

# COPY sources.list /etc/apt/sources.list
RUN apt update \
    && apt upgrade -y\
    && apt install -y --fix-missing build-essential cmake clangd-12 ffmpeg autoconf automake \
    libgl1-mesa-dev libsm6 libxext6\
    ca-certificates\
    git ninja-build libglib2.0-dev curl openssh-client openssh-server vim\
    && apt clean \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-12 100

# create cap env

# RUN conda update -n base -c defaults conda
# && source /root/.bashrc 

# SHELL ["/bin/bash", "--login", "-c"]
# ENV SHELL=/bin/bash
# RUN conda create --name=cap python=3.8 -y 
# RUN conda init bash
# RUN conda activate cap

COPY requirements.txt /workpsace/requirements.txt
WORKDIR /workpsace
RUN python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ --upgrade pip \
    && pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    &&pip install --no-cache-dir torch==1.10.2+cu111 torchvision==0.11.3+cu111 torchaudio==0.10.2 -f https://download.pytorch.org/whl/cu111/torch_stable.html \
    && pip install --no-cache-dir openmim \
    && mim install --no-cache-dir "mmcv==1.6.1" "mmdet==2.25.1" \
    && pip install -r requirements.txt



# 配置环境变量，使ssh连接时 env 也生效
# RUN export $(cat /proc/1/environ |tr '\0' '\n' | xargs) \
#     && source /etc/profile
# 配置ssh
RUN mkdir /var/run/sshd \
    && echo "root:123456" | chpasswd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config\
    && sed -i '/^session\s\+required\s\+pam_loginuid.so/s/^/#/' /etc/pam.d/sshd\
    && sed -i 's/UsePAM yes/UsePAM no/g' /etc/ssh/sshd_config\
    && mkdir -p /root/.ssh && chown root.root /root && chmod 700 /root/.ssh 
EXPOSE 22 8888

COPY entrypoint.sh /entrypoint.sh
COPY ssh_env_setup.py /workpsace/ssh_env_setup.py

# vscode version:1.76.1,  commit: 5e805b79fcb6ba4c2d23712967df89a089da575b
# COPY .vscode-server /root/.vscode-server
ENTRYPOINT [ "/entrypoint.sh" ]