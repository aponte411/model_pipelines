FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

# TF dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    rsync \
    unzip \
    locales \
    git \
    htop \
    vim \
    tmux \
    sudo \
    lsof \
    docker.io \
    software-properties-common \
    libsnappy-dev &&
    apt-get clean &&
    rm -rf /var/lib/apt/lists/*

# Locales
RUN locale-gen en_GB.UTF-8
ENV LANG en_GB.UTF-8
ENV LANGUAGE en_GB:en
ENV LC_ALL en_GB.UTF-8

# APEX for 16-bit mixed precision training
RUN git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext

# Python requirements
COPY ./requirements.txt /root/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /root/requirements.txt

COPY . /root
WORKDIR /root
