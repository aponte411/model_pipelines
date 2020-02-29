FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

# TF dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    locales \
    software-properties-common \
    libsnappy-dev &&
    apt-get clean &&
    rm -rf /var/lib/apt/lists/*

# Locales
RUN locale-gen en_GB.UTF-8
ENV LANG en_GB.UTF-8
ENV LANGUAGE en_GB:en
ENV LC_ALL en_GB.UTF-8

# Python requirements
COPY ./requirements.txt /root/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /root/requirements.txt

COPY . /root
WORKDIR /root
