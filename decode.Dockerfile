# FROM mxnet/python:1.9.1_gpu_cu110_py3
FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

RUN export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}

RUN apt-get update \
  && apt-get upgrade -y \
  && DEBIAN_FRONTEND=noninteractive  apt-get install -y  --no-install-recommends \
    apt-utils \
    build-essential \
    cmake \
    wget \
    libgdal-dev \
    make \
    python3 \
    python3-dev \
    python3-pip \
    python3-wheel \
    python-pip-whl \
    curl \
    git \
    gzip \
    tar \
    unzip \
    libopenblas-dev\
    libfftw3-mpi-dev\
  && rm -rf /var/lib/apt/lists/*

COPY . /decode

# COPY ./requirements.txt /build/

# RUN python3 --version
# RUN pip --version
# RUN pip freeze | grep mxnet

# 1) especificar que trabajamos con python 3: python3, pip3

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb
RUN dpkg -i libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb

RUN pip install -r /decode/requirements.txt

WORKDIR /decode/examples
