FROM mxnet/python:1.9.1_gpu_cu102_py3

COPY . /decode
# COPY ./requirements.txt /build/

# RUN DEBIAN_FRONTEND=noninteractive apt install python3-pip -y

RUN python3 --version
RUN pip --version

RUN pip freeze | grep mxnet

# 1) especificar que trabajamos con python 3: python3, pip3
# 2) Actualizar python a 3.7

RUN pip install -r /decode/requirements.txt


WORKDIR /decode/examples