#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(1,'../../decode/')
sys.path.insert(1,'/')
sys.path.insert(1,'/FracTAL_ResUNet')
# print(sys.path)

import os
import time
import logging
import tracemalloc

import rioxarray
import importlib
import xarray as xr
import numpy as np
from os.path import join, getsize

from utils import *

import mxnet as mx
from mxnet import autograd, gluon, gpu
# from multiprocessing import cpu_count
from FracTAL_ResUNet.nn.loss.mtsk_loss import *
from FracTAL_ResUNet.models.semanticsegmentation.FracTAL_ResUNet import *

## Variables
# D6nf32 example 
nfilters_init=32
depth=6
psp_depth=4

ftdepth=5
norm_type='GroupNorm'
norm_groups=4
NClasses=1
nheads_start=4

num_epochs = 5 # 254 - 5 days for Australia
batch_size = 6

def main(args):
    # x = mx.nd.ones((3,4), ctx=gpu(0))
    # print(x)
    # print("Hello World!")

    base_path = '/decode/examples'
    # input_path = f'{base_path}/images'
    # label_path = f'{base_path}/masks'
    mxnet_dataset_path = f'{base_path}/input/mxnet_dataset'

    # starting the monitoring
    tracemalloc.start()

    print('Reading MXNet NDArray files')
    input_ndarray = mx.nd.load(f'{mxnet_dataset_path}/LU_input_ndarray.mat')[0]# .as_in_context(gpu(0))
    label_ndarray = mx.nd.load(f'{mxnet_dataset_path}/LU_label_ndarray.mat')[0]# .as_in_context(gpu(0))
    input_ndarray.shape, label_ndarray.shape

    print('Build dataset')
    dataset = mx.gluon.data.dataset.ArrayDataset(input_ndarray, label_ndarray)

    print('Build dataloader')
    # NOTE comment multiprocess because we are not working with multiple GPUs so
    #  it throws a context error
    train_data = gluon.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True) #, num_workers=4)
    
    print('Create FracTAL_ResUNet')
    net = FracTAL_ResUNet_cmtsk(
        nfilters_init=nfilters_init, NClasses=NClasses, depth=depth,
        ftdepth=ftdepth, psp_depth=psp_depth, norm_type=norm_type, 
        norm_groups=norm_groups, nheads_start=nheads_start)
    
    print('Initialize weights')
    net.initialize(init='Xavier', force_reinit=True, ctx=gpu(0))
    # net.load_parameters('FracTAL_ResUNet_cmtsk.params', ctx=gpu(0))

    print('Configure trainer')
    optimizer = mx.optimizer.Adam()
    trainer = gluon.Trainer(net.collect_params(), optimizer=optimizer)

    print('Build loss function')
    myMTSKL = mtsk_loss()

    # displaying the memory
    print(tracemalloc.get_traced_memory())

    # exit()
    print('Begin training')
    for epoch in range(num_epochs):
        train_loss = 0.
        tic = time.time()
        i = 0
        l = len(train_data)
        printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
        for data, label in train_data:
            # forward + backward
            data = data.as_in_context(gpu(0))
            label = label.as_in_context(gpu(0))
            with autograd.record():
                output = net(data)
                # Transform label to correct shape
                t_labels = [label[:, :, t , :, :] for t in range(label.shape[2])]
                # print(len(t_labels), t_labels[0].shape, len(output), output[0].shape)
                # print(output[0].ctx, t_labels[0].ctx)
                loss = myMTSKL.loss(output, t_labels)
                # print(loss)
            loss.backward()
            # update parameters
            trainer.step(batch_size)

            # displaying the memory
            print(tracemalloc.get_traced_memory())

            # calculate training loss
            train_loss += loss.mean().asscalar()
            printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
            # print(i, loss)
            i += 1
        print("Epoch %d: loss %.3f, in %.1f sec" % (
                epoch, train_loss/len(train_data), time.time()-tic))
        # exit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('integers', metavar='N', type=int, nargs='+',
    #                     help='an integer for the accumulator')
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')

    args = parser.parse_args()
    main(args)
