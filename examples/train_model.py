#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(1,'../../decode/')
sys.path.insert(1,'/')
sys.path.insert(1,'/FracTAL_ResUNet')
print(sys.path)

import os
import time

import rioxarray
import importlib
import xarray as xr
import numpy as np
from os.path import join, getsize

import mxnet as mx
from mxnet import autograd, gluon, gpu
from multiprocessing import cpu_count
from FracTAL_ResUNet.nn.loss import mtsk_loss 
from FracTAL_ResUNet.nn.loss.mtsk_loss import *
from FracTAL_ResUNet.models.semanticsegmentation.FracTAL_ResUNet import *


def main():
    x = mx.nd.ones((3,4), ctx=gpu(0))
    print(x)
    print("Hello World!")
    exit()

    base_path = '/home/chocobo/datasets/AI4Boundaries2/sentinel2'
    input_path = f'{base_path}/images'
    label_path = f'{base_path}/masks'

    mxnet_dataset_path = f'{base_path}/mxnet_dataset'
    input_ndarray = mx.nd.load(f'{mxnet_dataset_path}/LU_input_ndarray.mat')[0]
    label_ndarray = mx.nd.load(f'{mxnet_dataset_path}/LU_label_ndarray.mat')[0]
    input_ndarray.shape, label_ndarray.shape

    dataset = mx.gluon.data.dataset.ArrayDataset(input_ndarray, label_ndarray)
    train_data = gluon.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    

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


    net = FracTAL_ResUNet_cmtsk(
        nfilters_init=nfilters_init, NClasses=NClasses, depth=depth,
        ftdepth=ftdepth, psp_depth=psp_depth, norm_type=norm_type, 
        norm_groups=norm_groups, nheads_start=nheads_start)
    net.initialize()
    net.initialize(init='Xavier', force_reinit=True)
    optimizer = mx.optimizer.Adam()
    trainer = gluon.Trainer(net.collect_params(), optimizer=optimizer)

    importlib.reload(FracTAL_ResUNet.nn.loss.mtsk_loss)
    myMTSKL = mtsk_loss.mtsk_loss()

    for epoch in range(num_epochs):
        train_loss = 0.
        tic = time.time()
        i = 0
        l = len(train_data)
        # printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
        for data, label in train_data:
            # forward + backward
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
            # calculate training loss
            train_loss += loss.mean().asscalar()
            # printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
            # print(i, loss)
            i += 1
        print("Epoch %d: loss %.3f, in %.1f sec" % (
                epoch, train_loss/len(train_data), time.time()-tic))


if __name__ == "__main__":
    main()
