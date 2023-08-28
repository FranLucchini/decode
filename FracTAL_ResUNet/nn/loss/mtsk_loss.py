from FracTAL_ResUNet.nn.loss.ftnmt_loss import *
from mxnet import cpu

class mtsk_loss(object):
    """
    Here NClasses = 2 by default, for a binary segmentation problem in 1hot representation 
    """

    def __init__(self,depth=0, NClasses=2):

        self.ftnmt = ftnmt_loss(depth=depth)
        self.ftnmt.hybridize() 

        self.skip = NClasses

    def loss(self,_prediction,_label):
        # z = z.as_in_context(gpu(0))
        pred_segm  = _prediction[0]# .as_in_context(cpu(0))
        pred_bound = _prediction[1]# .as_in_context(cpu(0))
        pred_dists = _prediction[2]# .as_in_context(cpu(0))
        
        # In our implementation of the labels, we stack together the [segmentation, boundary, distance] labels, 
        # along the channel axis.
        # print(type(_label), len(_label))
        label_segm  = _label[0]#.as_in_context(cpu(0))
        label_bound = _label[1]#.as_in_context(cpu(0))
        label_dists = _label[2]#.as_in_context(cpu(0))
        # label_segm  = _label[:,:self.skip,:,:]
        # label_bound = _label[:,self.skip:2*self.skip,:,:]
        # label_dists = _label[:,2*self.skip:,:,:]

        # print(pred_segm.ctx, label_segm.ctx)
        # print(pred_segm.shape, label_segm.shape)
        loss_segm  = self.ftnmt(pred_segm,  label_segm)

        # print(pred_bound.ctx, label_bound.ctx)
        # print(pred_bound.shape, label_bound.shape)
        loss_bound = self.ftnmt(pred_bound, label_bound)

        # print(pred_dists.ctx, label_dists.ctx)
        # print(pred_dists.shape, label_dists.shape)
        loss_dists = self.ftnmt(pred_dists, label_dists)

        return (loss_segm+loss_bound+loss_dists)/3.0

