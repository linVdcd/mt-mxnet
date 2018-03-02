# --------------------------------------------------------
# Focal loss
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by unsky https://github.com/unsky/
# --------------------------------------------------------

"""
Focal loss
"""

import mxnet as mx
import numpy as np


class MTLossOperator(mx.operator.CustomOp):
    def __init__(self,scale):
        super(MTLossOperator, self).__init__()
        self.scale = scale

    def forward(self, is_train, req, in_data, out_data, aux):
        cls_score = in_data[0].asnumpy()
        labels = in_data[1].asnumpy()
        self._labels = labels

        #pro_ = np.exp(cls_score - cls_score.max(axis=1).reshape((cls_score.shape[0], 1)))
        pro_ = np.exp(cls_score)
        pro_ /= pro_.sum(axis=1).reshape((cls_score.shape[0], 1))

        self.pro_ = pro_



        self.assign(out_data[0], req[0], mx.nd.array(pro_))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        labels = self._labels.astype('int')
        pro_ = self.pro_
        # i!=j

        # dx =  self._alpha * np.power(1 - pt, self._gamma - 1) * (self._gamma * (-1 * pt * pro_) * np.log(pt) + pro_ * (1 - pt)) * 1.0
        dx = pro_
        ####i==j

        cc =0

        for i in range(labels.shape[0]):
            if labels[i]!=-1:
                dx[i,labels[i]]=pro_[i,labels[i]]-1.
                cc+=1
            else:
                dx[i,:]=0.0
        if cc!=0:
            dx/=float(cc)
        dx *= self.scale

        self.assign(in_grad[0], req[0], mx.nd.array(dx))
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register('MTLoss')
class FocalLossProp(mx.operator.CustomOpProp):
    def __init__(self,scale):
        super(FocalLossProp, self).__init__(need_top_grad=False)
        self._scale = float(scale)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['focal_loss']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        labels_shape = in_shape[1]
        out_shape = data_shape
        return [data_shape, labels_shape], [out_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return MTLossOperator(self._scale)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []