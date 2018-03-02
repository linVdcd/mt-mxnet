import mxnet as mx
from myloss import *
def get_symbol(num_classes,**kwargs):
    data = mx.symbol.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn_data')
    conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=32, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                                  no_bias=False)
    relu_conv1 = mx.symbol.Activation(name='relu_conv1', data=conv1, act_type='relu')
    pool1 = mx.symbol.Pooling(name='pool1', data=relu_conv1, pooling_convention='full', pad=(0, 0), kernel=(3, 3),
                              stride=(2, 2), pool_type='max')
    fire2_squeeze1x1 = mx.symbol.Convolution(name='fire2_squeeze1x1', data=pool1, num_filter=16, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire2_relu_squeeze1x1 = mx.symbol.Activation(name='fire2_relu_squeeze1x1', data=fire2_squeeze1x1, act_type='relu')
    fire2_expand1x1 = mx.symbol.Convolution(name='fire2_expand1x1', data=fire2_relu_squeeze1x1, num_filter=48,
                                            pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire2_relu_expand1x1 = mx.symbol.Activation(name='fire2_relu_expand1x1', data=fire2_expand1x1, act_type='relu')
    fire2_expand3x3 = mx.symbol.Convolution(name='fire2_expand3x3', data=fire2_relu_squeeze1x1, num_filter=48,
                                            pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False)
    fire2_relu_expand3x3 = mx.symbol.Activation(name='fire2_relu_expand3x3', data=fire2_expand3x3, act_type='relu')
    fire2_concat = mx.symbol.Concat(name='fire2_concat', *[fire2_relu_expand1x1, fire2_relu_expand3x3])
    pool2 = mx.symbol.Pooling(name='pool2', data=fire2_concat, pooling_convention='full', pad=(0, 0), kernel=(3, 3),
                              stride=(2, 2), pool_type='max')
    fire4_squeeze1x1 = mx.symbol.Convolution(name='fire4_squeeze1x1', data=pool2, num_filter=32, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire4_relu_squeeze1x1 = mx.symbol.Activation(name='fire4_relu_squeeze1x1', data=fire4_squeeze1x1, act_type='relu')
    fire4_expand1x1 = mx.symbol.Convolution(name='fire4_expand1x1', data=fire4_relu_squeeze1x1, num_filter=64,
                                            pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire4_relu_expand1x1 = mx.symbol.Activation(name='fire4_relu_expand1x1', data=fire4_expand1x1, act_type='relu')
    fire4_expand3x3 = mx.symbol.Convolution(name='fire4_expand3x3', data=fire4_relu_squeeze1x1, num_filter=64,
                                            pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False)
    fire4_relu_expand3x3 = mx.symbol.Activation(name='fire4_relu_expand3x3', data=fire4_expand3x3, act_type='relu')
    fire4_concat = mx.symbol.Concat(name='fire4_concat', *[fire4_relu_expand1x1, fire4_relu_expand3x3])
    fire5_squeeze1x1 = mx.symbol.Convolution(name='fire5_squeeze1x1', data=fire4_concat, num_filter=32, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire5_relu_squeeze1x1 = mx.symbol.Activation(name='fire5_relu_squeeze1x1', data=fire5_squeeze1x1, act_type='relu')
    fire5_expand1x1 = mx.symbol.Convolution(name='fire5_expand1x1', data=fire5_relu_squeeze1x1, num_filter=96,
                                            pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire5_relu_expand1x1 = mx.symbol.Activation(name='fire5_relu_expand1x1', data=fire5_expand1x1, act_type='relu')
    fire5_expand3x3 = mx.symbol.Convolution(name='fire5_expand3x3', data=fire5_relu_squeeze1x1, num_filter=96,
                                            pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False)
    fire5_relu_expand3x3 = mx.symbol.Activation(name='fire5_relu_expand3x3', data=fire5_expand3x3, act_type='relu')
    fire5_concat = mx.symbol.Concat(name='fire5_concat', *[fire5_relu_expand1x1, fire5_relu_expand3x3])
    pool5 = mx.symbol.Pooling(name='pool5', data=fire5_concat, pooling_convention='full', pad=(0, 0), kernel=(3, 3),
                              stride=(2, 2), pool_type='max')
    fire6_squeeze1x1 = mx.symbol.Convolution(name='fire6_squeeze1x1', data=pool5, num_filter=48, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire6_relu_squeeze1x1 = mx.symbol.Activation(name='fire6_relu_squeeze1x1', data=fire6_squeeze1x1, act_type='relu')
    fire6_expand1x1 = mx.symbol.Convolution(name='fire6_expand1x1', data=fire6_relu_squeeze1x1, num_filter=128,
                                            pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    fire6_relu_expand1x1 = mx.symbol.Activation(name='fire6_relu_expand1x1', data=fire6_expand1x1, act_type='relu')
    fire6_expand3x3 = mx.symbol.Convolution(name='fire6_expand3x3', data=fire6_relu_squeeze1x1, num_filter=128,
                                            pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False)
    fire6_relu_expand3x3 = mx.symbol.Activation(name='fire6_relu_expand3x3', data=fire6_expand3x3, act_type='relu')
    fire6_concat = mx.symbol.Concat(name='fire6_concat', *[fire6_relu_expand1x1, fire6_relu_expand3x3])
    drop = mx.symbol.Dropout(name='drop', data=fire6_concat, p=0.500000)

    #task1 hair
    conv10race1 = mx.symbol.Convolution(name='conv10race1', data=drop, num_filter=2, pad=(0, 0), kernel=(1, 1),
                                        stride=(1, 1), no_bias=False)
    relu_conv101 = mx.symbol.Activation(name='relu_conv101', data=conv10race1, act_type='relu')
    pool101 = mx.symbol.Pooling(name='pool101', data=relu_conv101, pooling_convention='full', global_pool=True,
                                kernel=(1, 1), pool_type='avg')
    flat1 = mx.symbol.Flatten(data=pool101,name='flat1')
    loss1 = mx.sym.SoftmaxOutput(name='softmax1', data=flat1,ignore_label=1000,use_ignore=True)
    #taks2 glasses
    conv10race2 = mx.symbol.Convolution(name='conv10race2', data=drop, num_filter=2, pad=(0, 0), kernel=(1, 1),
                                        stride=(1, 1), no_bias=False)
    relu_conv102 = mx.symbol.Activation(name='relu_conv102', data=conv10race2, act_type='relu')
    pool102 = mx.symbol.Pooling(name='pool102', data=relu_conv102, pooling_convention='full', global_pool=True,
                                kernel=(1, 1), pool_type='avg')
    flat2 = mx.symbol.Flatten(data=pool102,name='flat2')
    loss2 = mx.sym.SoftmaxOutput(name='softmax2', data=flat2,ignore_label=1000,use_ignore=True)

    #task3 gender
    conv10race3 = mx.symbol.Convolution(name='conv10race3', data=drop, num_filter=2, pad=(0, 0), kernel=(1, 1),
                                        stride=(1, 1), no_bias=False)
    relu_conv103 = mx.symbol.Activation(name='relu_conv103', data=conv10race3, act_type='relu')
    pool103 = mx.symbol.Pooling(name='pool103', data=relu_conv103, pooling_convention='full', global_pool=True,
                                kernel=(1, 1), pool_type='avg')
    flat3 = mx.symbol.Flatten(data=pool103,name='flat3')
    loss3 = mx.sym.SoftmaxOutput(name='softmax3', data=flat3,ignore_label=1000,use_ignore=True)

    #task4 headangle
    conv10race4 = mx.symbol.Convolution(name='conv10race4', data=drop, num_filter=3, pad=(0, 0), kernel=(1, 1),
                                        stride=(1, 1), no_bias=False)
    relu_conv104 = mx.symbol.Activation(name='relu_conv104', data=conv10race4, act_type='relu')
    pool104 = mx.symbol.Pooling(name='pool104', data=relu_conv104, pooling_convention='full', global_pool=True,
                                kernel=(1, 1), pool_type='avg')
    flat4 = mx.symbol.Flatten(data=pool104, name='flat4')
    loss4 = mx.sym.SoftmaxOutput(name='softmax4', data=flat4, ignore_label=1000, use_ignore=True)
    #task5 light
    conv10race5 = mx.symbol.Convolution(name='conv10race5', data=drop, num_filter=6, pad=(0, 0), kernel=(1, 1),
                                        stride=(1, 1), no_bias=False)
    relu_conv105 = mx.symbol.Activation(name='relu_conv105', data=conv10race5, act_type='relu')
    pool105 = mx.symbol.Pooling(name='pool105', data=relu_conv105, pooling_convention='full', global_pool=True,
                                kernel=(1, 1), pool_type='avg')
    flat5 = mx.symbol.Flatten(data=pool105, name='flat5')
    loss5 = mx.sym.SoftmaxOutput(name='softmax5', data=flat5, ignore_label=1000, use_ignore=True)
    return mx.symbol.Group([loss1,loss2,loss3,loss4,loss5])
