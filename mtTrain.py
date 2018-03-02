import argparse
import mxnet as mx
import logging
from sq import *
from metric import *
import os
import numpy as np
import matplotlib.pyplot as plt
import random

logging.basicConfig(level=logging.DEBUG)
random.seed(123)



def _get_lr_scheduler(lr1, lr_factor, lr_step_epochs, load_epoch, num_examples=500000):
    epoch_size = num_examples / args.batch_size

    begin_epoch = load_epoch if load_epoch else 0
    step_epochs = [int(l) for l in lr_step_epochs.split(',')]
    lr = lr1
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= lr_factor
    if lr != lr1:
        logging.info('Adjust learning rate to %e for epoch %d' % (lr, begin_epoch))

    steps = [epoch_size * (x - begin_epoch) for x in step_epochs if x - begin_epoch > 0]
    return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_factor))


class MultiTask_iter(mx.io.DataIter):
    def __init__(self, data_iter):
        super(MultiTask_iter, self).__init__('multitask_iter')
        self.data_iter = data_iter
        self.batch_size = self.data_iter.batch_size

    @property
    def provide_data(self):
        return self.data_iter.provide_data

    @property
    def provide_label(self):
        provide_label = self.data_iter.provide_label[0]
        # the name of the label if corresponding to the model you define in get_fine_tune_model() function
        return [('softmax1_label', [provide_label[1][0]]), ('softmax2_label', [provide_label[1][0]])]

    def hard_reset(self):
        self.data_iter.hard_reset()

    def reset(self):
        self.data_iter.reset()

    def next(self):
        batch = self.data_iter.next()

        label = batch.label[0]
        ll = label.asnumpy()
        label1 = mx.nd.array(ll[:, 0]).astype('float32')
        label2 = mx.nd.array(ll[:, 1]).astype('float32')

        return mx.io.DataBatch(data=batch.data, label=[label1, label2],
                               pad=batch.pad, index=batch.index)


'''
class Multi_Accuracy(mx.metric.EvalMetric):
    """Calculate accuracies of multi label"""

    def __init__(self, num=None):
        super(Multi_Accuracy, self).__init__('multi-accuracy', num)

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)

        if self.num is not None:
            assert len(labels) == self.num

        for i in range(len(labels)):
            pred_label = mx.nd.argmax_channel(preds[i]).asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')

            mx.metric.check_label_shapes(label, pred_label)

            if i is None:
                for j in range(len(label)):
                    if label[j]!=1000:
                        self.sum_metric += pred_label.flat[j] == label.flat[j]
                        self.num_inst += 1
            else:

                for j in range(len(label)):
                    if label[j]!=1000:
                        self.sum_metric[i] += pred_label.flat[j] == label.flat[j]
                        self.num_inst[i] += 1
                #self.sum_metric[i] += (pred_label.flat == label.flat).sum()
                #self.num_inst[i] += len(pred_label.flat)

'''


def train_model(model, gpus, batch_size, image_shape, epoch=0, num_epoch=20, kv='device'):
    train = mx.io.ImageRecordIter(
        path_imgrec='/home/research/data/linmingan/multiTask/faceM_train.rec',
        # path_imglist='/home/research/data/linmingan/hair/hair_train.lst',
        data_name='data',
        label_name=['softmax1_label', 'softmax2_label'],
        label_width=2,
        # mean_r              = rgb_mean[0],
        # mean_g              = rgb_mean[1],
        # mean_b              = rgb_mean[2],
        # data_name='data',
        # label_name='softmax_label',
        data_shape=[3, 79, 79],
        batch_size=64,
        rand_crop=1,
        max_random_scale=1,

        # max_random_illumination = 0.3,
        # max_random_contrast = 0.3,
        # pad=8,
        # fill_value          = 127,
        min_random_scale=1,
        # max_aspect_ratio    = args.max_random_aspect_ratio,
        random_h=0,
        random_s=0,
        random_l=0,
        max_rotate_angle=60,
        rand_mirror=1,
        preprocess_threads=12,
        shuffle=True,
        num_parts=1,
        part_index=0,
    )

    val = mx.io.ImageRecordIter(
        path_imgrec='/home/research/data/linmingan/multiTask/faceM_val.rec',
        # path_imglist='/home/research/data/linmingan/hair/hair_val.lst',
        label_name=['softmax1_label', 'softmax2_label'],
        data_name='data',
        label_width=2,
        # mean_r              = rgb_mean[0],
        # mean_g              = rgb_mean[1],
        # mean_b              = rgb_mean[2],
        # data_name='data',
        # label_name='softmax_label',
        batch_size=64,
        data_shape=[3, 79, 79],
        preprocess_threads=12,
        rand_crop=False,
        rand_mirror=False,
        num_parts=1,
        part_index=0,
    )

    train = MultiTask_iter(train)
    val = MultiTask_iter(val)
    batch = train.next()
    data = batch.data[0]
    l = batch.label[1].asnumpy()
    for i in range(64):
        d = data[i].asnumpy()
        # d *=255.0

        plt.imsave('da/' + str(i) + '.png', d.astype(np.uint8).transpose((1, 2, 0)))

    # sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    # new_sym = get_fine_tune_model(
    # sym, args.num_classes, 'flatten', args.batch_size)
    new_sym = get_symbol(10)
    # mx.viz.plot_network(new_sym, shape={"data": (32, 3, 79, 79)}).view()
    loadStep = None
    arg_params = None
    aux_params = None
    lr, lr_scheduler = _get_lr_scheduler(0.001, 0.1, '30000', loadStep)
    optimizer_params = {
        'learning_rate': lr,
        'momentum': args.mom,
        'wd': args.wd,
        'lr_scheduler': lr_scheduler
    }
    initializer = mx.init.Xavier(factor_type="in", magnitude=2.34)

    if gpus == '':
        devs = mx.cpu()
    else:
        devs = [mx.gpu(int(i)) for i in gpus.split(',')]
    saveroot = args.save_result + '/' + args.save_name

    if loadStep != None:
        new_sym, arg_params, aux_params = mx.model.load_checkpoint(
            saveroot, loadStep)

    input_shapes = {"data":(64,3,79,79),"softmax1_label":(64,),"softmax2_label":(64,)}
    ex = new_sym.simple_bind(ctx=mx.gpu(0),grad_req='write',**input_shapes)

    for pname, W, G in zip(new_sym.list_arguments(), ex.arg_arrays, ex.grad_arrays):
        G[:]=0.0
    for r in ex.arg_arrays:
        r[:] = np.random.randn(*r.shape) * 0.01
    for epoch in range(10):
        batch = train.next()
        ex.arg_dict['data'][:]=batch.data[0]
        ex.arg_dict['softmax1_label'][:]=batch.label[0]
        ex.arg_dict['softmax2_label'][:] = batch.label[1]
        ex.forward(is_train=True)

        ex.backward()
        for pname, W, G in zip(new_sym.list_arguments(), ex.arg_arrays, ex.grad_arrays):
            # Don't update inputs
            # MXNet makes no distinction between weights and data.
            if pname in ['data', 'softmax1_label','softmax2_label']:
                continue

            # what ever fancy update to modify the parameters
            W[:] = W - G * .001


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    # parser.add_argument('--model',         type=str, required=True,)
    parser.add_argument('--gpus', type=str, default='1,2,3')
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--image-shape', type=str, default='3,79,79')
    parser.add_argument('--data-train', type=str)
    parser.add_argument('--image-train', type=str)
    parser.add_argument('--data-val', type=str)
    parser.add_argument('--image-val', type=str)
    parser.add_argument('--num-classes', type=int)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--num-epoch', type=int, default=2)
    parser.add_argument('--kv-store', type=str, default='device', help='the kvstore type')
    parser.add_argument('--save-result', type=str, help='the save path')
    parser.add_argument('--num-examples', type=int, default=20000)
    parser.add_argument('--mom', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--wd', type=float, default=0.0005, help='weight decay for sgd')
    parser.add_argument('--save-name', type=str, help='the save name of model')
    args = parser.parse_args()

    if not os.path.exists(args.save_result):
        os.mkdir(args.save_result)

    train_model(model=None, gpus=args.gpus, batch_size=args.batch_size,
                image_shape='3,79,79', epoch=args.epoch, num_epoch=args.num_epoch)
