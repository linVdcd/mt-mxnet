import argparse
import mxnet as mx
import logging
from resnet import *
from metric import *
logging.basicConfig(level=logging.DEBUG)

class MultiTask_iter(mx.io.DataIter):
    def __init__(self, data_iter):
        super(MultiTask_iter,self).__init__('multitask_iter')
        self.data_iter = data_iter
        self.batch_size = self.data_iter.batch_size

    @property
    def provide_data(self):
        return self.data_iter.provide_data

    @property
    def provide_label(self):
        provide_label = self.data_iter.provide_label[0]
        # the name of the label if corresponding to the model you define in get_fine_tune_model() function
        return [('softmax1_label', [provide_label[1][0]]),('softmax2_label', [provide_label[1][0]]),('softmax3_label', [provide_label[1][0]])]

    def hard_reset(self):
        self.data_iter.hard_reset()

    def reset(self):
        self.data_iter.reset()

    def next(self):
        batch = self.data_iter.next()

        label = batch.label[0]
        ll = label.asnumpy()
        label1 = mx.nd.array(ll[:,0]).astype('float32')
        label2 = mx.nd.array(ll[:,1]).astype('float32')
        label3 = mx.nd.array(ll[:,2]).astype('float32')
        # we set task 2 as: if label>0 or not

        return mx.io.DataBatch(data=batch.data, label=[label1,label2,label3], \
                pad=batch.pad, index=batch.index)

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
                self.sum_metric += (pred_label.flat == label.flat).sum()
                self.num_inst += len(pred_label.flat)
            else:
                self.sum_metric[i] += (pred_label.flat == label.flat).sum()
                self.num_inst[i] += len(pred_label.flat)


def train_model(model, gpus, batch_size, image_shape, epoch=0, num_epoch=20, kv='device'):
    train = mx.io.ImageRecordIter(
        path_imgrec='/home/research/data/linmingan/hair/hair_train.rec',
        #path_imglist='/home/research/data/linmingan/hair/hair_train.lst',
        label_name=['softmax1_label', 'softmax2_label', 'softmax3_label'],
        label_width=3,
        # mean_r              = rgb_mean[0],
        # mean_g              = rgb_mean[1],
        # mean_b              = rgb_mean[2],
        #data_name='data',
        # label_name='softmax_label',
        data_shape=[3,224,224],
        batch_size=256,
        rand_crop=1,
        # max_random_scale=args.max_random_scale,

        # max_random_illumination = 0.3,
        # max_random_contrast = 0.3,
        pad=0,
        # fill_value          = 127,
        # min_random_scale=args.min_random_scale,
        # max_aspect_ratio    = args.max_random_aspect_ratio,
        random_h=100,
        random_s=100,
        random_l=100,
        max_rotate_angle=0,
        max_shear_ratio     = 30,
        rand_mirror=1,
        preprocess_threads=8,
        shuffle=True,
        #num_parts=1,
        #part_index=,
    )

    val = mx.io.ImageRecordIter(
        path_imgrec='/home/research/data/linmingan/hair/hair_val.rec',
        #path_imglist='/home/research/data/linmingan/hair/hair_val.lst',
        label_name=['softmax1_label', 'softmax2_label', 'softmax3_label'],
        label_width=3,
        # mean_r              = rgb_mean[0],
        # mean_g              = rgb_mean[1],
        # mean_b              = rgb_mean[2],
        #data_name='data',
        # label_name='softmax_label',
        batch_size=256,
        data_shape=[3,224,224],
        preprocess_threads=4,
        rand_crop=False,
        rand_mirror=False,
        #num_parts=1,
        #part_index=0,
    )
    '''
    train = mx.image.ImageIter(
        batch_size          = args.batch_size,
        data_shape          = (3,224,224),        
        label_width         = 1,
        path_imglist        = args.data_train,
        path_root           = args.image_train,
        part_index          = kv.rank,
        num_parts           = kv.num_workers,
        shuffle             = True,
        data_name           = 'data',
        aug_list            = mx.image.CreateAugmenter((3,224,224),resize=224,rand_crop=True,rand_mirror=True,mean = True,std = True))
    
    val = mx.image.ImageIter(
        batch_size          = args.batch_size,
        data_shape          = (3,224,224),
        label_width         = 1,
        path_imglist        = args.data_val,
        path_root           = args.image_val,
        part_index          = kv.rank,
        num_parts           = kv.num_workers,       
        data_name           = 'data',
        aug_list            = mx.image.CreateAugmenter((3,224,224),resize=224, mean = True,std = True))
    '''
    train = MultiTask_iter(train)
    val = MultiTask_iter(val)




    #sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    #new_sym = get_fine_tune_model(
        #sym, args.num_classes, 'flatten', args.batch_size)
    new_sym = get_symbol(10,50,image_shape)

    optimizer_params = {
            'learning_rate': 0.001,
            'momentum' : args.mom,
            'wd' : args.wd,
           }
    initializer = mx.init.Xavier(factor_type="in", magnitude=2.34)

    if gpus == '':
        devs = mx.cpu()
    else:
        devs = [mx.gpu(int(i)) for i in gpus.split(',')]

    model = mx.mod.Module(
        context       = devs,
        symbol        = new_sym,
        data_names=['data'],
        label_names=['softmax1_label','softmax2_label','softmax3_label']
    )
    saveroot = args.save_result+'/' + args.save_name
    checkpoint = mx.callback.do_checkpoint(saveroot)

    #eval_metric = mx.metric.CompositeEvalMetric()
    #eval_metric.add(Multi_Accuracy(num=3,output_names=['softmax1','softmax2','softmax3']))

    model.fit(train,
              begin_epoch=0,
              num_epoch=100000,
              eval_data=val,
              eval_metric=Multi_Accuracy(num=3),
              #validation_metric=eval_metric,
              #kvstore=kv,
              optimizer='sgd',

              optimizer_params=optimizer_params,

              initializer=initializer,
              allow_missing=True,
              batch_end_callback=mx.callback.Speedometer(64, 50),

              epoch_end_callback=checkpoint
              )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    #parser.add_argument('--model',         type=str, required=True,)
    parser.add_argument('--gpus',          type=str, default='0')
    parser.add_argument('--batch-size',    type=int, default=200)
    parser.add_argument('--epoch',         type=int, default=0)
    parser.add_argument('--image-shape',   type=str, default='3,224,224')
    parser.add_argument('--data-train',    type=str)
    parser.add_argument('--image-train',   type=str)
    parser.add_argument('--data-val',      type=str)
    parser.add_argument('--image-val',     type=str)
    parser.add_argument('--num-classes',   type=int)
    parser.add_argument('--lr',            type=float, default=0.001)
    parser.add_argument('--num-epoch',     type=int, default=2)
    parser.add_argument('--kv-store',      type=str, default='device', help='the kvstore type')
    parser.add_argument('--save-result',   type=str, help='the save path')
    parser.add_argument('--num-examples',  type=int, default=20000)
    parser.add_argument('--mom',           type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--wd',            type=float, default=0.0001, help='weight decay for sgd')
    parser.add_argument('--save-name',     type=str, help='the save name of model')
    args = parser.parse_args()





    if not os.path.exists(args.save_result):
        os.mkdir(args.save_result)



    train_model(model=None, gpus=args.gpus, batch_size=args.batch_size,
          image_shape='3,224,224', epoch=args.epoch, num_epoch=args.num_epoch)
