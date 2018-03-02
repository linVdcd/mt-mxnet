import mxnet as mx
import cv2
import os
sym, arg_params, aux_params = mx.model.load_checkpoint('model/resnet-50', 500)
mod = mx.mod.Module(symbol=sym, context=mx.gpu(0), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,75,75))],
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)

labels = [['no_hair','have_hair'],['have_glasses','no_glasses'],['female','male'],['bow','front','top'],['good','lack','left','right','tool','top']]

import matplotlib.pyplot as plt

import numpy as np
# define a simple data batch
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

def get_image(fname, show=False):
    # download and show the image
    img = cv2.imread(fname)
    imgg = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    showimg = img
    if img is None:
         return None

    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (75, 75))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return showimg,img,imgg

def predict(fname):

    showimg,img ,imgg= get_image(fname, show=True)
    plt.imshow(showimg)
    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()
    npProb = [p.asnumpy()[0] for p in prob]

    l = []
    for i in range(5):
        l.append(labels[i][np.argmax(npProb[i])])


    return l

root = '/home/research/data/linmingan/multiTask/crop/test'

for path,dir,files in os.walk(root):
    if len(files)==0:
        continue
    for f in files:
        print f
        c = predict(path+'/'+f)

