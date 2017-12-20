from __future__ import division, print_function
import numpy as np
from utils import plots
from vgg16 import Vgg16

path = "data/dogscats/"
np.set_printoptions(precision=4, linewidth=100)
batch_size = 64
vgg = Vgg16()
batches = vgg.get_batches(path + 'train', batch_size=batch_size)
val_batches = vgg.get_batches(path + 'valid', batch_size=batch_size * 2)
batches = vgg.get_batches(path + 'train', batch_size=4)
imgs, labels = next(batches)
plots(imgs, titles=labels)
